# -*- coding: utf-8 -*-
"""
任意尺寸 tif -> 256滑窗推理(TransUnet) -> 拼回原尺寸 -> 输出单波段 mask.tif (0..K-1)

✅ 关键：推理预处理完全对齐训练 Dataset：
    img_chw = np.transpose(preprocess_input(np.array(img_hwc, np.float64)), [2,0,1])

依赖：
  pip install torch rasterio numpy tqdm einops
并且你的工程里需存在：
  from nets.vit import ViT
  from utils.utils import preprocess_input

输出：
  单波段 uint8，像元值为 0..CLASS_NUM-1
  可选把无效像元写成 255 (nodata)
"""

import os
import glob
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import rasterio
from tqdm import tqdm

import torch
import torch.nn as nn
from einops import rearrange

from nets.vit import ViT
from utils.utils import preprocess_input


# =========================================================
# ✅ 你只需要改这里：输入/输出/权重路径
# =========================================================
INPUT_DIR    = r""
OUTPUT_DIR   = r""
WEIGHTS_PATH = r""

DEVICE = "cuda"      # "cuda" or "cpu"
AMP = True           # cuda下可True；cpu建议False
RECURSIVE = False    # 是否递归子文件夹

# 滑窗参数
TILE_SIZE = 256      # 训练 input_shape=256，推理也必须喂256给模型
OVERLAP   = 64       # 0/32/64；有拼接缝就用64（更稳但更慢）

# 输出设置
OUT_NODATA = 255     # 不想写nodata就设 None
SUFFIX = "_mask"

# 仅打印第一张的诊断信息
DEBUG_FIRST_IMAGE = True
# =========================================================


# =========================================================
# 模型超参（必须与训练一致）
# 你训练脚本里是：out_channels=128, head_num=4, mlp_dim=512, block_num=8, patch_dim=16, class_num=4
# 最关键：in_channels 必须与训练一致（你训练脚本里写的是 6）
# 这里我们会从权重里自动推断 in_channels / class_num / out_channels（更稳）
# =========================================================
HEAD_NUM  = 4
MLP_DIM   = 512
BLOCK_NUM = 8
PATCH_DIM = 16


# =========================
# 你的 TransUnet 结构（与训练保持一致）
# =========================
class EncoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, base_width=64):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        width = int(out_channels * (base_width / 64))
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        self.norm1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.norm2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, stride=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_down = self.downsample(x)
        x = self.conv1(x); x = self.norm1(x); x = self.relu(x)
        x = self.conv2(x); x = self.norm2(x); x = self.relu(x)
        x = self.conv3(x); x = self.norm3(x); x = self.relu(x)
        x = x + x_down
        x = self.relu(x)
        return x


class DecoderBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_concat=None):
        x = self.upsample(x)
        if x_concat is not None:
            x = torch.cat([x_concat, x], dim=1)
        x = self.layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.encoder1 = EncoderBottleneck(out_channels, out_channels * 2, stride=2)
        self.encoder2 = EncoderBottleneck(out_channels * 2, out_channels * 4, stride=2)
        self.encoder3 = EncoderBottleneck(out_channels * 4, out_channels * 8, stride=2)

        self.vit_img_dim = img_dim // patch_dim
        self.vit = ViT(
            img_dim=self.vit_img_dim,
            in_channels=out_channels * 8,
            embedding_dim=out_channels * 8,
            head_num=head_num,
            mlp_dim=mlp_dim,
            block_num=block_num,
            patch_dim=1,
            classification=False
        )

        self.conv2 = nn.Conv2d(out_channels * 8, 512, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.conv1(x); x = self.norm1(x); x1 = self.relu(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x  = self.encoder3(x3)

        x = self.vit(x)
        x = rearrange(x, "b (x y) c -> b c x y", x=self.vit_img_dim, y=self.vit_img_dim)

        x = self.conv2(x); x = self.norm2(x); x = self.relu(x)
        return x, x1, x2, x3


class Decoder(nn.Module):
    def __init__(self, out_channels, class_num):
        super().__init__()
        self.decoder1 = DecoderBottleneck(out_channels * 8, out_channels * 2)
        self.decoder2 = DecoderBottleneck(out_channels * 4, out_channels)
        self.decoder3 = DecoderBottleneck(out_channels * 2, int(out_channels * 1 / 2))
        self.decoder4 = DecoderBottleneck(int(out_channels * 1 / 2), int(out_channels * 1 / 8))
        self.conv1 = nn.Conv2d(int(out_channels * 1 / 8), class_num, kernel_size=1)

    def forward(self, x, x1, x2, x3):
        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x)
        x = self.conv1(x)
        return x


class TransUnet(nn.Module):
    def __init__(self, img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim, class_num):
        super().__init__()
        self.encoder = Encoder(img_dim, in_channels, out_channels, head_num, mlp_dim, block_num, patch_dim)
        self.decoder = Decoder(out_channels, class_num)

    def forward(self, x):
        x, x1, x2, x3 = self.encoder(x)
        x = self.decoder(x, x1, x2, x3)
        return x


# =========================
# 工具函数
# =========================
def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def smart_load_state_dict(weights_path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(weights_path, map_location="cpu")
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ["state_dict", "model", "model_state_dict"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
        return obj
    raise ValueError("不支持的权重格式。请确认保存的是 state_dict 或 checkpoint(dict)。")


def infer_model_params_from_state(state: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """
    从权重推断 (in_channels, out_channels, class_num)
    - in/out_channels：encoder.conv1.weight -> [out_ch, in_ch, 7,7]
    - class_num：decoder.conv1.weight -> [class_num, ?, 1,1]
    """
    conv1_key = "encoder.conv1.weight"
    if conv1_key not in state:
        # 兼容可能的命名差异：找到以 encoder.conv1.weight 结尾的
        cand = [k for k in state.keys() if k.endswith("encoder.conv1.weight")]
        if not cand:
            raise KeyError("找不到 encoder.conv1.weight，无法从权重推断通道数。")
        conv1_key = cand[0]

    dec_key = "decoder.conv1.weight"
    if dec_key not in state:
        cand = [k for k in state.keys() if k.endswith("decoder.conv1.weight")]
        if not cand:
            raise KeyError("找不到 decoder.conv1.weight，无法从权重推断类别数。")
        dec_key = cand[0]

    w1 = state[conv1_key]   # [out_ch, in_ch, 7,7]
    w2 = state[dec_key]     # [class_num, ?, 1,1]
    out_ch = int(w1.shape[0])
    in_ch  = int(w1.shape[1])
    cls    = int(w2.shape[0])
    return in_ch, out_ch, cls


def read_tif_chw(path: str) -> Tuple[np.ndarray, Dict[str, Any], Optional[np.ndarray]]:
    """
    用 rasterio 读 tif 为 (C,H,W) float32，返回 profile 和 valid_mask(如果可读到)
    """
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        arr = src.read().astype(np.float32)  # (C,H,W)
        try:
            m = src.read_masks(1)
            valid_mask = (m > 0)
        except Exception:
            valid_mask = None
    return arr, profile, valid_mask


def write_mask_geotiff(out_path: str, mask_hw: np.ndarray, ref_profile: Dict[str, Any], nodata: Optional[int]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    prof = ref_profile.copy()
    prof.update(count=1, dtype=rasterio.uint8, compress="deflate", nodata=nodata)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(mask_hw.astype(np.uint8), 1)


def get_start_positions(length: int, tile: int, stride: int) -> List[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile + 1, stride))
    last = length - tile
    if pos[-1] != last:
        pos.append(last)
    return pos


def preprocess_tile_like_training(tile_chw: np.ndarray) -> np.ndarray:
    """
    完全对齐训练 Dataset 的写法：
      img_chw = np.transpose(preprocess_input(np.array(img_hwc, np.float64)), [2,0,1])
    """
    # tile_chw: (C,H,W) -> (H,W,C)
    tile_hwc = np.transpose(tile_chw, (1, 2, 0)).astype(np.float64)
    tile_hwc = preprocess_input(np.array(tile_hwc, np.float64))
    tile_chw_out = np.transpose(tile_hwc, (2, 0, 1)).astype(np.float32)
    return tile_chw_out


@torch.no_grad()
def predict_tile_logits(model: nn.Module, tile_chw_raw: np.ndarray, device: str, amp: bool) -> np.ndarray:
    """
    输入 raw tile (C,256,256) -> 先按训练 preprocess_input -> 再推理 -> 输出 logits (K,256,256)
    """
    tile_chw = preprocess_tile_like_training(tile_chw_raw)
    x = torch.from_numpy(tile_chw).unsqueeze(0).to(device)  # (1,C,H,W)

    if amp and device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model(x)
    else:
        y = model(x)

    return y[0].detach().float().cpu().numpy()  # (K,H,W)


def sliding_window_predict(
    model: nn.Module,
    x_chw_raw: np.ndarray,
    device: str,
    amp: bool,
    tile_size: int,
    overlap: int,
    class_num: int
) -> np.ndarray:
    """
    对任意尺寸 raw 输入 (C,H,W) 做滑窗推理，输出 (H,W) 类别ID
    用 logits 平均融合，减少拼接缝
    """
    C, H, W = x_chw_raw.shape
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("OVERLAP 必须小于 TILE_SIZE")

    ys = get_start_positions(H, tile_size, stride)
    xs = get_start_positions(W, tile_size, stride)

    logits_sum = np.zeros((class_num, H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + tile_size, x0 + tile_size

            tile = x_chw_raw[:, y0:min(y1, H), x0:min(x1, W)]
            th, tw = tile.shape[1], tile.shape[2]

            # padding 到 256×256
            if th != tile_size or tw != tile_size:
                pad = np.zeros((C, tile_size, tile_size), dtype=tile.dtype)
                pad[:, :th, :tw] = tile
                tile = pad

            logits_tile = predict_tile_logits(model, tile, device=device, amp=amp)  # (K,256,256)

            vh = min(tile_size, H - y0)
            vw = min(tile_size, W - x0)
            logits_sum[:, y0:y0+vh, x0:x0+vw] += logits_tile[:, :vh, :vw]
            weight_sum[y0:y0+vh, x0:x0+vw] += 1.0

    logits_avg = logits_sum / np.maximum(weight_sum[None, ...], 1e-6)
    pred = np.argmax(logits_avg, axis=0).astype(np.uint8)
    return pred


def main():
    device = DEVICE
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA不可用，自动切到CPU")
        device = "cpu"

    pattern = "**/*.tif" if RECURSIVE else "*.tif"
    tif_list = sorted(glob.glob(os.path.join(INPUT_DIR, pattern), recursive=RECURSIVE))
    if not tif_list:
        raise FileNotFoundError(f"在 {INPUT_DIR} 没找到 tif")

    # 读权重并推断通道/类别数
    state = strip_module_prefix(smart_load_state_dict(WEIGHTS_PATH))
    in_ch, out_ch, class_num = infer_model_params_from_state(state)
    print(f"[INFO] 从权重推断：in_channels={in_ch}, out_channels={out_ch}, class_num={class_num}")

    # 构建模型（img_dim 固定 256，因为模型本身是按 256 设计的；大图通过滑窗实现）
    model = TransUnet(
        img_dim=256,
        in_channels=in_ch,
        out_channels=out_ch,
        head_num=HEAD_NUM,
        mlp_dim=MLP_DIM,
        block_num=BLOCK_NUM,
        patch_dim=PATCH_DIM,
        class_num=class_num,
    )

    # ✅ 严格加载：不匹配直接报错，避免“没加载成功导致全0”
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print("[OK] 权重 strict=True 加载成功 ✅")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    debug_once = DEBUG_FIRST_IMAGE

    for tif_path in tqdm(tif_list, desc="Predict any-size masks"):
        x_chw_raw, profile, valid_mask = read_tif_chw(tif_path)  # (C,H,W)

        # 通道一致性检查
        if x_chw_raw.shape[0] != in_ch:
            raise ValueError(
                f"{os.path.basename(tif_path)} 读取到通道数={x_chw_raw.shape[0]}，"
                f"但权重期望 in_channels={in_ch}。\n"
                f"请确保推理 tif 与训练 tif 的波段数一致（训练 Dataset 是直接读 tif 的所有波段）。"
            )

        if debug_once:
            C, H, W = x_chw_raw.shape
            print("\n[DEBUG] file:", os.path.basename(tif_path))
            print("[DEBUG] raw shape(C,H,W):", x_chw_raw.shape, "dtype:", x_chw_raw.dtype)
            for c in range(min(C, 10)):
                b = x_chw_raw[c]
                print(f"[DEBUG] raw band{c}: min={b.min():.3f} max={b.max():.3f} mean={b.mean():.3f}")

        # 滑窗推理（tile 内部会按训练 preprocess_input 处理）
        mask = sliding_window_predict(
            model=model,
            x_chw_raw=x_chw_raw,
            device=device,
            amp=AMP,
            tile_size=TILE_SIZE,
            overlap=OVERLAP,
            class_num=class_num
        )

        # nodata（可选）
        if valid_mask is not None and OUT_NODATA is not None:
            mask = mask.copy()
            mask[~valid_mask] = np.uint8(OUT_NODATA)

        if debug_once:
            uniq, cnt = np.unique(mask, return_counts=True)
            print("[DEBUG] pred unique:", dict(zip(uniq.tolist(), cnt.tolist())))
            debug_once = False

        # 写出（保持原地理参考）
        rel = os.path.relpath(tif_path, INPUT_DIR)
        base = os.path.splitext(rel)[0]
        out_path = os.path.join(OUTPUT_DIR, base + SUFFIX + ".tif")
        write_mask_geotiff(out_path, mask, profile, nodata=OUT_NODATA)

    print("Done. 已输出任意尺寸输入的单波段类别ID GeoTIFF。")


if __name__ == "__main__":
    main()

