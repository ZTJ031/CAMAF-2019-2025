import datetime
import os
from functools import partial
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.transunet import TransUnet
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader_tif2 import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit2 import fit_one_epoch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def infinite_mixed_gen(dl_sfra, dl_other):
    """
    从两个 DataLoader 中取 batch，按固定数量拼成一个大 batch，
    并在 batch 内随机打乱。
    """
    it_s, it_o = iter(dl_sfra), iter(dl_other)
    while True:
        try:
            imgs_s, pngs_s, labels_s = next(it_s)
        except StopIteration:
            it_s = iter(dl_sfra)
            imgs_s, pngs_s, labels_s = next(it_s)

        try:
            imgs_o, pngs_o, labels_o = next(it_o)
        except StopIteration:
            it_o = iter(dl_other)
            imgs_o, pngs_o, labels_o = next(it_o)

        imgs = torch.cat([imgs_s, imgs_o], dim=0)
        pngs = torch.cat([pngs_s, pngs_o], dim=0)
        labels = torch.cat([labels_s, labels_o], dim=0)

        idx = torch.randperm(imgs.size(0))
        yield imgs[idx], pngs[idx], labels[idx]


if __name__ == "__main__":
    # ---------------- 基本配置 ---------------- #
    Cuda            = True
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = False

    num_classes     = 4
    backbone        = "resnet50"
    pretrained      = False
    model_path      = ""

    input_shape     = [256, 256]

    Init_Epoch          = 0
    Freeze_Epoch        = 0
    Freeze_batch_size   = 8
    UnFreeze_Epoch      = 70
    Unfreeze_batch_size = 12
    Freeze_Train        = False

    Init_lr         = 8e-5
    Min_lr          = Init_lr * 0.01
    optimizer_type  = "adam"
    momentum        = 0.9
    weight_decay    = 0
    lr_decay_type   = 'cos'

    save_period     = 5
    save_dir        = 'logs'
    eval_flag       = True
    eval_period     = 5

    VOCdevkit_path  = 'VOCdevkit'

    dice_loss       = False
    focal_loss      = False
    #cls_weights     = np.array([0.00527, 0.04346, 0.41585, 1], np.float32)
    cls_weights = np.ones([num_classes], np.float32)
    num_workers     = 4

    # 固定比例：每个 batch 中 S-fra patch 数量占比
    # 当前 batch_size=12 -> 7 张 sfra + 5 张 other
    ratio_sfra      = 7 / 12

    # ---------------- 环境 & 模型 ---------------- #
    seed_everything(seed)
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank  = 0
        rank        = 0

    # 预训练权重下载（如果需要）
    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(backbone)
            dist.barrier()
        else:
            download_weights(backbone)

    # 构建模型
    model = TransUnet(
        img_dim=256,
        in_channels=6,
        out_channels=128,
        head_num=4,
        mlp_dim=512,
        block_num=8,
        patch_dim=16,
        class_num=4,
    ).train()

    if not pretrained:
        weights_init(model)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500],
                  "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500],
                  "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # Loss 记录
    if local_rank == 0:
        time_str    = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir     = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # amp
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------- 读取 txt 列表 ---------------- #
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train_sfra.txt"), "r") as f:
        train_sfra_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train_other.txt"), "r") as f:
        train_other_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val2.txt"), "r") as f:
        val_lines = f.readlines()

    num_train = len(train_sfra_lines) + len(train_other_lines)
    num_val   = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
            Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir,
            num_workers=num_workers, num_train=num_train, num_val=num_val
        )

    # -------- 构建 dataloader 的小函数（会在冻结/解冻阶段重复调用） -------- #
    def build_dataloaders(batch_size):
        # 两个训练子集
        train_dataset_sfra  = UnetDataset(train_sfra_lines,  input_shape, num_classes, True,  VOCdevkit_path)
        train_dataset_other = UnetDataset(train_other_lines, input_shape, num_classes, True,  VOCdevkit_path)
        val_dataset         = UnetDataset(val_lines,         input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler_sfra  = torch.utils.data.distributed.DistributedSampler(train_dataset_sfra,  shuffle=True)
            train_sampler_other = torch.utils.data.distributed.DistributedSampler(train_dataset_other, shuffle=True)
            val_sampler         = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
            real_batch_size     = batch_size // ngpus_per_node
            shuffle_flag        = False
        else:
            train_sampler_sfra  = None
            train_sampler_other = None
            val_sampler         = None
            real_batch_size     = batch_size
            shuffle_flag        = True

        # 按比例分配 sfra/other 的 batch 大小
        bs_sfra = int(real_batch_size * ratio_sfra)
        bs_sfra = max(1, min(bs_sfra, real_batch_size - 1))
        bs_other = real_batch_size - bs_sfra

        dl_sfra = DataLoader(
            train_dataset_sfra, shuffle=shuffle_flag, batch_size=bs_sfra,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            collate_fn=unet_dataset_collate, sampler=train_sampler_sfra,
            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
        )
        dl_other = DataLoader(
            train_dataset_other, shuffle=shuffle_flag, batch_size=bs_other,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            collate_fn=unet_dataset_collate, sampler=train_sampler_other,
            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
        )

        gen = infinite_mixed_gen(dl_sfra, dl_other)
        # 以 other 这一路的长度作为 epoch_step（每个 epoch 基本遍历一遍 other）
        epoch_step = len(dl_other)

        gen_val = DataLoader(
            val_dataset, shuffle=shuffle_flag, batch_size=real_batch_size,
            num_workers=num_workers, pin_memory=True, drop_last=True,
            collate_fn=unet_dataset_collate, sampler=val_sampler,
            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed)
        )
        epoch_step_val = num_val // real_batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        return gen, gen_val, epoch_step, epoch_step_val, \
            train_sampler_sfra, train_sampler_other, val_sampler

    # ---------------- 正式训练 ---------------- #
    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            model.freeze_backbone()

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        # 自适应学习率
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum,
                                nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # 初始 dataloader
        gen, gen_val, epoch_step, epoch_step_val, \
            train_sampler_sfra, train_sampler_other, val_sampler = build_dataloaders(batch_size)

        # eval 回调
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines,
                                         VOCdevkit_path, log_dir, Cuda,
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None

        # 训练循环
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            # 冻结 -> 解冻阶段切换时重建 dataloader
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                model.unfreeze_backbone()

                gen, gen_val, epoch_step, epoch_step_val, \
                    train_sampler_sfra, train_sampler_other, val_sampler = build_dataloaders(batch_size)

                UnFreeze_flag = True

            if distributed:
                if train_sampler_sfra is not None:
                    train_sampler_sfra.set_epoch(epoch)
                if train_sampler_other is not None:
                    train_sampler_other.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(
                model_train, model, loss_history, eval_callback, optimizer, epoch,
                epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch,
                Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16,
                scaler, save_period, save_dir, local_rank
            )

            if distributed:
                dist.barrier()
        if local_rank == 0 and loss_history is not None:
            loss_history.writer.close()
