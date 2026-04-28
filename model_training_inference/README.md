# Model Training and Inference

This folder contains the model training and sliding-window inference code used in CAMAF-2019-2025.

## Purpose

The scripts in this folder are used for:

1. Training the TransUNet semantic segmentation model.
2. Applying the trained model to annual six-channel feature composites.
3. Producing annual raster classification maps before spatial post-processing.

## Input data

The model uses six input channels selected from the 18-band candidate feature stack exported by the GEE preprocessing script:

| Channel | Description |
|---|---|
| B2 | Sentinel-2 blue band |
| B3 | Sentinel-2 green band |
| B4 | Sentinel-2 red band |
| B8 | Sentinel-2 near-infrared band |
| NDWI | Normalized Difference Water Index |
| VV | Sentinel-1 VV backscatter |

The image patches used for model training have a size of 256 × 256 pixels. The label masks contain four classes:

| Class ID | Class name |
|---|---|
| 0 | Background |
| 1 | Northern Rafts |
| 2 | Cages |
| 3 | Southern Rafts |

## Training settings

The production model was trained using TransUNet with the following main settings:

- Input size: 256 × 256 pixels
- Number of input channels: 6
- Number of output classes: 4
- Batch size: 12
- Training epochs: 70
- Optimizer: Adam
- Initial learning-rate parameter: 8e-5
- Minimum learning-rate parameter: 8e-7
- Effective initial learning rate: 1e-4
- Effective minimum learning rate: 1e-6
- Learning-rate schedule: cosine decay
- Random seed: 11

## Dynamic mixing strategy

During training, Southern Raft samples were oversampled using a batch-level dynamic mixing strategy. Each training batch contained:

- 7 Southern Raft samples
- 5 other samples

The files `train_sfra.txt` and `train_other.txt` are used to define the two training subsets.

## Inference settings

Annual prediction was conducted using a sliding-window strategy:

- Tile size: 256 × 256 pixels
- Overlap: 64 pixels
- Stride: 192 pixels
- Merging method: pixel-wise averaging of class scores in overlapping areas
- Final label assignment: class with the highest averaged score
- Confidence threshold: not applied

## Output

The inference output is a single-band unsigned 8-bit GeoTIFF classification map. Pixel values represent class IDs:

| Pixel value | Description |
|---|---|
| 0 | Background |
| 1 | Northern Rafts |
| 2 | Cages |
| 3 | Southern Rafts |
| 255 | Nodata, when a valid-data mask is available |

The annual raster classification maps are used as input for the spatial post-processing scripts.

## Notes

The full training, validation, and test patches are intermediate products and are not included in this repository. The sample preparation procedure, data augmentation, 8:1:1 data split, and independent validation strategy are described in the manuscript.

Users should modify the input paths, output paths, model checkpoint path, and configuration parameters according to their local environment before running the scripts.
