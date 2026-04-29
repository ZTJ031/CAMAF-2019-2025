# CAMAF-2019-2025

Processing code for the **China Annual Marine Aquaculture Facility Dataset from 2019 to 2025 (CAMAF-2019-2025)**.

This repository provides the main code used to generate annual marine aquaculture facility maps along the coast of China from 2019 to 2025. The workflow includes Google Earth Engine preprocessing, TransUNet model training and inference, and spatial post-processing.

## Dataset

CAMAF-2019-2025 contains annual vector maps of marine aquaculture facilities along the coast of China from 2019 to 2025.

The dataset includes three aquaculture facility classes:

| Class ID | Class name | Abbreviation |
|---|---|---|
| 1 | Northern Rafts | N-fra |
| 2 | Cages | CA |
| 3 | Southern Rafts | S-fra |

The released vector dataset is available from Figshare:

https://doi.org/10.6084/m9.figshare.32124859

## Repository structure

```text
CAMAF-2019-2025/
├── gee_preprocessing/
│   ├── README.md
│   └── gee_preprocessing.js
├── model_training_inference/
│   ├── README.md
│   ├── config.yaml
│   └── [training and inference scripts]
├── postprocessing/
│   ├── README.md
│   └── postprocess_refine_vectorize.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Workflow

The processing workflow consists of three main steps:

1. **GEE preprocessing**  
   Sentinel-1 and Sentinel-2 imagery are processed in Google Earth Engine. Annual February composites are generated, NDWI and MNDWI are calculated, and an 18-band candidate feature image is exported.

2. **Model training and inference**  
   Six channels are selected from the 18-band candidate feature stack as model inputs: B2, B3, B4, B8, NDWI, and VV. A TransUNet semantic segmentation model is trained and applied to annual feature composites using sliding-window inference.

3. **Post-processing and vectorization**  
   The annual raster classification maps are refined using coastline-based land masking, class-specific connected-component filtering, hole filling, and morphological processing. The refined raster maps are then converted into vector polygons.

## Code folders

### gee_preprocessing/

This folder contains the Google Earth Engine script for Sentinel-1 and Sentinel-2 preprocessing and 18-band candidate feature export.

### model_training_inference/

This folder contains the model training and sliding-window inference code. The config.yaml file records the main model, training, and inference parameters used in the manuscript.

### postprocessing/

This folder contains the post-processing script used to refine annual raster classification maps and convert them into vector polygons.

## Input data

The main input data include:

- Sentinel-2 surface reflectance imagery from COPERNICUS/S2_SR_HARMONIZED
- Sentinel-1 GRD imagery from COPERNICUS/S1_GRD
- Cloud Score+ data from GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED
- Manually prepared training labels
- Coastline-derived land or sea polygon mask

The full training, validation, and test patches are intermediate products and are not included in this repository. The sample preparation procedure, data-split strategy, and validation design are described in the manuscript.

## Environment

The Python scripts were tested with:

- Python 3.10
- PyTorch 2.0.2
- Torchvision
- NumPy
- Rasterio
- GeoPandas
- SciPy
- scikit-image
- tqdm

Users can install the required Python packages using:

pip install -r requirements.txt

## Usage

Users should modify the input paths, output paths, year settings, model checkpoint paths, and coastline vector paths according to their local environment before running the scripts.

A general execution order is:

1. Run the GEE preprocessing script to export annual 18-band feature images.
2. Select the six production channels: B2, B3, B4, B8, NDWI, and VV.
3. Train the TransUNet model or use the trained model for annual inference.
4. Apply post-processing and vectorization to generate refined vector outputs.

## Notes

The code in this repository is intended to document and support the main dataset production workflow. Users may need to adapt file paths, data organization, and model checkpoint settings according to their own computing environment.

The full training, validation, and test image patches are not included because they are intermediate products derived from Sentinel-1/2 imagery and manual interpretation. The training sample preparation procedure is described in the associated manuscript.

## License

This code is released under the MIT License.

## Citation

If you use this code or the CAMAF-2019-2025 dataset, please cite the associated Data Descriptor and the Figshare data record.

The formal manuscript citation will be added after publication.

Dataset DOI: https://doi.org/10.6084/m9.figshare.32124859
