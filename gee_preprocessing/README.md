# GEE Preprocessing

This folder contains the Google Earth Engine (GEE) script used to generate annual Sentinel-1 and Sentinel-2 feature composites for CAMAF-2019-2025.

## Purpose

The GEE preprocessing script performs the following steps:

1. Loads Sentinel-2 surface reflectance imagery from `COPERNICUS/S2_SR_HARMONIZED`.
2. Filters images by study area and annual February acquisition window.
3. Applies cloud masking using the Cloud Score+ product.
4. Calculates NDWI and MNDWI from Sentinel-2 bands.
5. Loads Sentinel-1 GRD imagery from `COPERNICUS/S1_GRD`.
6. Filters Sentinel-1 imagery by VV polarization, IW mode, and 10-m spatial resolution.
7. Generates annual median Sentinel-2 composites and mean Sentinel-1 VV composites.
8. Constructs an 18-band candidate feature image.
9. Normalizes each feature band to 0–255 and exports the result as GeoTIFF.

## Main parameters

The main parameters that need to be modified before running the script are:

- `startdate`: start date of the annual February image window.
- `enddate`: end date of the annual February image window.
- `geometry`: target coastal region or export region.
- `description`: export task name.
- `region`: export region.
- `scale`: export spatial resolution. The default value is 10 m.

## Output

The output is an annual 18-band GeoTIFF image exported to Google Drive. The 18 bands are:

| Band | Description |
|---|---|
| B1 | Sentinel-2 coastal aerosol band |
| B2 | Sentinel-2 blue band |
| B3 | Sentinel-2 green band |
| B4 | Sentinel-2 red band |
| B5 | Sentinel-2 red-edge band |
| B6 | Sentinel-2 red-edge band |
| B7 | Sentinel-2 red-edge band |
| B8 | Sentinel-2 near-infrared band |
| B8A | Sentinel-2 narrow near-infrared band |
| B9 | Sentinel-2 water vapour band |
| B11 | Sentinel-2 shortwave infrared band |
| B12 | Sentinel-2 shortwave infrared band |
| TCI_R | Sentinel-2 true-colour red band |
| TCI_G | Sentinel-2 true-colour green band |
| TCI_B | Sentinel-2 true-colour blue band |
| NDWI | Normalized Difference Water Index |
| MNDWI | Modified Normalized Difference Water Index |
| VV | Sentinel-1 VV backscatter |

## Notes

The GEE script exports the 18-band candidate feature image. In the final production workflow described in the manuscript, six channels were selected from this candidate feature stack as the model input: B2, B3, B4, B8, NDWI, and VV.

Users should define the target `geometry` in the GEE Code Editor before running the script. For CAMAF-2019-2025, February imagery was used for each mapping year from 2019 to 2025 to maintain seasonal consistency among annual maps.
