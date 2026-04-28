# -*- coding: utf-8 -*-
"""
Post-processing for marine aquaculture classification masks.

Input:
  - Single-band prediction mask GeoTIFF
    0 = Background
    1 = Northern Rafts
    2 = Cages
    3 = Southern Rafts
    255 = NoData

  - Coastline-derived polygon vector
    Recommended:
      land polygon mask: polygons represent land areas to remove
    or
      sea polygon mask: polygons represent sea areas to retain

Output:
  - Refined single-band uint8 GeoTIFF
  - Vector polygon Shapefile / GeoPackage

Main steps:
  1. Remove landward false positives using coastline-derived polygon mask
  2. Remove small isolated patches using class-specific thresholds
  3. Fill small internal holes using class-specific thresholds
  4. Apply optional light morphological closing
  5. Export refined raster and vector polygons

The output vector attribute table contains:
  - gridcode
  - class_name
  - year
"""

import os
import glob
import numpy as np
import rasterio
import geopandas as gpd

from rasterio.features import rasterize, shapes
from shapely.geometry import shape
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import binary_closing, binary_dilation
from tqdm import tqdm


# =========================================================
# User settings
# =========================================================

# Mapping year. Change this value for each annual map.
YEAR = 2025

# Input and output paths.
# Users should modify these paths according to their local environment.
INPUT_MASK_DIR = r"/path/to/prediction_masks/2025"
OUTPUT_RASTER_DIR = r"/path/to/postprocess_results/2025_raster"
OUTPUT_VECTOR_DIR = r"/path/to/postprocess_results/2025_vector"

# Coastline-derived polygon vector.
# Recommended input:
#   land polygon mask: polygons represent land areas to remove
# or
#   sea polygon mask: polygons represent sea areas to retain
COAST_VECTOR_PATH = r"/path/to/coastline_or_land_polygon.shp"

# "land" = vector polygons represent land areas to remove
# "sea"  = vector polygons represent sea areas to retain
MASK_VECTOR_MODE = "land"

# If the coastline mask has slight spatial offset, the land mask can be dilated.
# Because marine aquaculture facilities can be close to the coastline,
# 0 is recommended as the default value.
LAND_MASK_DILATE_PIXELS = 0

# Whether to export vector results
EXPORT_VECTOR = True

# Output vector format:
# Use "ESRI Shapefile" to match the released CAMAF-2019-2025 dataset.
# Optional alternative: "GPKG"
VECTOR_FORMAT = "ESRI Shapefile"

# Class definitions
BACKGROUND = 0
NODATA = 255

CLASS_NAMES = {
    1: "N-fra",
    2: "CA",
    3: "S-fra",
}


# =========================================================
# Class-specific post-processing thresholds at 10 m resolution
# 1 pixel = 100 m²
# =========================================================

POSTPROCESS_PARAMS = {
    # Northern Rafts: usually larger and more regular
    1: {
        "min_object_pixels": 30,     # 3,000 m²
        "max_hole_pixels": 100,      # 10,000 m²
        "closing_iterations": 1,
    },

    # Cages: compact but can be smaller than raft areas
    2: {
        "min_object_pixels": 15,     # 1,500 m²
        "max_hole_pixels": 50,       # 5,000 m²
        "closing_iterations": 1,
    },

    # Southern Rafts: more fragmented; conservative threshold
    3: {
        "min_object_pixels": 8,      # 800 m²
        "max_hole_pixels": 30,       # 3,000 m²
        "closing_iterations": 0,
    },
}

# If morphological operations cause label conflicts in rare cases,
# this order is used only for assigning newly filled background pixels.
FILL_PRIORITY = [2, 1, 3]


# =========================================================
# Utility functions
# =========================================================

def read_single_band_mask(mask_path):
    """Read a single-band prediction mask."""
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        shape_hw = (src.height, src.width)

    return arr, profile, transform, crs, shape_hw


def write_refined_mask(out_path, refined, profile):
    """Write refined mask as uint8 GeoTIFF."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(
        count=1,
        dtype=rasterio.uint8,
        compress="deflate",
        nodata=NODATA
    )

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(refined.astype(np.uint8), 1)


def load_and_rasterize_mask_vector(
    vector_path,
    raster_crs,
    raster_transform,
    raster_shape
):
    """
    Rasterize coastline-derived polygon vector to match the prediction raster.

    The vector should be polygon data.
    If only coastline lines are available, they should first be converted
    into land or sea polygon masks.
    """
    gdf = gpd.read_file(vector_path)

    if gdf.empty:
        raise ValueError("Coastline vector is empty.")

    if gdf.crs is None:
        raise ValueError("Coastline vector has no CRS. Please define CRS before processing.")

    gdf = gdf.to_crs(raster_crs)

    geoms = [
        (geom, 1)
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]

    if len(geoms) == 0:
        raise ValueError("No valid geometries found in coastline vector.")

    mask = rasterize(
        geoms,
        out_shape=raster_shape,
        transform=raster_transform,
        fill=0,
        dtype="uint8",
        all_touched=False
    ).astype(bool)

    return mask


def apply_coast_mask(mask, coast_mask, mode="land", land_dilate_pixels=0):
    """
    Remove landward false positives.

    If mode == "land":
      coast_mask=True means land. Target classes on land are set to background.

    If mode == "sea":
      coast_mask=True means sea. Target classes outside sea are set to background.
    """
    if mode not in ["land", "sea"]:
        raise ValueError("MASK_VECTOR_MODE must be either 'land' or 'sea'.")

    refined = mask.copy()
    nodata_mask = refined == NODATA

    if mode == "land":
        land_mask = coast_mask.copy()

        if land_dilate_pixels > 0:
            structure = np.ones((3, 3), dtype=bool)
            for _ in range(land_dilate_pixels):
                land_mask = binary_dilation(land_mask, structure=structure)

        refined[(land_mask) & (~nodata_mask)] = BACKGROUND

    else:
        sea_mask = coast_mask
        refined[(~sea_mask) & (~nodata_mask)] = BACKGROUND

    refined[nodata_mask] = NODATA
    return refined


def clean_class_mask(base_mask, class_id, params):
    """
    Clean one class at a time:
      - remove small objects
      - fill small holes
      - optional light binary closing

    Newly filled or closed pixels are only allowed to expand into
    original background pixels, preventing overwriting of other classes.
    """
    original_class = base_mask == class_id
    original_background = base_mask == BACKGROUND

    # 1. Remove small isolated objects
    cleaned = remove_small_objects(
        original_class,
        min_size=params["min_object_pixels"],
        connectivity=2
    )

    # 2. Fill small internal holes
    filled = remove_small_holes(
        cleaned,
        area_threshold=params["max_hole_pixels"],
        connectivity=2
    )

    # Only allow hole filling into original background, not other classes
    cleaned = cleaned | ((filled & ~cleaned) & original_background)

    # 3. Optional light closing to improve patch continuity
    closing_iterations = params.get("closing_iterations", 0)

    if closing_iterations > 0:
        structure = np.ones((3, 3), dtype=bool)
        closed = cleaned.copy()

        for _ in range(closing_iterations):
            closed = binary_closing(closed, structure=structure)

        # Only allow expansion into original background
        cleaned = cleaned | ((closed & ~cleaned) & original_background)

    return cleaned


def apply_class_specific_cleaning(mask):
    """
    Apply class-specific filtering to all aquaculture classes.
    """
    nodata_mask = mask == NODATA

    processed_masks = {}
    for class_id, params in POSTPROCESS_PARAMS.items():
        processed_masks[class_id] = clean_class_mask(mask, class_id, params)

    refined = np.zeros_like(mask, dtype=np.uint8)
    refined[nodata_mask] = NODATA

    # Keep cleaned pixels that were originally the same class.
    # This preserves original labels and removes only small components.
    for class_id in POSTPROCESS_PARAMS.keys():
        keep_original = processed_masks[class_id] & (mask == class_id)
        refined[keep_original] = class_id

    # Assign newly filled background pixels according to priority.
    for class_id in FILL_PRIORITY:
        add_pixels = (
            processed_masks[class_id]
            & (mask == BACKGROUND)
            & (refined == BACKGROUND)
            & (~nodata_mask)
        )
        refined[add_pixels] = class_id

    refined[nodata_mask] = NODATA
    return refined


def polygonize_mask(mask_path, vector_out_path, year):
    """
    Convert refined raster mask to vector polygons.
    Only class values 1, 2, and 3 are exported.

    Output fields:
      - gridcode
      - class_name
      - year
    """
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

        valid = np.isin(arr, list(CLASS_NAMES.keys()))

        records = []
        for geom, value in shapes(arr, mask=valid, transform=transform):
            value = int(value)

            if value in CLASS_NAMES:
                records.append({
                    "geometry": shape(geom),
                    "gridcode": value,
                    "class_name": CLASS_NAMES[value],
                    "year": int(year)
                })

    if len(records) == 0:
        print(f"[WARN] No target polygons found in {mask_path}")
        return

    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Remove invalid and empty geometries
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]

    # Repair invalid geometries where necessary
    gdf["geometry"] = gdf.geometry.buffer(0)

    os.makedirs(os.path.dirname(vector_out_path), exist_ok=True)

    if VECTOR_FORMAT == "GPKG":
        gdf.to_file(vector_out_path, driver="GPKG")
    elif VECTOR_FORMAT == "ESRI Shapefile":
        gdf.to_file(vector_out_path, driver="ESRI Shapefile", encoding="utf-8")
    else:
        raise ValueError("VECTOR_FORMAT must be 'GPKG' or 'ESRI Shapefile'.")


def process_one_mask(mask_path, coast_vector_path):
    """Process one prediction mask."""
    arr, profile, transform, crs, shape_hw = read_single_band_mask(mask_path)

    coast_mask = load_and_rasterize_mask_vector(
        vector_path=coast_vector_path,
        raster_crs=crs,
        raster_transform=transform,
        raster_shape=shape_hw
    )

    # Step 1: coastline / land mask
    arr = apply_coast_mask(
        mask=arr,
        coast_mask=coast_mask,
        mode=MASK_VECTOR_MODE,
        land_dilate_pixels=LAND_MASK_DILATE_PIXELS
    )

    # Step 2: class-specific small-object removal and hole filling
    refined = apply_class_specific_cleaning(arr)

    base = os.path.splitext(os.path.basename(mask_path))[0]
    out_raster = os.path.join(OUTPUT_RASTER_DIR, base + "_refined.tif")

    write_refined_mask(out_raster, refined, profile)

    if EXPORT_VECTOR:
        if VECTOR_FORMAT == "GPKG":
            out_vector = os.path.join(OUTPUT_VECTOR_DIR, base + "_refined.gpkg")
        else:
            out_vector = os.path.join(OUTPUT_VECTOR_DIR, base + "_refined.shp")

        polygonize_mask(out_raster, out_vector, YEAR)

    return out_raster


def main():
    os.makedirs(OUTPUT_RASTER_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VECTOR_DIR, exist_ok=True)

    mask_list = sorted(glob.glob(os.path.join(INPUT_MASK_DIR, "*.tif")))

    if not mask_list:
        raise FileNotFoundError(f"No tif files found in {INPUT_MASK_DIR}")

    for mask_path in tqdm(mask_list, desc="Post-processing masks"):
        out_raster = process_one_mask(mask_path, COAST_VECTOR_PATH)
        print("[OK]", out_raster)

    print("Done.")


if __name__ == "__main__":
    main()
