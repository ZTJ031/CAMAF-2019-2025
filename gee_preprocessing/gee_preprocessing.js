// =====================================================
// GEE preprocessing script for CAMAF-2019-2025
// Annual Sentinel-1/2 18-band candidate feature export
// =====================================================

// -----------------------------------------------------
// User parameters
// -----------------------------------------------------

// Mapping year.
// Change this value from 2019 to 2025 when exporting annual composites.
var year = 2023;

// February acquisition window.
// Note: filterDate uses an exclusive end date, so March 1 includes the full February window.
var startdate = year + '-02-01';
var enddate = year + '-03-01';

// Users should define or import their own target geometry before running this script.
// Example:
// var geometry = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax]);

// Cloud Score+ parameter for Sentinel-2 cloud masking.
var QA_BAND = 'cs_cdf';
var CLEAR_THRESHOLD = 0.70;

// Export parameters.
var exportScale = 10;
var exportDescription = 'CAMAF_18band_feature_' + year + '_February';


// -----------------------------------------------------
// Calculate NDWI and MNDWI
// -----------------------------------------------------

function addIndices(image) {
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI');

  return image
    .addBands([ndwi, mndwi])
    .select('B.*', 'TCI.*', 'NDWI', 'MNDWI');
}


// -----------------------------------------------------
// Normalize image bands to 0-255 uint8
// -----------------------------------------------------

function normalizeToUint8(image, geometry) {
  var bands = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
    'B7', 'B8', 'B8A', 'B9', 'B11', 'B12',
    'TCI_R', 'TCI_G', 'TCI_B',
    'NDWI', 'MNDWI',
    'VV'
  ];

  var minMax = image.select(bands).reduceRegion({
    reducer: ee.Reducer.minMax(),
    geometry: geometry,
    scale: exportScale,
    bestEffort: true,
    maxPixels: 1e13
  });

  var normalizedBands = bands.map(function(band) {
    var bandMin = ee.Number(minMax.get(band + '_min'));
    var bandMax = ee.Number(minMax.get(band + '_max'));
    var range = bandMax.subtract(bandMin);

    // Avoid division by zero when min and max are equal.
    var normalized = ee.Image(
      ee.Algorithms.If(
        range.eq(0),
        image.select(band).multiply(0),
        image.select(band)
          .subtract(bandMin)
          .divide(range)
          .multiply(255)
      )
    );

    return normalized
      .clamp(0, 255)
      .uint8()
      .rename(band);
  });

  return ee.ImageCollection(normalizedBands).toBands().rename(bands);
}


// -----------------------------------------------------
// Sentinel-2 preprocessing
// -----------------------------------------------------

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(geometry)
  .filterDate(startdate, enddate)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30));

var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

var s2Composite = s2
  .linkCollection(csPlus, [QA_BAND])
  .map(function(img) {
    return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
  })
  .map(addIndices)
  .median();


// -----------------------------------------------------
// Sentinel-1 preprocessing
// -----------------------------------------------------

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD');

var s1VV = s1
  .filterBounds(geometry)
  .filterDate(startdate, enddate)
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('resolution_meters', 10))
  .select('VV')
  .mean();


// -----------------------------------------------------
// Merge Sentinel-2 and Sentinel-1 features
// -----------------------------------------------------

var fusedImage = ee.Image.cat([
  s2Composite,
  s1VV.toFloat()
]);


// -----------------------------------------------------
// Normalize and export 18-band candidate feature image
// -----------------------------------------------------

var normalizedImage = normalizeToUint8(fusedImage, geometry);

print('Start date:', startdate);
print('End date:', enddate);
print('18-band normalized feature image:', normalizedImage);
print('Band names:', normalizedImage.bandNames());

Export.image.toDrive({
  image: normalizedImage,
  description: exportDescription,
  scale: exportScale,
  region: geometry,
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});


// -----------------------------------------------------
// Map visualization
// -----------------------------------------------------

Map.centerObject(geometry, 8);

Map.addLayer(
  normalizedImage,
  {bands: ['TCI_R', 'TCI_G', 'TCI_B'], min: 0, max: 255},
  'RGB'
);

Map.addLayer(
  normalizedImage,
  {bands: ['NDWI'], min: 0, max: 255},
  'NDWI'
);

Map.addLayer(
  normalizedImage,
  {bands: ['MNDWI'], min: 0, max: 255},
  'MNDWI'
);

Map.addLayer(
  normalizedImage,
  {bands: ['VV'], min: 0, max: 255},
  'VV'
);
