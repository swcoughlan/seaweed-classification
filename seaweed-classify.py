import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from osgeo import gdal, gdalconst, osr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import plot_confusion_matrix
from rasterio.features import rasterize

# Read multispectral raster and metadata
rst = rasterio.open("../input/dji_multispec_clipped.tif", mode='r')
meta = rst.meta.copy()
multisp = rst.read([1, 2, 3, 4, 5])

# Reproject DSM raster to match multispectral raster
inputfile = "../input/dji_rgb_ppk_itm_dsm_clipped.tif"
referencefile = "../input/dji_multispec_clipped.tif"
outputfile = "../input/elevation.tif"

input = gdal.Open(inputfile, gdalconst.GA_ReadOnly)
reference = gdal.Open(referencefile, gdalconst.GA_ReadOnly)

gdal.ReprojectImage(input, outputfile, input.GetProjection(), reference.GetProjection(), gdalconst.GRA_Bilinear)

# Read reprojected elevation data
elev = rasterio.open(outputfile).read(1)

# Calculate NDVI
ndvi = (multisp[2] - multisp[4]) / (multisp[2] + multisp[4])

# Stack all bands in a numpy array
stacked = np.dstack((multisp[0], multisp[1], multisp[2], multisp[3], multisp[4], ndvi, elev))

# Write stacked bands to a new raster file
with rasterio.open('../output/stack.tif', 'w', **meta) as dst:
    for i in range(1, stacked.shape[2] + 1):
        dst.write(stacked[:, :, i-1], i)

# Normalize the data
def normalize(arr):
    arr_min = arr.min()
    arr_max = arr.max()
    arr_range = arr_max - arr_min
    return -1 + (arr - arr_min) / arr_range * 2

# Load and preprocess shapefile for training
shp = gpd.read_file("../input/label.shp")
shp = shp.dropna(subset=['geometry'])

# Classify remaining classes to 'no seaweed'
shp['class'] = np.where(shp['class'] == 'seaweed', 'seaweed', 'no seaweed')

# Burn shapefile features into a raster
def burn_features(geometry, values):
    shapes = ((geom, value) for geom, value in zip(geometry, values))
    burned = rasterize(shapes, out_shape=rst.shape, transform=meta['transform'])
    return burned.astype(np.uint8)

aoi = burn_features(shp.geometry, shp['class'].astype('category').cat.codes)

# Train Random Forest Classifier
def train_data(stack, aoi):
    x = stack[aoi > 0]
    y = aoi[aoi > 0]
    return x, y

X, Y = train_data(stacked, aoi)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=56)
train_index, test_index = next(sss.split(X, Y))
X_train, X_test = X[train_index], X[test_index]
Y_train, Y_test = Y[train_index], Y[test_index]

rf = RandomForestClassifier(n_jobs=-1)
rf.fit(X_train, Y_train)
y_pred = rf.predict(X_test)

# Plot confusion matrix and display accuracy
plot_confusion_matrix(rf, X_test, Y_test, cmap=plt.cm.Blues)
plt.show()
print("Accuracy:", accuracy_score(Y_test, y_pred))

# Apply model to the entire raster
predicted = rf.predict(stacked.reshape(-1, stacked.shape[2])).reshape(rst.shape)

# Write prediction to a new raster file
with rasterio.open('../output/predicted.tif', 'w', **meta) as dst:
    dst.write(predicted, 1)
