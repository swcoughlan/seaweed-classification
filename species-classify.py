import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
import rioxarray as rxr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# Load and normalize data
lidar = rxr.open_rasterio("../input/composite.tif").sel(band=[7])
img = lidar.values.reshape(lidar.shape[1], lidar.shape[2], lidar.shape[0])

def normalize(arr):
    arr_min, arr_max = arr.min(), arr.max()
    arr_range = arr_max - arr_min
    scaled = (arr - arr_min) / float(arr_range)
    return -1 + scaled * 2

img = normalize(img)

# Load shapefile and preprocess
shp_file = "/content/drive/MyDrive/Colab Notebooks/SurveyData/shapefile - Copy/seaweed.shp"
shp = gpd.read_file(shp_file)
shp['CLASS_ID'] = pd.Categorical(pd.factorize(shp.species)[0] + 1)
shp = shp.set_crs(epsg=4326).to_crs(epsg=2157)
shp_poly = shp.copy()
shp_poly["geometry"] = shp.geometry.buffer(0.05)
shp_poly = shp_poly[~shp_poly['CLASS_ID'].isin([6, 8])]

# Rasterize the shapefile
def burn_features(geometry, id):
    shapes = ((geom, value) for geom, value in zip(geometry, id.astype(np.uint8)))
    burned = features.rasterize(shapes, fill=0, out_shape=rst.shape, transform=meta['transform'])
    return burned.astype(np.uint8)

rst = rasterio.open("/content/drive/MyDrive/Colab Notebooks/SurveyData/DJI_RGB/composite.tif")
meta = rst.meta.copy()
aoi = burn_features(shp_poly.geometry, shp_poly.CLASS_ID)

# Prepare training data
def train_data(img, aoi):
    labels = np.unique(aoi[aoi > 0]) 
    x = img[aoi > 0, :] 
    y = aoi[aoi > 0]
    return x, y

X, Y = train_data(img, aoi)

# Split data into training and testing sets
sss = StratifiedShuffleSplit(n_splits=10000, test_size=0.1, random_state=56)
train_index, test_index = next(sss.split(X, Y))
X_train, X_test = X[train_index], X[test_index]
Y_train, Y_test = Y[train_index], Y[test_index]

# Train RandomForestClassifier
rf1 = RandomForestClassifier(n_jobs=-1).fit(X_train, Y_train)
y_pred = rf1.predict(X_test)

# Plot confusion matrix and print accuracy
plot_confusion_matrix(rf1, X_test, Y_test, display_labels=[1,2,3,4,5,7], cmap=plt.cm.Blues)
plt.show()
print("Accuracy:", accuracy_score(Y_test, y_pred))

# Reshape image for prediction
img_as_array = img.reshape(-1, img.shape[2])

# Predict classes
class_prediction = rf1.predict(img_as_array)

# Reshape back to original image shape
class_prediction = class_prediction.reshape(img[:, :, 0].shape)
print('Reshaped back to {}'.format(class_prediction.shape))

# Process additional seaweed data
seaweed = rxr.open_rasterio("../results/sieved_seaweed.tif").values.reshape(seaweed.shape[1], seaweed.shape[2])
species = class_prediction.copy()
species[seaweed == 2] = 8  # Assigning a specific class to seaweed data

# Save final classified image
with rasterio.open('../results/output.tif', 'w', driver='GTiff', 
                   height=img.shape[0], width=img.shape[1], count=1, dtype=species.dtype, 
                   crs=rasterio.crs.CRS({"init": "epsg:2157"}), transform=rst.transform) as dst:
    dst.write(species, 1)