## resample_raster

Resample the source raster array to match the destination raster's resolution and extent.

Parameters:
- src_array: 2D numpy array of the source raster
- src_transform: affine.Affine transform of the source raster
- src_crs: CRS of the source raster
- dst_transform: affine.Affine transform of the destination raster
- dst_crs: CRS of the destination raster
- dst_shape: Shape of the destination raster (height, width)
- resampling_method: rasterio.enums.Resampling method to use for resampling

Returns:
- resampled_array: 2D numpy array of the resampled source raster

## scatter_area_mask

Create an accumulated scatter area mask from a set of raster images based on a given threshold. the input dataset is taken from ASF RTC processing.
The scattering area for each pixel in the RTC image in square meters. The values are calculated based on the effectively illuminated gamma-0 terrain surface using a digital elevation model, 
the local incidence angle map, and the layover-shadow map. see detailes at the following website https://hyp3-docs.asf.alaska.edu/guides/rtc_product_guide/#scattering-area-map

The function processes each raster image in the input folder, crops it based on the provided AOI
from the shapefile, normalizes the cropped raster, and then converts the normalized image to a binary
mask based on the scatter_percentageArea_threshold. The binary masks from each raster are then accumulated
to generate the final scatter area mask.

Parameters:
-----------
input_folder : str
    Path to the folder containing raster files to be processed.

output_folder : str
    Directory where the final accumulated mask raster file will be saved.

plot_folder : str
    Directory where the visual representation (plot) of the accumulated mask will be saved.

shapefile_path : str
    Path to the shapefile containing the Area of Interest (AOI) for cropping the raster images.

scatter_Area_threshold : float, optional (default=10) unit is perentage  square 
    Threshold for determining the binary mask from the normalized raster image. Pixels with values 
    less than this threshold are set to 0 and those above are set to 1.

Returns:
--------
Shadow Mask for SAR image for sites less likey to have quality measurment points.
The results are saved as files in the specified output and plot directories.

Notes:
------
- Assumes that there is only one geometry in the provided shapefile.
- The accumulated mask is a result of multiplying binary masks from each raster. Therefore, a pixel in 
  the accumulated mask will have a value of 1 only if all rasters have a value of 1 at that pixel location.

## utm_to_latlon

This program converts geographic projection of shapefiles from UTM to LATLONG

Parameters
----------
easting: Geopandas column with Easting 

northing: Geopandas column with Northing

zone_number: int

zone_letter: "N" or "S"

Returns
-------
[lon , lat ]: List

## assign_fake_projection

Note
====

This program assigns fake latlon geographic coordinates to ground-based images 
so that it can be ingest using gdal and rasterio geospatial libraries for further processing

input_dir: str
    path to image directories without projection info

output_dir: str
    output path image directory for images included projection info

## move_files

This function reorganizes files in the specified directory. 
It searches for timestamps in filenames, creates subdirectories based on the hour part of the timestamp,
and moves files to the appropriate subdirectories. The files are renamed based on the year, month, and day of the timestamp.

Args:
    base_directory (str): Path of the directory containing the files to be reorganized.

## Lenet_Model_training

This function, Lenet_Model_train(), is designed to train a convolutional neural network (CNN) using the LeNet architecture. The network is trained on a dataset of images to classify whether they are "foggy" or "not foggy".

Parameters:
-----------

dataset: str
  (default="DataForTraining") Path to the directory containing the image data for training. The images are expected to be in separate directories named after their corresponding class ("foggy" or "not foggy").
model_out: str
  (default="foggy_not_foggy.model") The name or path for the output file where the trained model will be saved in the h5 format.
plot: str
 (default="Model_stat_plot.png") The name or path for the output image file where a plot of the training loss and accuracy will be saved.
EPOCHS: int
  (default=100)The number of epochs to use for training.
INIT_LR: float
  (default=1e-3)The initial learning rate for the Adam optimizer.
BS: int
  (default=32)The batch size for training.

Returns:
--------
- Trains a LeNet model on the given dataset.
- Saves the trained model to disk in the h5 format.
- Plots the training and validation loss and accuracy as a function of epoch number, and saves the plot to disk. The plot also includes the model summary.
- Note: The function uses data augmentation techniques during training, including random rotations, width and height shifts, shearing, zooming, and horizontal flipping.
- This function uses the TensorFlow, Keras, OpenCV, and matplotlib libraries.

## classification

Classifies images in the specified directory using a trained model.

Inputs:
-------
    - input_dir (str, optional): Path to the directory containing the input images. Defaults to "dataset_imagery".
    - trained_model (str, optional): Path to the trained model file. Defaults to "foggy_not_foggy.model".



Returns:
--------
    - The function assumes that the input directory contains image files in JPG format.
    - The function uses a trained convolutional neural network model to classify the images.
    - It saves the classified images into separate directories based on their classification.

## calculate_volume

Calculate the volume based on an elevation map and a slope map,
and save the volume map as a GeoTIFF file. Optionally, plot the volume map as a figure and save it.

Args:
    elevation_map (str): File path to the elevation map raster.
    slope_map (ndarray): 2D array representing the slope values.
    cell_size (float): Size of each cell in the map (e.g., length of one side of a square cell).
    output_file (str): Output file path for saving the volume map as a GeoTIFF.
    plot_map (bool, optional): Whether to plot the volume map as a figure. Default is False.
    plot_file (str, optional): Output file path for saving the volume map plot. Required if plot_map is True.

Returns:
    ndarray: The calculated volume map.

## akhdefo_fitPlane

Fit planes to points in a Digital Elevation Model (DEM) and visualize the results.

Parameters:
    - dem_data (str): Path to the DEM data file (GeoTIFF format).
    - line_shapefile (str): Path to the shapefile containing line features representing planes.
    - out_planeFolder (str): Path to the folder where the output plane data will be saved.

How It Works:
    This function reads a DEM and shapefile, allows the user to interactively select points from the DEM,
    fits planes to the selected points, and visualizes the results in 2D and 3D plots. It also provides options
    to save the fitted planes as XYZ and DXF files. additionally plots poles to planes on polar grid and rose diagram for strike/trends of planes

Note:
    - The function utilizes various libraries such as numpy, matplotlib, tkinter, osgeo (GDAL), and geopandas.
    - Ensure that the required libraries and dependencies are installed to use this function effectively.

Example Usage:
    akhdefo_fitPlane(dem_data='path/to/dem.tif',line_shapefile='path/to/lines.shp',out_planeFolder='output/folder')
    

## move_files_with_string

Move files from a source directory to a destination directory based on a search string present in their paths.

Parameters:
- source_dir (str): The directory from which files are to be moved.
- dest_dir (str): The destination directory where files will be moved.
- search_string (str): The string to search for in the file paths.

This function traverses the source directory, including its subdirectories. 
Files whose paths contain the search string are moved to the destination directory. 
If a file with the same name exists in the destination, it's renamed to avoid overwriting.

Errors during file movement (e.g., permission issues, non-existent directories) are logged but do not stop the process.

## partial_derivative

Calculate the central difference approximation for the partial derivatives
of a 2D function with respect to x and y.

:param f: 2D numpy array of function values, representing the raster
:param dx: Spacing in the x-direction (assumed to be constant)
:param dy: Spacing in the y-direction (assumed to be constant)
:return: Two 2D numpy arrays, one for the partial derivative with respect to x (df_dx)
         and one for the partial derivative with respect to y (df_dy)

## calculate_slope

Calculate the slope at each pixel using the aspect to determine direction.

Parameters:
dem (numpy.ndarray): Digital Elevation Model (DEM) array.
aspect (numpy.ndarray): Aspect array.
dx (float):  x Spatial resolution of the raster (distance between pixels).
dy (float):  y Spatial resolution of the raster (distance between pixels).

Returns:
numpy.ndarray: Array of slope values in degrees.

## calculate_height_change

Calculate the height change using the slope and distance for each pixel.

Parameters:
slope (numpy.ndarray): Array of slope values in degrees.
distance (numpy.ndarray): Array of distance values.

Returns:
numpy.ndarray: Array of height changes.

## calculate_volume_change

Calculate the volume change for each pixel.

Parameters:
height_change (numpy.ndarray): Array of height changes.
pixel_area (float): Area of a single pixel.

Returns:
numpy.ndarray: Array of volume changes.

## displacement_to_volume

Process the DEM, aspect, and displacement rasters to calculate and export the slope, height change, and volume change.

Parameters:
dem_path (str): Path to the DEM raster file.
aspect_path (str): Path to the aspect raster file.
displacement_path (str): Path to the displacement raster file.
slope_output_path (str): Path for the slope output GeoTIFF file.
height_output_path (str): Path for the height change output GeoTIFF file.
volume_output_path (str): Path for the volume change output GeoTIFF file.
dx (float):  x Spatial resolution of the raster (distance between pixels).
dy (float):  y Spatial resolution of the raster (distance between pixels).
pixel_area (float): Area of a single pixel.

Returns:
None: Outputs GeoTIFF files at the specified output paths.

## calculate_and_save_aspect_raster

Calculate the aspect raster from east-west and north-south displacement rasters and save it.

This function reads two raster files representing east-west (EW) and north-south (NS) 
displacements, calculates the aspect of each pixel, and saves the result as a new raster file.
Aspect is calculated in degrees from north (0 degrees), east (90 degrees), south (180 degrees),
and west (270 degrees).

Parameters:
- ew_raster_path (str): File path for the east-west displacement raster.
- ns_raster_path (str): File path for the north-south displacement raster.
- output_raster_path (str): File path for the output aspect raster.

Returns:
None. The result is saved as a new raster file at the specified output path.

## random_color

Generate a random color.

## bounding_box

Find the bounding box for a set of points.

## collect_points_from_line

Collect start, middle, and end points from each line feature.

## read_raster

Read a raster file and return the data array and geotransform.

## calculate_aspect

Calculate the aspect from EW and NS displacement data.

## save_raster

Save the data as a raster file.
