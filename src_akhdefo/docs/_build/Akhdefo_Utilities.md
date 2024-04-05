## Akhdefo_resample

This program performs raster resampling for  rasters

Parameters:
------------

input_raster: str
    path to input raster

output_raster: str
    path to output raster


xres: float
    horizontal resolution

yres: float 
    vertical resolution

SavFig: bool
    True to save output plot False to ignore exporting plot

convert_units: float 
    if not None converts raster value units from meter to mm: depends on your raster unit adjust this value

Returns:
--------
    geotif raster

## Akhdefo_inversion

This program calculates 3D displacement velocity (East-West,North-South and vertical) using combined optical and InSAR products

Parameters:
------------

horizontal_InSAR: str
    path to East Velocity InSAR product in geotif format

Vertical_InSAR: str
    path to Vertical Velocity InSAR product in geotif format

EW_Akhdefo: str 
    path to east-west velocity  akhdefo(optical) product in geotif format

NS_Akhdefo: str
    path to north-south velocity  akhdefo(optical) product in geotif format

dem_path: str
    path to DEM raster in geotif format

output_folder : str
    path to save raster products 


Returns:
---------
Three geotif rasters
    3D-Velocity (D3D in mm/year) raster
    Plunge raster in degrees
    Trend raster in degress

## Auto_Variogram

This function performs automatic selection of the optimal variogram model for spatial data interpolation. 
It also supports clipping of the interpolation results to a specified Area of Interest (AOI). The function 
accepts both GeoDataFrame objects and file paths (specifically, shapefile paths) as input data sources.

Parameters:
------------
data : str or gpd.GeoDataFrame, optional
    The path to a shapefile containing point data, or a GeoDataFrame. For shapefiles, they must include 
    'x', 'y' coordinates (or 'lat', 'lon' if latlon is set to True). Defaults to an empty string.

column_attribute : str, optional
    The name of the attribute within the shapefile or GeoDataFrame to be interpolated. 

latlon : bool, optional
    Indicates whether the input data uses latitude and longitude (True) or Cartesian coordinates (False). 
    Defaults to False.

aoi_shapefile : str, optional
    The path to a shapefile that defines the Area of Interest (AOI) for clipping the interpolation results. 
    Defaults to an empty string.

pixel_size : int, optional
    The resolution size for the interpolated grid. Defaults to 20.

num_chunks : int, optional
    The number of chunks for processing to mitigate memory issues. Adjust as needed. Defaults to 10.

overlap_percentage : float, optional
    The percentage of overlap between chunks, ranging from 0 to 1.0. Defaults to 0.

out_fileName : str, optional
    The name for the output GeoTIFF file storing the interpolation results. Defaults to 'interpolated_kriging'.

plot_folder : str, optional
    Directory path to save plots. Defaults to 'kriging_plots'.

geo_folder : str, optional
    Directory path for saving geospatial raster files. Defaults to 'geo_rasterFolder'.

smoothing_kernel : int, optional
    The size of the smoothing kernel to be used. Defaults to 2.

mask : [np.ndarray], optional
    A numpy array to be used as a mask for the interpolation.

UTM_Zone : str, optional
    The UTM Zone designation ('N' for Northern Hemisphere or 'S' for Southern Hemisphere).

krig_method : str, optional
    The method of kriging to be used, either 'ordinary' or 'universal'. Defaults to 'ordinary'.



drift_functions: str, optional 
    only works if krig_method='universal' available options "linear", "quadratic", "x"  , "y"

detrend_data : bool , optional
    if True removes the linear trend from the interpolated data and save the detrended data as geotif

use_zscore : float, optional
    use statistical outlier removal before performing interpolation default is None. options 2, 3.5, 5 , etc..

Returns:
--------
numpy.ndarray
    A 2D grid of interpolated values, clipped to the specified AOI if an AOI shapefile is provided.

Raises:
-------
ValueError
If the input data is not a valid shapefile path or GeoDataFrame.
If essential columns (x, y or lat, lon) are missing in the input data.

Notes:
------
The function generates two types of plots:
1. Fitted variogram models against the experimental variogram.
2. The interpolation result using the selected variogram model.

Dependencies:
-------------
    Requires geopandas, gstools, pykrige, matplotlib, and rasterio libraries.

## akhdefo_download_planet

Parameters:
-----------
Note: To use this function need to call await as below:

await akhdefo_download_planet()


planet_api_key: str = "  "
    input planet labs api 

AOI: str = "plinth.json"
    input area of interest in json file format

start_date: str = "May 1, 2016"
    input start date as the following format "Month Day, Year"

end_date: str = "May 31, 2017"
    input end date as the following format "Month day, Year"


limit: int = 5
    input Maxumum number of images want to download; type None if you want to download images daily into the future but need to set the end_date empty as follow end_date=""

item_type: str = "PSOrthoTile"
    input item type to downoload please refere to planet labs website for further detalis: 
    PSScene:        PlanetScope 3, 4, and 8 band scenes captured by the Dove satellite constellation
    REOrthoTile:    RapidEye OrthoTiles captured by the RapidEye satellite constellation
    REScene:        Unorthorectified strips captured by the RapidEye satellite constellation
    SkySatScene:    SkySat Scenes captured by the SkySat satellite constellation
    SkySatCollect:  Orthorectified scene composite of a SkySat collection
    SkySatVideo:    Full motion videos collected by a single camera from any of the active SkySats
    Landsat8L1G     Landsat8 Scenes: provided by USgs Landsat8 satellite
    Sentinel2L1C:   Copernicus Sentinel-2 Scenes provided by ESA Sentinel-2 satellite

product_bundle: str = "analytic_sr_udm2"
    please refer to planetlabs website for further details and different options of product_bundle: default is analytic_sr_udm2
    (analytic,analytic_udm2, analytic_3b_udm2, analytic_5b , analytic_5b_udm2 , analytic_8b_udm2, visual, uncalibrated_dn, 
    uncalibrated_dn_udm2, basic_analytic, basic_analytic_udm2, basic_analytic_8b_udm2, basic_uncalibrated_dn,
    basic_uncalibrated_dn_udm2, analytic_sr, analytic_sr_udm2, analytic_8b_sr_udm2, basic_analytic_nitf, 
    basic_panchromatic, basic_panchromatic_dn, panchromatic, panchromatic_dn, panchromatic_dn_udm2,
    pansharpened, pansharpened_udm2 , basic_l1a_dn)

clear_percent: int = 90
    Quality of the scene, if you need to download best images keep this value min 90. although, it will end up with less image acquistion 

cloud_filter: float = 0.1
    cloud percentage

output_folder: str = "raw_data"
    output directory to save the orders

clip_flag: bool
    True to clip the downloads to Area of interest json file format provided above
        

download_data: bool 
    True to download the data or False to preview the data

## akhdefo_orthorectify

Parameteres:
============
input_Dir: str
    input unortho image directory path

dem_path: str 
    input path to DEM file

output_path: str
    input path to output directory

ortho_usingRpc: bool
    Use of RPC file for raw none-georectified satallite images
 

## download_RTC

Initiates the download of Synthetic Aperture Radar (SAR) products from ASF's HyP3 platform.

This function facilitates the submission and download of SAR processing jobs, including Radiometric Terrain 
Correction (RTC), AutoRIFT, and InSAR products. It filters and selects granules based on the provided ASF datapool 
results file and other specified parameters.

Parameters:
-----------
username (str): 
    ASF HyP3 username. Required unless prompt=True.

password (str):
    ASF HyP3 password. Required unless prompt=True.

prompt (bool):
    If True, use interactive prompts for login instead of username/password.

asf_datapool_results_file (str): 
    Path to ASF datapool results CSV file.

save_dir (str):
    Directory where downloaded files will be saved.

job_name (str, optional): 
    Name for the job. Defaults to 'rtc-test'.

dem_matching (bool, optional):
    If True, perform DEM matching for RTC jobs. Defaults to True.

include_dem (bool, optional):
    If True, include Digital Elevation Model in RTC jobs. Defaults to True.

include_inc_map (bool, optional):
    If True, include incidence angle map in RTC jobs. Defaults to True.

include_rgb (bool, optional):
    If True, include RGB decomposition in RTC jobs. Defaults to False.

include_scattering_area (bool, optional): 
    If True, include scattering area in RTC jobs. Defaults to False.

scale (str, optional):
    Scale for the image in RTC jobs. Defaults to 'amplitude'.

resolution (int, optional): 
    Desired resolution in meters for RTC jobs. Defaults to 10.

speckle_filter (bool, optional):
    Apply Enhanced Lee speckle filter in RTC jobs. Defaults to True.

radiometry (str): 
    Radiometry normalization (either 'sigma0' or 'gamma0') for RTC jobs.

dem_name (str): 
    DEM to use for RTC jobs. 'copernicus' or 'legacy'.

download (bool): 
    If True, submit jobs and download data. Defaults to False.

limit (int, optional): 
    Limit the number of images to download.

path_number (int, optional):
    Filter granules by path number.

frame_number (int, optional): 
    Filter granules by frame number.

RTC (bool): 
    If True, process RTC jobs.

autorift (bool): 
    If True, process AutoRIFT jobs.

insar (bool):
    If True, process InSAR jobs.

max_neighbors (int):
    Max number of neighbors for InSAR job pairing.

Returns:
--------
    sdk.Batch: A Batch object containing the submitted jobs.

Raises:
-------
    ValueError: If required arguments are not provided.

Notes:
-------
    - This function requires prior installation of the HyP3 SDK and relevant dependencies.
    - The ASF datapool results file should be in CSV format with specific columns for filtering.
    - Downloaded files will be saved in the specified directory with additional suffixes based on the job type.

## reproject_raster_to_match_shapefile

Reproject a raster to match the coordinate reference system (CRS) of a shapefile.

Parameters:
------------
- src_path (str): Path to the source raster file that needs to be reprojected.
- dst_path (str): Path to save the reprojected raster.
- dst_crs (CRS or str): Target coordinate reference system.

Returns:
---------
None. The reprojected raster is written to dst_path.

## create_vegetation_mask

Create a binary vegetation mask based on the NDVI (Normalized Difference Vegetation Index) calculation.

Parameters:
------------
red_band_path (str): 
    Path to the raster file containing the red band.

nir_band_path (str): 
    Path to the raster file containing the near-infrared (NIR) band.

output_path (str): 
    Path to save the generated vegetation mask raster.

shapefile_path (str): 
    Path to the shapefile that defines the area of interest (AOI).

threshold (float, optional):
    NDVI threshold for determining vegetation. Pixels with NDVI less than this threshold are considered vegetation. Default is 0.3.

save_plot (bool, optional):
    Whether to save a plot of the vegetation mask. Default is False.

plot_path (str, optional): 
    Path to save the plot if save_plot is True. Default is "plot.png".

Returns:
----------
None. The vegetation mask is written to output_path, and optionally a plot is saved.

Note:
The function assumes that the shapefile contains only one geometry.

## adjust_brightness

Adjust the brightness of an image.
brightness > 0 will increase brightness
brightness < 0 will decrease brightness

## measure_displacement_from_camera

Test data URLs; hls_url = "https://chiefcam.com/resources/video/events/september-2021-rockfall/september-2021-rockfall-1080p.mp4" or "https://chiefcam.com/video/hls/live/1080p/index.m3u8"

Measure the displacement from the camera feed using Dense Optical Flow.

Parameters:
------------

hls_url: str
    The URL of the HLS video stream. or type 0 to process video from live pc webcam or add path to your video

alpha: float, optional
    The weight of the image to update the background model, default is 0.1.
    
save_output : bool, optional
    Flag to save the output, default is False.
output_filename : str, optional
    The filename for saving the output video, required if save_output is True.
ssim_threshold: float , default 0.4
    if interetesed to identify rockfalls recommended value is 0.5 or less.
    
pyr_scale: float
    parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
levels: int
    number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
winsize: int
    averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
iterations: int
    number of iterations the algorithm does at each pyramid level.
poly_n: int
    size of the pixel neighborhood used to find polynomial expansion in each pixel; 
    larger values mean that the image will be approximated with smoother surfaces, 
    yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
poly_sigma: float
    standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; 
    for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
flags: 0 or 1
    operation flags that can be a combination of the following:
    0 OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
    1 OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize×winsize filter instead of a box filter of the same size for optical flow estimation; 
    usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed; 
    normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness.

Returns:
--------- 
    Video output with motion vectors and magnitude.

## measure_displacement_from_camera_toFile

Test data URLs; hls_url = "https://chiefcam.com/resources/video/events/september-2021-rockfall/september-2021-rockfall-1080p.mp4" or "https://chiefcam.com/video/hls/live/1080p/index.m3u8"

Measure the displacement from the camera feed using Dense Optical Flow.


Parameters:
------------

hls_url: str
    The URL of the HLS video stream. or type 0 to process video from live pc webcam or add path to your video

alpha: float, optional
    The weight of the image to update the background model, default is 0.1.
    
save_output : bool, optional
    Flag to save the output, default is False.
output_filename : str, optional
    The filename for saving the output video, required if save_output is True.
ssim_threshold: float , default 0.4
    if interetesed to identify rockfalls recommended value is 0.5 or less.
    
pyr_scale: float
    parameter, specifying the image scale (<1) to build pyramids for each image; pyr_scale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous one.
levels: int
    number of pyramid layers including the initial image; levels=1 means that no extra layers are created and only the original images are used.
winsize: int
    averaging window size; larger values increase the algorithm robustness to image noise and give more chances for fast motion detection, but yield more blurred motion field.
iterations: int
    number of iterations the algorithm does at each pyramid level.
poly_n: int
    size of the pixel neighborhood used to find polynomial expansion in each pixel; 
    larger values mean that the image will be approximated with smoother surfaces, 
    yielding more robust algorithm and more blurred motion field, typically poly_n =5 or 7.
poly_sigma: float
    standard deviation of the Gaussian that is used to smooth derivatives used as a basis for the polynomial expansion; 
    for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
flags: 0 or 1
    operation flags that can be a combination of the following:
    0 OPTFLOW_USE_INITIAL_FLOW uses the input flow as an initial flow approximation.
    1 OPTFLOW_FARNEBACK_GAUSSIAN uses the Gaussian winsize×winsize filter instead of a box filter of the same size for optical flow estimation; 
    usually, this option gives z more accurate flow than with a box filter, at the cost of lower speed; 
    normally, winsize for a Gaussian window should be set to a larger value to achieve the same level of robustness.

Returns:
---------- 
    Video output with motion vectors and magnitude.

## calculate_new_geotransform

Calculate a new geotransform matrix based on the overlap of image boundaries.

Parameters:
------------
- overlap_box (shapely.geometry.Polygon): The overlapping bounding box of all valid images.
- src_transform (tuple): The source geotransform matrix to be modified.

Returns:
---------
- tuple: A new geotransform matrix corresponding to the overlap_box.

## crop_to_overlap

Crop all valid image files in the specified folder to the overlapping area.

This function performs the following steps:
1. Iterate through all files in the folder and identify valid image files.
2. Calculate the overlapping bounding box of all valid images.
3. Crop each valid image to the overlapping bounding box.
4. Overwrite the original image files with the cropped images.

Parameters:
------------
- folder_path (str): The path to the folder containing the image files to be cropped.

Notes:
-------
- Supports '.tif', '.jpg', '.png', and '.bmp' file formats.
- Requires GDAL, OSR, and Shapely libraries.
- The folder should contain images that share the same spatial reference system (projection).

Exceptions are handled for permission errors and unexpected errors during the file overwriting process.

## crop_point_shapefile_with_aoi

Crop a point shapefile with an Area of Interest (AOI) polygon shapefile.

Parameters:
------------
- point_shapefile (str): Path to the input point shapefile.
- aoi_shapefile (str): Path to the AOI polygon shapefile for cropping.
- output_folder (str): Path to the folder where the cropped shapefile will be saved.

Returns:
---------
None

This function reads a point shapefile and an AOI polygon shapefile, and performs a spatial
intersection to crop the points within the specified AOI. The resulting GeoDataFrame is then
saved as a new shapefile in the specified output folder. If no points are within the AOI, no
file is created, and a message is printed.

Example:
```
import akhdefo_functions,
from akhdefo_functions import crop_point_shapefile_with_aoi
```


```
point_file = "path_to_points.shp"
aoi_file = "path_to_aoi.shp"
output_folder = "output_folder"
```
```
crop_point_shapefile_with_aoi(point_file, aoi_file, output_folder)
```

Note:
- Requires the GeoPandas library.
- The AOI polygon is dissolved into a single geometry for intersection.
- Both input GeoDataFrames must have the same Coordinate Reference System (CRS).
- The resulting shapefile is saved in the output folder with "_clipped" appended to the filename.
- If no points are within the AOI, no file is created, and a message is printed.

## get_transform_from_gdf

Generate an affine transform for a raster-like GeoDataFrame.

Parameters:
-----------
- gdf: The GeoDataFrame.
- pixel_size: The resolution (spacing) of your points/data.

Returns:
---------
- A rasterio Affine object.

## create_norm

Create a normalization instance based on the data.
If data has negative values, use TwoSlopeNorm with zero as the center.
Otherwise, use standard normalization.

## detect_outliers

Detect outliers based on deviation from local mean.

## simple_kriging

A very simplified kriging-like interpolation for outliers.

## set_video_bitrate

Set the bitrate of a video using FFmpeg. If the output file already exists, a new file with a suffix is created instead.

Parameters:
- input_video_path: Path to the input video file.
- output_video_path: Path where the output video with the new bitrate will be saved.
- bitrate_kbps: Desired video bitrate in kbps.

## delete_files_except_last_n

Delete all files in the given directory except the last n files based on modification time.

## find_and_clean_stats_folders

Find 'stats' folders in the given root_directory and its subdirectories, then clean them.

## find_polygons

Finds regions with pixels of value 1 and identifies polygons around them.

Parameters:
- image: A binary image (numpy array) where pixels of interest have a value of 1.

Returns:
- A list of polygons, where each polygon is represented by a list of its vertex coordinates.

## draw_circles_on_image

Draws circles on the base_image based on the polygons provided.

Parameters:
- polygons: A list of polygons, each represented by a list of its vertex coordinates.
- base_image: The image (numpy array) on which circles will be drawn.

Returns:
- The image with drawn circles.

## replace_outliers_with_nan_fixed

Replaces outliers in a 2D array with NaN based on a Z-score threshold, with a fix for data type issue.

Parameters:
- data: 2D NumPy array
- z_threshold: Z-score threshold to identify outliers

Returns:
- A 2D array with outliers replaced by NaN

## mask_outside_contour

Masks everything outside the given contour in the grayscale image with NaN.

Parameters:
- contour: The contour (as obtained from cv2.findContours) to mask inside of.
- gray_img: The grayscale image (2D numpy array).

Returns:
- The modified image with areas outside the contour masked as NaN.
