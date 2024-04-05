## mask_raster_with_template

Masks a georeferenced raster file using a binary raster mask template.

Parameters:
- input_raster_path (str): Path to the input georeferenced raster file.
- mask_raster_path (str): Path to the binary raster mask template.

Returns:
None. The input raster file will be replaced by the masked raster.

## mask_all_rasters_in_directory

Masks all georeferenced raster files in a specified directory using a binary raster mask template.

Parameters:
    - directory (str): Path to the directory containing the georeferenced raster files.
    - mask_raster_path (str): Path to the binary raster mask template.

Returns:
    -Each raster file in the specified directory will be replaced by its corresponding masked raster
    

## mask_raster

Parameters:
    - Mask a given raster (DEM) array using a binary mask and optionally filter scatter plot data based on the same mask.
    - dem_array (np.ndarray, optional): The 2D or 3D input raster array to be masked. If 3D, the last dimension is assumed to be the channel dimension (e.g., RGB).
    - mask_path (str): The path to the raster file containing the binary mask. Values of 1 in the mask represent areas to keep, and values of 0 represent areas to mask out.
    - no_data_value (scalar, optional): The value to replace the masked regions with in the `dem_array`. Defaults to np.nan.
    - scatter_x (np.ndarray, optional): The x-coordinates of scatter plot data to be filtered based on the mask. If provided, `scatter_y` must also be provided.
    - scatter_y (np.ndarray, optional): The y-coordinates of scatter plot data to be filtered based on the mask. If provided, `scatter_x` must also be provided.

Returns:
    - np.ndarray: The masked raster array. This array will be of the same shape and data type as the input `dem_array`.
    - (If scatter_x and scatter_y are provided)
    - np.ndarray: The x-coordinates of the scatter plot data after filtering with the mask.
    - np.ndarray: The y-coordinates of the scatter plot data after filtering with the mask.

Note:
    If the `dem_array` data type is integer and the `no_data_value` is np.nan, the function will 
    replace NaN values with a default "no data" integer value (-9999) before casting back to the 
    original data type.

## Optical_flow_akhdefo

Performs feature matching and velocity/displacement calculations across a series of images.

Parameters
----------
input_dir: str
    Path to the directory where the input images are stored.

output_dir : str
    Path to the directory where the output files will be saved.

AOI : str
    The shapefile that represents the Area of Interest (AOI).

zscore_threshold : float
    The threshold value used to filter matches based on their Z-score.

image_resolution : str
    The resolution of the images specified per pixel. This can be expressed in various units 
    like '3125mm', '3.125m' or '3.125meter'.

VEL_scale: (str, optional)
    options year, month, None , default year

VEL_Mode: str
    Options linear or mean , default linear

good_match_option: float
    ratio test as per Lowe's paper default 0.75

shapefile_output: bool
    True to export timeseries as deformation products as shapefile, default False

max_triplet_interval: int 
    Maximum interval days between images allowed to form triplets
    
master_reference: str 
    single, multiple, None
    
Vegetation_mask: (str, optional)
    Path to a raster file that represents a vegetation mask. Pixels in the input image
    that correspond to non-vegetation in the mask will be set to one.

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

start_date: str (example 20210203)
    The start date of the image series.

end_date: str (example 20210503)
    The end date of the image series. 
    
krig_method: str 'ordinary' , 'simple' , 'universal'
    selection of kriging interpolation method, the workflow is based on gstools library. default is 'ordinary'  

use_zscore_krig: float 
    default is None use this option to maintain interpolation within min max limit of data
    
orbit_dir: str None, 'asc' , 'desc' , 'NS' , 'EW'
    if optical image set orbit_dir=None , 'EW' or 'NS' based on slope face of area of interest, if Radar image set orbit_dir to 'asc' or 'desc'
    
    
Returns
-------
image1 : numpy.ndarray
    The first image in the series.

image3 : numpy.ndarray
    The third image in the series.

mean_vel_list : list
    A list of mean velocity arrays, each array corresponding to a pair of images.

mean_flowx_list : list
    A list of mean x-flow arrays, each array corresponding to a pair of images.

mean_flowy_list : list
    A list of mean y-flow arrays, each array corresponding to a pair of images.

points1_i : numpy.ndarray
    Array of keypoints for the first image in the last pair.

points2 : numpy.ndarray
    Array of keypoints for the second image in the last pair.
    
    

## interpolate_xyz

Save XYZ data as a GeoTIFF file using a reference raster for geospatial context.

This function takes X, Y, Z coordinate data and generates a GeoTIFF file. The geospatial context
is derived from a reference raster file. The function supports optional interpolation of Z values,
spatial smoothing, and masking based on a shapefile or a predefined  mask.

Parameters:
    x (array_like): Array of X coordinates.
    y (array_like): Array of Y coordinates.
    z (array_like): Array of Z values corresponding to X and Y coordinates.
    filename (str): The base name of the output GeoTIFF file (without file extension).
    reference_raster (str): Path to the reference raster file used for spatial context (CRS, bounds, etc.).
    shapefile (str, optional): Path to a shapefile for masking the output raster. Defaults to None.
    interpolate (str, optional): Interpolation method to be used (e.g., 'linear', 'nearest'). If None, a nearest-neighbor approach is applied. Defaults to None.
    smoothing_kernel_size (int or float, optional): The size of the Gaussian kernel used for smoothing the Z values. Defaults to None.
    mask (array_like, optional): A boolean mask array to mask vegetation areas. Defaults to None.

Returns:
    numpy.ndarray: The array of interpolated/smoothed Z values, which is also saved as a GeoTIFF file.

Raises:
    Exception: If an error occurs during the process.

    Notes:
    - The function uses `rasterio` for raster operations and `numpy` and `scipy` for data processing.
    - The output GeoTIFF file will have the same spatial extent, resolution, and coordinate reference system (CRS) as the reference raster.
    - If 'interpolate' is not None, Z values are interpolated over the grid defined by the reference raster. Out-of-range interpolated values are replaced with the mean of valid data points.
    - If 'smoothing_kernel_size' is provided, a Gaussian smoothing is applied to the Z values.
    - Masking with either a shapefile or a vegetation mask will set the corresponding areas to NaN.
    - The output file is named using the 'filename' parameter with '.tif' extension.

## is_valid_pixel

Check if a pixel at (x, y) is valid (not zero and not NaN).

## create_dense_keypoints_for_valid_pixels

Create a dense grid of keypoints across the image, excluding invalid pixels.

## extract_descriptors

Extract descriptors for the keypoints by using image patches.
:param image: The input image.
:param keypoints: List of keypoints.
:param patch_size: The size of the patch to extract around each keypoint.
:return: Numpy array of descriptors.

## correlate_descriptors

Correlate descriptors between two sets and return matches.
:param descriptors1: Descriptors from the first image.
:param descriptors2: Descriptors from the second image.
:return: List of DMatch objects.

## filter_matches_by_stability

Filter matches based on a distance threshold for stability.
:param matches: List of DMatch objects.
:param threshold: Distance threshold for filtering.
:return: Filtered list of DMatch objects.

## plot_reference_point

Extracts x, y coordinates from the given XML file and plots them on the provided axis.

Args:
    - xml_file_path (str): Path to the XML file containing the reference point data.
    - ax (matplotlib.axes.Axes, optional): Axes on which to plot. If None, a new figure and axes will be created.

Returns:
    - matplotlib.axes.Axes: The axes on which the data was plotted.
