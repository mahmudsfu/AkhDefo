## stackprep

This program collects velocity candiate points for time-series analysis.

Parameters
----------

path_to_flowxnFolder: str
    path to folder include east-west velocity files

path_toFlowynFolder: str
    path to folder include north-south velocity files

dem: str
    path to digital elevation model file will be used to geocode the products

print_list: bool
    print list of temporal proceesed dates default is False

start_date: str
    "YYYYMMDD"

end_date: str
    "YYYYMMDD"

output_stackedFolder: str

VEL_scale: str
    "month" or "year") at this stage you can ignore this option; will be removed from future versions
  
xres: float
    x resolution
yres: float
    y resolution
 
Velocity_shapeFile: bool

    set to True if need to generate points for temporal deformation analysis

Resampling: bool
    if True reduce number of measurement points but faster processing

Raster_stack_correction: bool
    if True this feature computes the linearly interpolated pixel values between subsequent time-slices(bands) in raster stack

Returns
-------
ESRI Shapefile
    This file include candiate velocity points for timeseries analysis
    

## is_outlier

Returns a boolean array with True if points are outliers and False 
otherwise.

Parameters:
-----------
    points : An numobservations by numdimensions array of observations
    thresh : The modified z-score to use as a threshold. Observations with
        a modified z-score (based on the median absolute deviation) greater
        than this value will be classified as outliers.

Returns:
--------
    mask : A numobservations-length boolean array.

References:
----------
    Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
    Handle Outliers", The ASQC Basic References in Quality Control:
    Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 

## interpolate_missing_pixels

:param image: a 2D image
:param mask: a 2D boolean image, True indicates missing values
:param method: interpolation method, one of
    'nearest', 'linear', 'cubic'.
:param fill_value: which value to use for filling up data outside the
    convex hull of known pixel values.
    Default is 0, Has no effect for 'nearest'.
:return: the image with missing values interpolated
