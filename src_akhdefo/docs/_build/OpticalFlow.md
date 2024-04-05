## binary_mask

Function that generates a binary mask from a vector file (shp)

Parameters
----------

raster_path: str
    path to the .tif;

shape_path: str
    path to the shapefile.

output_path: str
    Path to save the binary mask.

file_name: str
    Name of the file.

Returns
-------
Raster: ndarray
    Binary Mask in tif format




    

## DynamicChangeDetection

This program calculates optical flow velocity from triplets of daily optical satellite images.
Final Timeseris products will be a shapefile format using Time_Series function after stackprep step.

Parameters
----------

Path_working_Directory: str
    path to filtered raster images

Path_UDM2_folder: str
    path to planetlabs udm2 mask files

Path_to_DEMFile: str
    path to digital elevation model

AOI_shapefile: str
    path to area of interest file in esri shapefile format

Coh_Thresh: float
    similarity index threshold

vel_thresh: float
    maximum velocity magnitude allowed to be measured; this will help the program to exlude rockfall velocity.
    hence, only calculating displacement velocity.

image_sensor_resolution: float
    Resolution of the satallite image raster resolution in millimeters. 
    for instance Planetlabs ortho imagery 1 pixel=3125.0 mm 
udm_mask_option: bool
    True or False

cmap: str
    matplotlib colormap such as "jet", "hsv", etc...

Median_Filter: bool
    True or False

Set_fig_MinMax: bool
    True or False

show_figure: bool
    True or False

plot_option: str
    "origional",  "resampled"

xres: int

yres: int

Returns
-------
Rasters
     velocity in X direction(EW)
     Velocity in Y direction(NS)

Figures  
    Initial Timesereis Figures (those figures are only intermediate products needs calibration)
