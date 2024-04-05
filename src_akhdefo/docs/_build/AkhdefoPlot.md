## akhdefo_viewer

Overlays a raster file on a DEM hillshade and saves the plot as a PNG image.

Parameters:
path_to_dem_file (str): Path to the DEM file.
raster_file (str): Path to the raster file.
output_folder (str): Path to the folder where the output image will be saved.
title (str, optional): Title of the plot. Defaults to the raster file's basename.
pixel_resolution_meters (float, optional): Pixel resolution of the raster in meters. Default is None get resolution from raster input.
output_file_name (str, optional): Name of the output PNG image. Defaults to the raster file's basename.
alpha (float, optional): Alpha value for the raster overlay. Default is 0.5.
unit_conversion (str, optional): Unit conversion factor for the raster values. For example, '100cm' for meters to centimeters conversion.
no_data_mask (bool, optional): If True, masks pixels with a value of 0 in the raster. Default is False.
colormap (str, optional): Colormap to use for the raster. Default is 'jet'.
min_value (float, optional): Minimum value for normalization. Uses raster's minimum if None.
max_value (float, optional): Maximum value for normalization. Uses raster's maximum if None.
normalize (bool, optional): If True, normalizes the raster values. Default is False.
colorbar_label (str, optional): Label for the colorbar. 
show_figure (bool, optional): Whether to display the figure. Default is True.
aspect_raster (str, optional): whetehr to plot displacement vector. Dedulat is None 
cmap_aspect (str, optional): colormap to sue for the vector arrows
step (int, optional): density of the aspect vector arraows. Defulat is 10 pixel unit draw 1 arrow

Returns:
None

## _separate_floats_letters

Separates floats and letters from a string.

## _normalize_raster

Normalizes raster values between given minimum and maximum values.

## _create_plot

Creates and saves a plot of hillshade and raster overlay.

## plot_stackNetwork

Generates a scatter plot to visualize the time intervals between dates extracted from the filenames of .tif files.
The function handles filenames containing single or multiple dates. For multiple dates, it calculates the interval 
between the first and last date. For single dates, it calculates the interval between consecutive dates.

Parameters:
- src_folder (str): The source folder containing .tif files.
- output_folder (str): The folder where the plot image will be saved.
- cmap (str): The colormap used for the scatter plot. Default is 'tab20'.
- date_plot_interval (tuple): A tuple indicating the minimum and maximum number of ticks on the date axis. Default is (5, 30).
- marker_size (int): The size of markers in the scatter plot. Default is 15.

The function first attempts to find multiple dates in each filename. If multiple dates are found, it calculates the
interval (delta_dd) between the first and last dates. If only one date is found, it calculates the interval between 
the current and next date in the sequence. The scatter plot displays these intervals, with the color indicating the 
specific date and position reflecting the time interval.

The function saves the generated plot as 'Stack_Network.jpg' in the specified output folder.

Note:
- The function assumes that the filenames follow a specific date format, either 'YYYYMMDD' or 'YYYY-MM-DD'.
- Files are processed in alphabetical order, which should correspond to chronological order for accurate interval calculation.

## akhdefo_ts_plot

This program used for analysis time-series velocity profiles

Parameters
----------

user_data_points: str
    provide path to csv. file contains x and y coordinate for points of interest
    you can generate this file by providing path to path_saveData_points (POI.csv).
    This is useful to save mouse click positions to repeat the plots for different datasets for example if you plot several TS profiles for
    EW velocity product, you can recreate TS for the same exact position by saving POI.csv with path_saveData_points and then use that as input for the another
    plot such as NS velocity product via setting user_datapoints="POI.csv"

path_to_shapefile: str 
    type path to timeseries shapefile in stack_data/TS folder

dem_path: str
    path to dem raster in geotif fromat

point_size: float
    size of the sactter plot points

opacity: float 
    transparency of the scater overlay

cmap: str
    Matplotlib colormap options example "RdYlBu_r, jet, turbo, hsv, etc..."
                
Set_fig_MinMax: bool
    True or False

MinMaxRange: list
    [-50,50]  Normalize plot colormap range if Set_fig_MinMax=True

color_field: str 
    'VEL' ,"VEL_2D", 'VEL_N', 'VEL_E', 'VELDir_MEA'


path_saveData_points: str
    optional, provide directory path if you want to save profile data.
    the data will be saved under POI.csv file


save_plot: bool
    True or False

Fig_outputDir: str
    if save_plot=True then
    you save your profile plots in interactive html file and jpg image 

VEL_Scale: str
    'year' or 'month' projects the velocity into provided time-scale

filename_dates: str
    provide path to Names.txt file, this file generated at stack_prep step
    
    
Returns
-------
Interactive Figures

## MeanProducts_plot_ts

This program used to plot shapefile data

Parameters
----------

path_to_shapefile : str

dem_path: str 

out_folder: str

color_field: str
    geopandas column name

Set_fig_MinMax: bool

MinMaxRange: list
        
opacity: float

cmap: str

point_size: str 

cbar_label: str
    "mm/year" or "degrees", etc.. based on unit of the data column name in the color_field

Returns
-------
Figure

## aspect_to_uv

Convert aspect data to U and V components for arrows.
