
# '''
# #Under construction

# import numpy as np

# def los_to_vertical_ew_ns(los_asc, los_desc, los_angle_asc, los_angle_desc, heading_asc, heading_desc):
#     """
#     Converts LOS ascending and descending rasters to vertical, EW, and NS components.
    
#     Parameters:
#     - los_asc: LOS ascending raster as a NumPy array.
#     - los_desc: LOS descending raster as a NumPy array.
#     - los_angle_asc: LOS angle for ascending track in degrees.
#     - los_angle_desc: LOS angle for descending track in degrees.
#     - heading_asc: Satellite heading for ascending track in degrees.
#     - heading_desc: Satellite heading for descending track in degrees.
    
#     Returns:
#     - vertical: Vertical displacement component.
#     - ew: East-West displacement component.
#     - ns: North-South displacement component.
#     """
    
#     # Convert angles from degrees to radians
#     los_angle_asc_rad = np.radians(los_angle_asc)
#     los_angle_desc_rad = np.radians(los_angle_desc)
#     heading_asc_rad = np.radians(heading_asc)
#     heading_desc_rad = np.radians(heading_desc)
    
#     # Calculate vertical, EW, and NS components
#     # Form the system of equations based on trigonometry and solve for vertical, EW, NS components
#     A = np.array([
#         [np.sin(los_angle_asc_rad), np.cos(los_angle_asc_rad) * np.sin(heading_asc_rad), -np.cos(los_angle_asc_rad) * np.cos(heading_asc_rad)],
#         [np.sin(los_angle_desc_rad), np.cos(los_angle_desc_rad) * np.sin(heading_desc_rad), -np.cos(los_angle_desc_rad) * np.cos(heading_desc_rad)]
#     ])
    
#     # For each pixel, solve the system
#     vertical = np.zeros_like(los_asc)
#     ew = np.zeros_like(los_asc)
#     ns = np.zeros_like(los_asc)
    
#     for i in range(los_asc.shape[0]):
#         for j in range(los_asc.shape[1]):
#             b = np.array([los_asc[i, j], los_desc[i, j]])
#             try:
#                 sol = np.linalg.lstsq(A, b, rcond=None)[0]
#                 vertical[i, j] = sol[0]
#                 ew[i, j] = sol[1]
#                 ns[i, j] = sol[2]
#             except np.linalg.LinAlgError:
#                 # Handle cases where the system cannot be solved
#                 vertical[i, j] = np.nan
#                 ew[i, j] = np.nan
#                 ns[i, j] = np.nan
    
#     return vertical, ew, ns

# #only from one orbit
# import rasterio
# import numpy as np

# # Load the LOS displacement raster data
# file_path = './data/morenny/radar/asc/VEL_Folder/2DVEL_simple.tif'

# # Assuming typical values for Sentinel-1 ascending track over mid-latitudes
# los_angle_asc = 34  # LOS angle in degrees, typical for Sentinel-1
# heading_asc = 0  # Heading angle in degrees, roughly north for ascending tracks

# def read_los_raster(file_path):
#     with rasterio.open(file_path) as src:
#         los_data = src.read(1)  # Read the first band
#         transform = src.transform
#         crs = src.crs
#     return los_data, transform, crs

# los_asc, transform, crs = read_los_raster(file_path)

# # Now, convert the LOS data to vertical displacement
# def los_to_vertical(los_data, los_angle):
#     los_angle_rad = np.radians(los_angle)
#     vertical = los_data / np.cos(los_angle_rad)
#     return vertical

# vertical_disp = los_to_vertical(los_asc, los_angle_asc)

# # If you want to save the vertical displacement to a new raster file
# output_path = './data/morenny/radar/asc/VEL_Folder/vertical_displacement.tif'
# with rasterio.open(output_path, 'w', driver='GTiff', height=vertical_disp.shape[0], width=vertical_disp.shape[1], count=1, dtype=vertical_disp.dtype, crs=crs, transform=transform) as dst:
#     dst.write(vertical_disp, 1)

# output_path


# '''
###Start###
from akhdefo_functions import akhdefo_viewer
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os
import rasterio as rio
import requests
from datetime import datetime, timedelta
import planet
import json
from planet import Session, data_filter
from planet import Session, OrdersClient
from pathlib import Path
import shutil
from osgeo import osr
import cmocean
import geopandas as gpd
import gstools as gs
      
import numpy as np
import geopandas as gpd
import gstools as gs
import matplotlib.pyplot as plt
import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
import rasterio
from rasterio.features import geometry_mask
from scipy.ndimage import median_filter
from skimage.filters import gaussian
import os

import asf_search as asf
import hyp3_sdk as sdk
import pandas as pd


import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from rasterio.mask import mask
from skimage import exposure

import warnings
import time 
from skimage.registration import phase_cross_correlation
import matplotlib
import pandas as pd
import tempfile

from scipy.ndimage import convolve
import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
import subprocess
from skimage.metrics import structural_similarity as ssim
# import required libraries
# from vidgear.gears import CamGear
# from vidgear.gears import StreamGear
import queue
import time

from osgeo import gdal, osr
import os
import gc
import time
from shapely.geometry import box
import os
import geopandas as gpd
from shapely.geometry import Point


def Akhdefo_resample(input_raster="", output_raster="" , xres=3.125 , yres=3.125, SavFig=False , convert_units=None):
    """
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
    """
    
   
    ds = gdal.Open(input_raster)

    # resample
    dsRes = gdal.Warp(output_raster, ds, xRes = xres, yRes = yres, 
                    resampleAlg = "bilinear")
    
    dsRes =None 
    ds = None
  
    # # visualize
    src=rasterio.open(output_raster)
    meta= src.meta
    meta.update({'nodata': np.nan})
    array=src.read(1, masked=True)
    # #array = dsRes.GetRasterBand(1).ReadAsArray()
    if convert_units is not None:
        array[array==-32767.0]=np.nan
        array=array*convert_units

    basename=os.path.splitext(os.path.basename(input_raster))[0]
    def create_fancy_figure(array, basename, SavFig=SavFig, convert_units=convert_units):
       
        # Creating the figure
        fig, ax = plt.subplots()
        # Making sure 'north' is up, in case Y was flipped
        #ax.invert_yaxis()
        # Creating a colormap
        cmap = plt.get_cmap('viridis')
        # Creating the image
        img = ax.imshow(array, cmap=cmap)
        # Creating the colorbar
        cbar = fig.colorbar(img, ax=ax, orientation='vertical', pad=0.01)
        
        # Adding labels to the axes
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

        # Adding a title to the plot
        ax.set_title(basename)

        if SavFig==True:
        # Saving the figure
            fig.savefig(basename, dpi=300, bbox_inches='tight')

        plt.imshow()

        return fig
    if SavFig==True:
        a=create_fancy_figure(array, basename, SavFig=SavFig, convert_units=convert_units)
    if convert_units is not None:
        
        src.close() # close the rasterio dataset
        os.remove(output_raster) # delete the file 
        
        rs=rasterio.open(output_raster, "w+", **meta)
        rs.write(array, indexes=1)
    plt.show()

def Akhdefo_inversion(horizontal_InSAR="", Vertical_InSAR="", EW_Akhdefo="", NS_Akhdefo="", output_folder=r"" , dem_path=None):
    
    """
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


    """
    
    
    ##################
        # Open and read the raster files
   # Open the raster files and read the data
    with rio.open(EW_Akhdefo) as src_east:
        east_data = src_east.read(1, masked=True)
        east_meta = src_east.meta

    with rio.open(horizontal_InSAR) as src_horizontal:
        horizontal_data = src_horizontal.read(1, masked=True)
        horizontal_meta = src_horizontal.meta

    with rio.open(NS_Akhdefo) as src_north:
        north_data = src_north.read(1, masked=True)
        north_meta = src_north.meta

    with rio.open(Vertical_InSAR) as src_vertical:
        vertical_data = src_vertical.read(1, masked=True)
        vertical_meta = src_vertical.meta
       
    from scipy.ndimage import zoom
    if dem_path is not None:
        with rio.open(Vertical_InSAR) as dem_src:
            dem = dem_src.read(1, masked=True)
            dem_meta = src_vertical.meta
            xres, yres=dem_src.res
        zoom_factor_dem = (north_data.shape[0] / dem.shape[0], north_data.shape[1] / dem.shape[1])
        resized_dem_data = zoom(dem, zoom_factor_dem, order=1)
    

    # Calculate the zoom factors for each dataset
    zoom_factor_east = (north_data.shape[0] / east_data.shape[0], north_data.shape[1] / east_data.shape[1])
    zoom_factor_horizontal = (north_data.shape[0] / horizontal_data.shape[0], north_data.shape[1] / horizontal_data.shape[1])
    zoom_factor_vertical = (north_data.shape[0] / vertical_data.shape[0], north_data.shape[1] / vertical_data.shape[1])
   

    # Resize each dataset
    resized_east_data = zoom(east_data, zoom_factor_east, order=1)  # Using order=1 (bilinear) for continuous data
    resized_horizontal_data = zoom(horizontal_data, zoom_factor_horizontal, order=1)
    resized_vertical_data = zoom(vertical_data, zoom_factor_vertical, order=1)
    
    D3D=np.hypot(resized_east_data, resized_vertical_data )

    plung_radians=np.arcsin(resized_vertical_data/D3D)
    plung_degree=np.degrees(plung_radians)
    
    DH=np.hypot(resized_east_data, north_data)

    trend_radians=np.arcsin(north_data/DH)
    trend_degrees=np.degrees(trend_radians)
    print ("Trend in degree raw data: ", trend_degrees.min(), trend_degrees.max())
    trend_degrees=(450 - trend_degrees ) % 360
    
    
    # if dem_path is not None:
    #     zoom_factor_D3D = (north_data.shape[0] / D3D.shape[0], north_data.shape[1] / D3D.shape[1])
    #     resized_D3D_data = zoom(D3D, zoom_factor_D3D, order=1)
        
    #     #elevation_change = np.tan(plung_radians) * resized_D3D_data
    #     # Example of combining with DEM (this part depends on what exactly you want to achieve)
    #     # Create masks for NaNs
    #     mask1 = np.isnan(resized_dem_data)
    #     mask2 = np.isnan(resized_vertical_data)

# Perform subtraction, treating NaNs as zeros
        # modified_dem = np.where(mask1, 0, resized_dem_data) - np.where(mask2, 0, resized_vertical_data)
   
    meta=north_meta
    
    ####################
    
    if not os.path.exists(output_folder ):
        os.makedirs(output_folder)
    # #Load images with rasterio
    # D_EW_InSAR=rio.open(horizontal_InSAR)
    # D_vertical_insar=rio.open(Vertical_InSAR)
    # D_EW_akhdefo=rio.open(EW_Akhdefo)
    # D_NS_akhdefo=rio.open(NS_Akhdefo)
    # #read images with rasterio
    # DEW_insar=D_EW_InSAR.read(1, masked=True)
    # #DEW_insar[DEW_insar==-32767.0]=np.nan
    # #DEW_insar=DEW_insar*1000

    # DEW_akhdefo=D_EW_akhdefo.read(1, masked=True)
    # D_vertical=D_vertical_insar.read(1, masked=True)
    # # D_vertical[D_vertical==-32767.0]=np.nan
    # # D_vertical=D_vertical*1000

    # DNS_akhdefo=D_NS_akhdefo.read(1, masked=True)

    # print (DEW_akhdefo.shape)
    # print(D_vertical.shape)
    # print (DEW_akhdefo.shape)
    # print(DEW_insar.shape)
    # DH=np.hypot(DEW_akhdefo, DNS_akhdefo)
    # D3D=np.hypot(DH, D_vertical )

    # meta=D_EW_InSAR.meta

    # trend_radians=np.arcsin(DNS_akhdefo/DH)
    # trend_degrees=np.degrees(trend_radians)
    # print ("Trend in degree raw data: ", trend_degrees.min(), trend_degrees.max())
    # trend_degrees=(450 - trend_degrees ) % 360

    # plung_radians=np.arcsin(D_vertical/D3D)
    # plung_degree=np.degrees(plung_radians)
    # #plung_degree=(90-plung_degree)% 90

    # print ("DH: ", DH.max(), DH.min())
    # print("D3D: ", D3D.max(), D3D.min())

    #Save products
    _3D_vel=output_folder + "/" + "D3D.tif"
    plung=output_folder+ "/" + "plung_degree.tif"
    trend=output_folder+ "/" + "trend_degrees.tif"
    # with rio.open("DH.tif", 'w', **meta) as dst:
    #         dst.write(DH, indexes=1)
    meta.update(nodata=np.nan)
    with rio.open(_3D_vel, 'w', **meta) as dst:
            dst.write(D3D, indexes=1)
    with rio.open(trend, 'w', **meta) as dst:
            dst.write(trend_degrees, indexes=1)
    with rio.open(plung, 'w', **meta) as dst:
            dst.write(plung_degree, indexes=1)
            
    # if dem_path is not None:
    #     dempath=output_folder+ "/" + "modified_dem.tif"
    #     with rio.open(dempath, 'w', **meta) as dst:
    #         dst.write(modified_dem, indexes=1)
        


    
    # p1=akhdefo_viewer(Path_to_DEMFile=demFile, rasterfile=_3D_vel , colorbar_label="mm/year", title="3D Velocity", pixel_resolution_meter=3.125, outputfolder=output_folder,
    # outputfileName="3D_Disp.jpg",  cmap=cmocean.cm.speed, alpha=0.8, noDATA_Mask=True, normalize=True)
    
    # p2=akhdefo_viewer(Path_to_DEMFile=demFile, rasterfile=plung , colorbar_label="degrees", title="Plunge of Dispalcement Velocity", pixel_resolution_meter=3.125, outputfolder=output_folder,
    # outputfileName="plunge.jpg", cmap=cmocean.cm.delta, alpha=0.8, noDATA_Mask=True, normalize=True)
    # p3=akhdefo_viewer(Path_to_DEMFile=demFile, rasterfile=trend , colorbar_label="degress", title="Trend of Dispalcement Velocity", pixel_resolution_meter=3.125, outputfolder=output_folder,
    # outputfileName="trend.jpg", cmap=cmocean.cm.phase, alpha=0.8, noDATA_Mask=True, normalize=True)
    
    akhdefo_viewer(path_to_dem_file=dem_path, raster_file=_3D_vel, output_folder=output_folder, title='3D Velocity', 
                   pixel_resolution_meters=None, output_file_name="3D_Disp.jpg", 
                   alpha=0.5, unit_conversion=None, no_data_mask=True, 
                   colormap=cmocean.cm.speed, min_value=None, max_value=None, 
                   normalize=True, colorbar_label=' ', show_figure=True)
    
    akhdefo_viewer(path_to_dem_file=dem_path, raster_file=plung, output_folder=output_folder, title='Plunge of Dispalcement Velocity', 
                   pixel_resolution_meters=None, output_file_name="plunge.jpg", 
                   alpha=0.5, unit_conversion=None, no_data_mask=True, 
                   colormap='hsv', min_value=None, max_value=None, 
                   normalize=True, colorbar_label='degrees', show_figure=True)
    
    akhdefo_viewer(path_to_dem_file=dem_path, raster_file=trend, output_folder=output_folder, title='Trend of Dispalcement Velocity', 
                   pixel_resolution_meters=None, output_file_name="trend.jpg", 
                   alpha=0.5, unit_conversion=None, no_data_mask=True, 
                   colormap=cmocean.cm.phase, min_value=None, max_value=None, 
                   normalize=True, colorbar_label='degrees', show_figure=True)


        
  
#from akhdefo_functions import mask_raster

def set_gdf_to_utm(gdf):
    # Ensure the GeoDataFrame has a valid CRS
    if gdf.crs is None:
        gdf.crs = "EPSG:4326"  # WGS 84
    
    # Calculate the median longitude to estimate the UTM zone
    median_lon = gdf.geometry.x.median()

    # Determine the UTM zone from the median longitude
    utm_zone = int(np.floor((median_lon + 180) / 6) + 1)
    
    # Check if it's in the northern or southern hemisphere
    median_lat = gdf.geometry.y.median()
    hemisphere = 'north' if median_lat >= 0 else 'south'
    
    # Construct the EPSG code for the UTM zone
    epsg_code = f"EPSG:{32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone}"
    
    # Set the GeoDataFrame to the calculated UTM CRS
    gdf.to_crs(epsg_code, inplace=True)
    return gdf 

def Auto_Variogram(data="", column_attribute="", latlon=False, aoi_shapefile="", pixel_size=20,num_chunks=10,overlap_percentage=0, out_fileName='interpolated_kriging', 
                   plot_folder='kriging_plots', geo_folder='geo_rasterFolder', smoothing_kernel=2, mask: [np.ndarray] = None , 
                   UTM_Zone=None, krig_method='ordinary' , drift_functions='linear', detrend_data=None, use_zscore=None):
    
     
    """
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
    
    """
    
    if isinstance(data, str):
        if data[-4:] == '.shp':
            geodata = gpd.read_file(data)
            
            if geodata.crs == "EPSG:4326":
                geodata=set_gdf_to_utm(geodata)
                
            crs_ini=geodata.crs
           
            
        else:
            raise ValueError("Unsupported file format.")
    elif isinstance(data, gpd.GeoDataFrame):
        geodata = data
        if geodata.crs == "EPSG:4326":
            geodata=set_gdf_to_utm(geodata)
        
    else:
        raise ValueError("Unsupported data type.")
    
    # Check for necessary columns in the shapefile
    #check_shapefile_columns(geodata, latlon)
    
    # Change the dtype of a specific column to float32
    geodata[column_attribute] = geodata[column_attribute].astype('float32')
    # Now interpolate NaN values in the column
    
    
    if latlon:
        # Convert CRS to EPSG:4326 (Wgs84)
        geodata = geodata.to_crs(epsg=4326)
        # if geodata.crs is not None and geodata.crs.to_epsg() == 4326: 
        x=geodata.geometry.x
        y=geodata.geometry.y
        
        # else:
        #     x=geodata.geometry.x
        #     y=geodata.geometry.y
            # x, y = utm_to_latlon(easting=geodata.geometry.x, northing=geodata.geometry.y, zone_number=10, zone_letter=UTM_Zone)
    else:
        x, y = geodata.geometry.x, geodata.geometry.y

    z = geodata[column_attribute]
    
    # Ensure x, y, and z are float32
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    z = geodata[column_attribute].astype(np.float32)
    
    # if geodata.crs is not None and geodata.crs.to_epsg() == 4326:
    #     latlon=True
    
    if latlon==True:
        try:
            #bins = gs.standard_bins(pos=(y, x), max_dist=10)
            #bins, sampling_size=2000, sampling_seed=19920516 ,
            bin_center, gamma = gs.vario_estimate((y, x), z,  latlon=latlon)
        except Exception as e:
        # Catching any exception and handling it
            print(f"Operation failed with error: {e}")
    else:
        try:
            bins = gs.standard_bins(pos=(y, x), max_dist=10)
            bin_center, gamma = gs.vario_estimate((x, y), z,  latlon=latlon)
        except Exception as e:
        # Catching any exception and handling it
            print(f"Operation failed with error: {e}")
    
    # Load AOI shapefile if provided
    if aoi_shapefile:
        aoi = gpd.read_file(aoi_shapefile)
        
        if not aoi.crs.equals(geodata.crs):
            aoi = aoi.to_crs(geodata.crs)
    
    # Models to test
    models = {
        "Gaussian": gs.Gaussian,
        "Exponential": gs.Exponential,
        "Matern": gs.Matern,
        "Integral": gs.Integral,
        "Cubic": gs.Cubic,
        "Stable": gs.Stable,
        "Rational": gs.Rational,
        "Spherical": gs.Spherical,
        "SuperSpherical": gs.SuperSpherical,
        "JBessel": gs.JBessel,
        "HyperSpherical": gs.HyperSpherical,
        "TPLSimple": gs.TPLSimple
    
        
    }

    scores = {}
    #fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(15, 10))
    

    import matplotlib.gridspec as gridspec
    # Setting up a professional style
    plt.style.use('seaborn-white')
    plt.rcParams.update({'font.size': 12})
    
    # Create a grid for the subplots
    fig = plt.figure(figsize=(18, 12))
    grid_fig = gridspec.GridSpec(2, 2, height_ratios=[1, 2])

    ax1=plt.subplot(grid_fig[0,:])
    ax1.scatter(bin_center, gamma, color="k", label="data")

    best_model, best_score, best_fit = None, -10, None
    
    successful_model = None

    
    for model_name, Model in models.items():
        try:
          
            if latlon==True:
                bin_center, gamma = gs.vario_estimate((y,x), z, latlon=True)
                fit_model = Model(dim=2, latlon=latlon)
                _, _, r2 = fit_model.fit_variogram( bin_center, gamma, sill=np.var(z), nugget=False, return_r2=True)
                scores[model_name] = {"model": fit_model, "score": r2}
                
                if r2 > best_score:
                    best_score = r2
                    best_model = model_name
                    best_fit = fit_model
                
            else:
                fit_model = Model(dim=2, latlon=latlon)
                #Model(dim=2, len_scale=4, anis=0.2, angles=-0.5, var=0.5, nugget=0.1)
                
                #if fit_model is not None:
            
                _, _, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True,  nugget=True)
                scores[model_name] = {"model": fit_model, "score": r2}
            
            # scores[model_name] = model_name
            # scores['score'] = r2
            # scores['best_fit']=fit_model
            
                if r2 > best_score:
                    best_score = r2
                    best_model = model_name
                    best_fit = fit_model
            
            # Plot the model
            #if fit_model is not None:
            fit_model.plot(x_max=max(bin_center), ax=ax1, label=f"{model_name} (R2: {r2:.5f})")
        except Exception as e:  # If there's any error
            print(f"Error with model {model_name}: {e}")
        
    ax1.legend()
    ax1.set_title('Variogram Models with Fitted Data')
    
    if best_fit is None:
        best_fit="spherical"
    
    # Sort scores based on score
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]['score'], reverse=True)
    def get_zgrid(x, y, z, best_fit , gridy, krig_method, drift_functions=drift_functions, use_zscore=use_zscore):
        
        z = z[~np.isnan(z)]
        x = x[~np.isnan(z)]
        y = y[~np.isnan(z)]
        
        x=np.array(x)
        y=np.array(y)
        z=np.array(z)
        
       
        # Calculate the mean and standard deviation of the Z column
        mean_z = np.mean(z)
        std_z = np.std(z)

        if use_zscore is not None:
            # Define the threshold for Z-score (e.g., ±2.5)
            z_score_threshold = use_zscore

            # Identify indices where Z is below or above the threshold
            below_threshold_indices = np.where(z < (mean_z - z_score_threshold * std_z))
            above_threshold_indices = np.where(z > (mean_z + z_score_threshold * std_z))

            # Replace values below or above the threshold with the mean of Z
            z[below_threshold_indices] = mean_z
            z[above_threshold_indices] = mean_z

        
        
        range_x =np.nanmax(x) - np.nanmin(x)
        num_points_x = int(range_x / pixel_size) + 1
        gridx = np.linspace(np.nanmin(x), np.nanmax(x), num_points_x)
        gridx = gridx.astype(np.float32)
        
        
        
        
        if krig_method=='ordinary':
            
            # OK = OrdinaryKriging(x, y, z, best_fit, weight=False, verbose=False, enable_plotting=False , exact_values=True, pseudo_inv=True)
            # z_grid, _ = OK.execute("grid", gridx, gridy)
            
            OK = gs.krige.Ordinary(model=best_fit, cond_pos=(y, x), cond_val=z , normalizer=None, trend=None, exact=False, cond_err='nugget', 
                                 pseudo_inv=True, pseudo_inv_type='pinvh', fit_normalizer=False, fit_variogram=True)
            OK.set_pos((gridy, gridx), mesh_type="structured")
            OK(return_var=True, store="vel")
            OK(only_mean=True, store="mean_vel")
            z_grid=OK["vel"]
            
    
        elif krig_method=='simple':
            SK = gs.krige.Simple(model=best_fit, cond_pos=(y, x), cond_val=z, mean=np.nanmean(z), normalizer=None, trend=None, exact=False, cond_err='nugget', 
                                 pseudo_inv=True, pseudo_inv_type='pinvh', fit_normalizer=False, fit_variogram=True)
            SK.set_pos((gridy, gridx), mesh_type="structured")
            SK(return_var=True, store="vel")
            SK(only_mean=True, store="mean_vel")
            z_grid=SK["vel"]
            
          
        
        elif krig_method=='universal':   
            # UK = UniversalKriging(x, y, z, variogram_model=best_fit, exact_values=True, pseudo_inv=True, drift_terms=["regional_linear"])
            # z_grid, _ = UK.execute("grid", gridx, gridy) 
            # drift_terms=["regional_linear"]
            if drift_functions=='x':
                def drift(x, y):
                    return x
            elif drift_functions=='y':
                def drift(x, y):
                    return y
            else:
                drift=drift_functions
            
            uk = gs.krige.Universal(model=best_fit, cond_pos=(y, x), cond_val=z, normalizer=None , trend=None, exact=True, cond_err='nugget', 
                                 pseudo_inv=True, pseudo_inv_type='pinvh', fit_normalizer=False, fit_variogram=True, drift_functions=drift)
            
            uk.set_pos((gridy, gridx), mesh_type="structured")
            uk(return_var=True, store="vel")
            uk(only_mean=True, store="mean_vel")
            z_grid=uk["vel"]
            #uk1.structured((gridx, gridy))
            # z_grid=z_grid[0]
            
            
           
        
        
        # from skimage import exposure
        z_min = np.nanmin(z)
        z_max = np.nanmax(z)
        
        # # Ensure interpolated values do not exceed original z data range
        
        #if np.nanmax(z_grid) > z_max or np.nanmin(z_grid) < z_min:
            # z_mean = np.nanmean(z_grid)
            # z_grid[np.where(z_grid > z_max)] = z_mean
            # z_grid[np.where(z_grid < z_min)] = z_mean
        z_grid = exposure.rescale_intensity(z_grid, in_range=(np.nanmin(z_grid), np.nanmax(z_grid)), out_range=(z_min, z_max))
            
        
        #z_grid=gs.normalizer.remove_trend_norm_mean((gridy, gridx), z_grid, mean=np.nanmean(z_grid), normalizer=None, trend=None, mesh_type='unstructured', value_type='scalar', check_shape=True, stacked=False, fit_normalizer=True)

            
        # z_grid[np.where(z_grid < z_min)] = z_mean
        # z_grid[np.where(z_grid > z_max)] = z_mean

        
       
        # normalizers = [
        #     gs.normalizer.BoxCox,
        #     gs.normalizer.YeoJohnson,
        #     gs.normalizer.Modulus,
        #     gs.normalizer.Manly,
        # ]
        
                # external drift at conditioning points
        # (given as a sinusodial drift in x direciton)
        # ext_drift_cond = np.sin(y)

        # # external drift at the output grid
        # ext_drift_grid = np.repeat(np.sin(gridx), len(gridy))


        # # perform the kriging and plot results
        # EDK = gs.krige.ExtDrift(
        #     model=best_fit, 
        #     cond_pos=(x, y), 
        #     cond_val=z,
        #     ext_drift=ext_drift_cond,
        # )
        # EDK.structured([gridy, gridx], ext_drift=ext_drift_grid)
        # EDK.plot()
    
        
        return z_grid, gridx
    
    
    # Function to detrend a 2D array with NaN handling
    from scipy.stats import linregress
    def detrend_2d(array):
        nrows, ncols = array.shape
        x = np.arange(nrows)
        
        # Initialize the detrended data array
        detrended = np.empty_like(array)

        # Detrend each column
        for col in range(ncols):
            y = array[:, col]
            if np.all(np.isnan(y)):
                detrended[:, col] = np.nan
            else:
                slope, intercept, _, _, _ = linregress(x[~np.isnan(y)], y[~np.isnan(y)])
                detrended[:, col] = y - (slope*x + intercept)

        return detrended 

    # Get global y grid before splitting
    range_y_global = y.max() - y.min()
    num_points_y_global = int(range_y_global / pixel_size) + 1
    gridy_global = np.linspace(y.min(), y.max(), num_points_y_global).astype(np.float32)
    
    # Get global x grid before splitting
    range_x_global = x.max() - x.min()
    num_points_x_global = int(range_x_global / pixel_size) + 1
    gridx_global = np.linspace(x.min(), y.max(), num_points_x_global).astype(np.float32)
    

    def get_transform_from_gdf(gdf, pixel_size):
        """
        Generate an affine transform for a raster-like GeoDataFrame.

        Parameters:
        -----------
        - gdf: The GeoDataFrame.
        - pixel_size: The resolution (spacing) of your points/data.

        Returns:
        ---------
        - A rasterio Affine object.
        """
        west, south, east, north = gdf.total_bounds
        transform = rasterio.transform.from_origin(west, north, pixel_size, pixel_size)
        extent=gdf.total_bounds
        # Calculate the number of rows and columns based on the extent and pixel size
        width = int((east - west) / pixel_size)
        height = int((north - south) / pixel_size)
        return transform, extent, width, height

    # # Break x, y, z into chunks
    # sorted_idxs = x.argsort()
    # chunks_x, chunks_y, chunks_z = [], [], []
    
    # for i in range(num_chunks):
    #     start_idx = int(i * len(x) / num_chunks)
    #     end_idx = int((i+1) * len(x) / num_chunks)
    #     chunks_x.append(x[sorted_idxs[start_idx:end_idx]])
    #     chunks_y.append(y[sorted_idxs[start_idx:end_idx]])
    #     chunks_z.append(z[sorted_idxs[start_idx:end_idx]])
    
    ############
    
    # Calculate overlap based on percentage of data (5% overlap in this example)
    overlap = overlap_percentage * (x.max() - x.min()) / num_chunks

    # This list will store all z_grids and their corresponding gridx to be merged later
    z_grids = []
    gridxs = []

    # Split and process data in chunks
    for i in range(num_chunks):
        # Define start and end for each chunk considering overlap
        start = x.min() + i * (x.max() - x.min()) / num_chunks - overlap
        end = x.min() + (i + 1) * (x.max() - x.min()) / num_chunks + overlap

        x_chunk = x[(x >= start) & (x <= end)]
        y_chunk = y[(x >= start) & (x <= end)]
        z_chunk = z[(x >= start) & (x <= end)]

        #z_grid_chunk, gridx_chunk = get_zgrid(x_chunk, y_chunk, z_chunk, best_fit, gridy_global)
        
        # Try using the best model, then next best if it fails, and so on
        for model_name, model_info in sorted_scores:
            try:
                z_grid_chunk, gridx_chunk = get_zgrid(x_chunk, y_chunk, z_chunk, model_info['model'], gridy_global, krig_method=krig_method, drift_functions=drift_functions, use_zscore=use_zscore)
                z_grids.append(z_grid_chunk)
                gridxs.append(gridx_chunk)
                successful_model = model_name
                # Use z_grid and gridx as needed
                break
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                continue
    print(f"kriging succeed with Model: {model_name} and score: {model_info['score']}")
        
        
        # z_grids.append(z_grid_chunk)
        # gridxs.append(gridx_chunk)
        
        # Update the label of the successful model
    if successful_model:
        for line in ax1.get_lines():
            if line.get_label() == successful_model:
                line.set_label(f"{successful_model} (Success)")
                break

    ax1.legend()
    ax1.set_title(f'Variogram Models with successful model:{successful_model}')
    
    if overlap > 0:
        # Blend the chunks together.
        for i in range(1, len(z_grids)):
            # Get overlapping columns
            overlap_cols = max(1, int(round(overlap / pixel_size)))

            # Create a combined section of the overlapping regions from both chunks
            combined_overlap = np.hstack((z_grids[i-1][:, -overlap_cols:], z_grids[i][:, :overlap_cols]))

            # if smoothing_kernel is not None:
                
            # # Apply gaussian filter to the combined overlap
            #     filtered_overlap = gaussian(combined_overlap, sigma=smoothing_kernel)
            # else:
            #     filtered_overlap=combined_overlap
            filtered_overlap=combined_overlap
            # Assign the filtered overlap back to the respective chunks
            z_grids[i-1][:, -overlap_cols:] = filtered_overlap[:, :overlap_cols]
            z_grids[i][:, :overlap_cols] = filtered_overlap[:, overlap_cols:]
    
    
    
    # # Function to blend two arrays
    # def blend_arrays(array1, array2, overlap_width=1):
    #     weights = np.linspace(0, 1, overlap_width).reshape(1, overlap_width)
    #     weights = np.repeat(weights, array1.shape[0], axis=0)  # Repeat for each row
        
    #     overlap1 = array1[:, -overlap_width:]
    #     overlap2 = array2[:, :overlap_width]
        
        
        
    #     blended_overlap = overlap1 * (1 - weights) + overlap2 * weights
    #     blended_overlap = gaussian(blended_overlap, sigma=3, mode='wrap', preserve_range=True)
        
    #     return np.hstack([array1[:, :-overlap_width], blended_overlap, array2[:, overlap_width:]])
   
    
        
    # Merge all chunks together
    z_grid = np.hstack(z_grids)
    
   
    transform_i, extent, width, height=get_transform_from_gdf(geodata, pixel_size)
    # Apply gaussian filter to the smooth edge artifcats
    if smoothing_kernel is not None:
        z_grid = gaussian(z_grid, sigma=smoothing_kernel, mode='nearest', preserve_range=True)
   
    def create_norm(data):
        """
        Create a normalization instance based on the data.
        If data has negative values, use TwoSlopeNorm with zero as the center.
        Otherwise, use standard normalization.
        """
        import matplotlib.colors as mcolors
        
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        if vmin < 0 and vmax > 0:
            # Data has negative values, use TwoSlopeNorm
            return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            # Data has no negative values, use standard normalization
            return mcolors.Normalize(vmin=vmin, vmax=vmax)

    # if mask is not None:
    #     z_grid=mask_raster(z_grid, mask )
    
    # Mask the interpolated values outside of AOI
    if aoi_shapefile:
        shp_crs=aoi.crs
        transform, extent, width, height=get_transform_from_gdf(geodata, pixel_size)
        shapes = aoi.geometry.values
        #transform = rasterio.transform.from_origin(min(gridx1.min(), gridx2.min()), max(gridy1.max(), gridy2.max()), abs(gridx1[1]-gridx1[0]), abs(gridy1[1]-gridy1[0]))
        mask_aoi = geometry_mask(shapes, transform=transform, out_shape=z_grid.shape, invert=False, all_touched=True)
        z_grid=z_grid.reshape(mask_aoi.shape)
        z_grid = np.where(mask_aoi,  np.nan, z_grid)
    else:
        transform, extent, width, height=get_transform_from_gdf(geodata, pixel_size)
        shp_crs=geodata.crs
    if mask is not None:
        z_grid[mask]=np.nan


    
    norm = create_norm(z_grid)
        
   
    ax2 = plt.subplot(grid_fig[1,0])
    plt.colorbar(ax2.imshow(z_grid, cmap='rainbow', extent=extent, norm=norm), ax=ax2)
    ax2.set_title(f'Interpolation: Krige-Method: {krig_method} kriging and Best Variogram Model: {successful_model} ')

    

    meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': np.nan,
        'width': width,
        'height': height,
        'count': 1,
        'crs': geodata.crs,
        'transform': transform
    }
    
    
    
    if not os.path.exists(geo_folder):
        os.makedirs(geo_folder)
    
    out_fileName_reg=out_fileName +"_"+ krig_method
    geo_folder_reg=geo_folder+"/"+ out_fileName_reg
        
    with rasterio.open(os.path.join(geo_folder_reg + '.tif'), 'w', **meta) as dst:
        dst.write(z_grid, 1)

    raster_data_detrended = detrend_2d(z_grid)
    # Detrended data
    cmap_c = 'coolwarm' if np.nanmin(raster_data_detrended) < 0 else 'rainbow'
    norm = create_norm(raster_data_detrended)
        # Detrended Interpolated Data
    ax3 = plt.subplot(grid_fig[1,1])
    plt.colorbar(ax3.imshow(raster_data_detrended, cmap=cmap_c, extent=extent, norm=norm), ax=ax3)
    ax3.set_title("Detrended Interpolated Data")
        
    
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    
    plot_folder=plot_folder+"/"+ out_fileName_reg
    
    fig.tight_layout()

    plt.savefig(os.path.join(plot_folder + '.png'))
    
   
    plt.close()
    
    
    geo_folder_detrend=geo_folder+"/detrend"
        

        
    if not os.path.exists(geo_folder_detrend):
        os.makedirs(geo_folder_detrend)
        
        
    # Apply the custom detrending function
    if detrend_data==True:
        
        with rasterio.open(os.path.join(geo_folder_detrend +"/" + out_fileName_reg+ '_detrend.tif'), 'w', **meta) as dst:
            dst.write(raster_data_detrended, 1)


    return z_grid

def check_shapefile_columns(geodata, latlon):
    required_columns = ['x', 'y'] if not latlon else ['lat', 'lon']
    for col in required_columns:
        if col not in geodata.columns:
            raise ValueError(f"Column '{col}' not found in the shapefile.")



async def akhdefo_download_planet(planet_api_key="", AOI="plinth.json", start_date= "May 1, 2016", end_date= "", limit=5, 
                           item_type="PSOrthoTile", product_bundle="analytic_sr_udm2", clear_percent=90, cloud_filter=0.1, output_folder="raw_data", clip_flag=True, download_data=False):

    ''' 
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
        PSScene:	PlanetScope 3, 4, and 8 band scenes captured by the Dove satellite constellation
        REOrthoTile:	RapidEye OrthoTiles captured by the RapidEye satellite constellation
        REScene:	Unorthorectified strips captured by the RapidEye satellite constellation
        SkySatScene:	SkySat Scenes captured by the SkySat satellite constellation
        SkySatCollect:	Orthorectified scene composite of a SkySat collection
        SkySatVideo:	Full motion videos collected by a single camera from any of the active SkySats
        Landsat8L1G	Landsat8 Scenes: provided by USgs Landsat8 satellite
        Sentinel2L1C:	Copernicus Sentinel-2 Scenes provided by ESA Sentinel-2 satellite

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


    '''
    
    

    #############
    # if your Planet API Key is not set as an environment variable, you can paste it below

    planet_api_key=planet_api_key
    if planet_api_key in os.environ:
        API_KEY = os.environ[planet_api_key]
    else:
        API_KEY = planet_api_key
        os.environ['PL_API_KEY'] = API_KEY
        
        
    # Setup the session
    session = requests.Session()

    # Authenticate


    #client = Auth.from_key(API_KEY)
    session.auth = (API_KEY, "")

    #########
    #orders_url = 'https://api.planet.com/compute/ops/orders/v2' 

    ############
    # response = requests.get(orders_url, auth=session.auth)
    # response
    # orders = response.json()['orders']
    # len(orders)


    ############



    # We will also create a small helper function to print out JSON with proper indentation.
    def indent(data):
        print(json.dumps(data, indent=2))

    #Searching
    #We can search for items that are interesting by using the quick_search member function. Searches,
    #however, always require a proper request that includes a filter that selects the specific items to return as seach results.
    
    if AOI.endswith(".shp"):
        
        shapefile_path = AOI

    # Read the shapefile using geopandas
        gdf = gpd.read_file(shapefile_path)

    # Convert to GeoJSON format
        geojson_str = gdf.to_json()
         # Save the GeoJSON to a file
        with open('output.geojson', 'w') as geojson_file:
            geojson_file.write(geojson_str)
        with open('output.geojson', 'r', encoding='ISO-8859-1') as f:
            geom = json.loads(f.read())
    else:
        with open(AOI) as f:
            geom = json.loads(f.read())
    
    base_name = os.path.basename(AOI)
    task_name = os.path.splitext(base_name)[0]
    #Shapefile AOI

    ######

    # Define the filters we'll use to find our data

    #item_types = [ "PSOrthoTile"]

    geom_filter = data_filter.geometry_filter(geom)
    clear_percent_filter = data_filter.range_filter('clear_percent', clear_percent)
    date_string_start = start_date  # Example string representing a date in "month, day, year" format
    format_string = "%B %d, %Y"  # Format of the input string
    datetime_object_start = datetime.strptime(date_string_start, format_string)

    
    if end_date=="":
        date_range_filter = data_filter.date_range_filter("acquired", gte=  datetime_object_start)
    else:
        date_string_end = end_date  # Example string representing a date in "month, day, year" format
        format_string = "%B %d, %Y"  # Format of the input string
        datetime_object_end = datetime.strptime(date_string_end, format_string)

        date_range_filter = data_filter.date_range_filter("acquired",  gt=  datetime_object_start, lte=datetime_object_end)

    #Date Filter range
    Date_Range_Filter2={
    "type":"DateRangeFilter",
    "field_name":"acquired",
    "config":{
        "gt":"2019-12-31T00:00:00Z",
        "lte":"2020-05-05T00:00:00Z"
    }
    }
    cloud_cover_filter = data_filter.range_filter('cloud_cover', None, cloud_filter)

    combined_filter = data_filter.and_filter([geom_filter, clear_percent_filter, date_range_filter, cloud_cover_filter])

    async with Session() as sess:
        cl = sess.client('data')
        request = await cl.create_search(name='planet_client_demo',search_filter=combined_filter, item_types=[item_type])
        
    async with Session() as sess:
        cl = sess.client('data')
        items = cl.run_search(search_id=request['id'], limit=limit)
        item_list = [i async for i in items]
        
    #####################################
    
    item_ids=[item['id'] for item in item_list]

    print ("\033[1m Number of Items to Download: \033[1m", len(item_ids),  "\n" +" \033[1m Set download_data=True to download the results...\033[1m")
    for item in item_list:
            print(item['id'], item['properties']['item_type'])


    

    print ("\033[1m Number of Items to Download: \033[1m", len(item_ids),  "\n" +" \033[1m Strat Downloading...\033[1m")

    


    #########################

    ################
    # define the clip tool
    clip = {
        "clip": {
            "aoi": geom
        }
    }

    single_product = [
        {
        "item_ids": item_ids,
        "item_type": "PSScene",
        "product_bundle": "analytic_sr_udm2"
        }
    ]
    same_src_products = [
        {
        "item_ids":item_ids,
        "item_type": item_type,
        "product_bundle": product_bundle
        }
    ]
    multi_src_products = [
        {
        "item_ids": item_ids,
        "item_type": "PSScene",
        "product_bundle": "analytic_udm2"
        },
        {
        "item_ids": item_ids,
        "item_type": "PSOrthoTile",
        "product_bundle": "analytic_sr_udm2"
        },
        
    ]
    # create an order request with the clipping tool
    if clip_flag==False:
        request_clip = {
    "name": "just clip",
    "products": same_src_products,
    "delivery": {  
        "archive_type": "zip",
        "archive_filename": "{{name}}_{{order_id}}.zip"
    }
    }

    else:
            
        request_clip = {
        "name": task_name,
        "products": same_src_products,
        "tools": [clip], "delivery": {  
            "archive_type": "zip",
            "archive_filename": "{{name}}_{{order_id}}.zip"
        }
        }

    output_folder=output_folder+"_"+task_name
    from planet import Auth, reporting
    async def poll_and_download(order):
        async with Session() as sess:
            cl = OrdersClient(sess)

            # Use "reporting" to manage polling for order status
            with reporting.StateBar(state='creating') as bar:
                # Grab the order ID
                bar.update(state='created', order_id=order['id'])

                # poll...poll...poll...
                
                try:

                    await cl.wait(order['id'], callback=bar.update_state, max_attempts=2000)
                except:
                    pass
                try:

                # if we get here that means the order completed. Yay! Download the files.
                    filenames = await cl.download_order( order_id=order['id'], directory=output_folder, overwrite=False,  progress_bar=True)
                except:
                    pass

    async with Session() as sess:

        max_attempts = len(item_ids)
        print("Check for bad items...")
        errors_list= []
        cl = OrdersClient(sess)
        for attempt in range(max_attempts):

            try:
                
                # Setup the session
                # session = requests.Session()
                # # # set content type to json
                # headers = {'content-type': 'application/json'}
                # orders_url = 'https://api.planet.com/compute/ops/orders/v2'
                # response = requests.post(orders_url, data=json.dumps(request_clip), auth=session.auth, headers=headers)
                # print(response)
                # order = response.json()['id']

                order = await cl.create_order(request_clip)

                #download =  await poll_and_download(order)
            except planet.exceptions.APIError as e:
                print(e)
                errors_list.append(str(e))
                # Remove the items from the list that are present in the assets_in_errors list
                

                #print ("new_list", len(errors_list_dict), print(errors_list_dict))
            
                pass
        
            # Extract the assets from the error messages
            
            print(errors_list)
            
            # Convert the JSON string to a Python object
            #error_dicts = [json.loads(s) for s in errors_list]

            
            # Convert the JSON strings to Python dictionaries
            dictionaries = [json.loads(json_string) for json_string in errors_list]

            # Function to recursively extract all values from a dictionary
            def extract_values(obj):
                if isinstance(obj, dict):
                    for value in obj.values():
                        if value is not None:
                            yield from extract_values(value)
                elif isinstance(obj, list):
                    for item in obj:
                        if item is not None:
                            yield from extract_values(item)
                else:
                    yield obj

            # Extract all values from the dictionaries
            all_values = []
            for dictionary in dictionaries:
                all_values.extend(list(extract_values(dictionary)))

            # Check if any keyword is in any of the values
            bad_ids = [keyword for keyword in item_ids if any(keyword in value for value in all_values)]

            print('Matches found:', bad_ids)

                # # Initialize an empty list to store the IDs
                # bad_ids = []

                # # Go through each error message in each dictionary
                # for error_dict in error_dicts:
                #     for detail in error_dict["field"]["Details"]:
                #         message = detail["message"]
                #         # Extract the ID from the message
                #         id = message.split('Bundle type/')[1].split('/')[0]
                #         # Add the ID to the list
                #         bad_ids.append(id)
                

            print("bad items has been removed from the item list to avoid api error: \n", bad_ids)
            item_ids_updated = [item for item in item_ids if item not in bad_ids]

            print("Below is list of final items to be downloaded \n " "Number of Items: ", len(item_ids_updated), "\n",  item_ids_updated)


            same_src_products = [
            {
            "item_ids":item_ids_updated,
            "item_type": item_type,
            "product_bundle": product_bundle
            }
                ]

            
            if clip_flag==False:
                request_clip = {
            "name": "just clip",
            "products": same_src_products,
            "delivery": {  
                "archive_type": "zip",
                "archive_filename": "{{name}}_{{order_id}}.zip"
            }
            }

            else:
                    
                request_clip = {
                "name": task_name,
                "products": same_src_products,
                "tools": [clip], "delivery": {  
                    "archive_type": "zip",
                    "archive_filename": "{{name}}_{{order_id}}.zip"
                }
                }


                
        order = await cl.create_order(request_clip)

        if download_data==True:
            download =  await poll_and_download(order)
            
            
        print("Downloading Data is Completed :D")
    #task1=asyncio.current_task(await download_planet())
       

        def search_and_move_zip_files(source_folder, destination_folder):
            # Iterate through all files and directories within the source folder
            for root, dirs, files in os.walk(source_folder):
                for file in files:
                    # Check if the file is a zip file
                    if file.endswith('.zip'):
                        # Get the full path of the zip file
                        zip_file_path = os.path.join(root, file)
                        
                        # Move the zip file to the destination folder
                        shutil.move(zip_file_path, destination_folder)

    
        

        # Specify the directory path
        directory_path = Path("zip_folders")

        # Create the directory
        directory_path.mkdir(parents=True, exist_ok=True)
        # Specify the source folder to search for zip files
        source_folder = output_folder

        # Specify the destination folder to move the zip files
        destination_folder = directory_path

        # Call the function to search and move zip files
        search_and_move_zip_files(source_folder, destination_folder)


def akhdefo_orthorectify(input_Dir: str, dem_path: str, output_path: str, ortho_usingRpc: bool, file_ex=".tif"):
    
    ''' 
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
     
    
    '''
    

    # Define the paths to your image, DEM, and output
    #input_Dir = glob.glob(input_Dir +"/"+ "*.tif")

    

    def get_file_paths(directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(file_ex):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    # replace 'your_directory_path' with the path to the directory you want to iterate through
    input_Dir = get_file_paths(input_Dir)
    if not os.path.exists(output_path):
            os.makedirs(output_path)
    
    for n in range(len(input_Dir)):
        image_path=input_Dir
        print(image_path[n], n)
        file_name = os.path.basename(image_path[n])
        
        output_path1 = output_path +"/"+ file_name
        print(output_path)

        # Open the DEM
        dem_ds = gdal.Open(dem_path)

        # Get the coordinate system of the DEM
        dem_srs = osr.SpatialReference()
        dem_srs.ImportFromWkt(dem_ds.GetProjectionRef())

        # Define the gdalwarp options
        warp_options = gdal.WarpOptions(
            format='GTiff',
            rpc=ortho_usingRpc,
            outputType=gdal.GDT_UInt16,
            multithread=True,
            creationOptions=['COMPRESS=DEFLATE', 'TILED=YES'],
            dstSRS=dem_srs.ExportToProj4(),
            srcNodata=0,
            dstNodata=0,
            resampleAlg='cubic',
        )

        # Open the source image
        src_ds = gdal.Open(image_path[n])

        # Perform the orthorectification
        dst_ds = gdal.Warp(output_path1, src_ds, options=warp_options, cutlineDSName=dem_path)

    #Clean up
    src_ds = None
    dem_ds = None
    dst_ds = None




def download_RTC(username: str = '', password: str = '', prompt=False, asf_datapool_results_file: str = '', save_dir: str = '',
                 job_name: str = 'rtc-test', dem_matching: bool = True, include_dem: bool = True,
                 include_inc_map: bool = True, include_rgb: bool = False, 
                 include_scattering_area: bool = False, scale: str = 'amplitude',
                 resolution: int = 10, speckle_filter: bool = True, radiometry='gamma0', dem_name='copernicus',
                   download: bool=False, limit: int =None , path_number: int =None, frame_number: int =None,
                   RTC: bool =False, autorift: bool =False , insar: bool =False, max_neighbors: int =2):
    """
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
    """

    if prompt==True:
        if not asf_datapool_results_file or not save_dir:
            raise ValueError("asf_datapool_results_file, and save_dir are required.")

        hyp3 = sdk.HyP3(prompt=prompt)
    else:
            
        # Check if required parameters are provided
        if not username or not password or not asf_datapool_results_file or not save_dir:
            raise ValueError("Username, password, asf_datapool_results_file, and save_dir are required.")

        # Establish connection
        hyp3 = sdk.HyP3(username=username, password=password)

    # Read ASF datapool results file
    df = pd.read_csv(asf_datapool_results_file)
    col_list=['Path Number', "Frame Number"]
    unique_values = {col: df[col].dropna().unique().tolist() for col in col_list}
    print (unique_values)
    if path_number is not None:
        df=df[df['Path Number'] == path_number]
    if frame_number is not None:
        df=df[df['Frame Number'] == frame_number]
          
    granule_names = df['Granule Name'].tolist()
    
    
    if limit is not None:
        granule_names=granule_names[:limit]
        
    

    # Select first two granules
    granules_to_download = granule_names
    

    
    ######
    from typing import Optional
    from tqdm.auto import tqdm
    def get_nearest_neighbors(granule: str, max_neighbors: Optional[int] = None) -> asf.ASFSearchResults:
        granule = asf.granule_search(granule)[-1]
        stack = reversed([item for item in granule.stack() if item.properties['temporalBaseline'] < 0])
        return asf.ASFSearchResults(stack)[:max_neighbors]
    
    #######
    
    if download==True:
        print(f"Downloading granules: {granules_to_download}")

        # Submit jobs for each granule
        if RTC:
            
            # Initialize job batch
            rtc_jobs = sdk.Batch()
            for granule in granules_to_download:
                rtc_jobs += hyp3.submit_rtc_job(granule, name=job_name, dem_matching=dem_matching, 
                                                include_dem=include_dem, include_inc_map=include_inc_map,
                                                include_rgb=include_rgb, 
                                                include_scattering_area=include_scattering_area,
                                                scale=scale, resolution=resolution, 
                                                speckle_filter=speckle_filter, radiometry=radiometry, dem_name=dem_name)
        
            print(f"Submitted jobs: {rtc_jobs}")
            # Monitor job progress and wait until completion
            rtc_jobs = hyp3.watch(rtc_jobs)

            # Download completed jobs to the specified location
            rtc_jobs.download_files(location=save_dir+"_rtc", create=True)
            
        if autorift:
            autorift_jobs = sdk.Batch()
            def create_pairs(lst):
                return [(lst[i], lst[i+1]) for i in range(len(lst) - 1)]
            
            autorift_pairs = create_pairs(granules_to_download)
            
            for reference, secondary in autorift_pairs:
                autorift_jobs += hyp3.submit_autorift_job(reference, secondary, name=job_name+"_autorift")
            print(autorift_jobs)
                # Monitor job progress and wait until completion
            autorift_jobs = hyp3.watch(autorift_jobs)

            # Download completed jobs to the specified location
            autorift_jobs.download_files(location=save_dir+"_autorift", create=True)
                
        if insar:
            insar_jobs = sdk.Batch()
            for reference in tqdm(granules_to_download):
                neighbors = get_nearest_neighbors(reference, max_neighbors=max_neighbors)
                for secondary in neighbors:
                    insar_jobs += hyp3.submit_insar_job(reference, secondary.properties['sceneName'], name='insar-example')
            print(insar_jobs)
            insar_jobs = hyp3.watch(insar_jobs)

            # Download completed jobs to the specified location
            insar_jobs.download_files(location=save_dir+"_insar", create=True)
        

        # # Monitor job progress and wait until completion
        # rtc_jobs = hyp3.watch(rtc_jobs)

        # # Download completed jobs to the specified location
        # rtc_jobs.download_files(location=save_dir, create=True)
    else:
        print(f"set value download= {download} to True to submit the jobs and download RTC data")
        print(f"List of granules: {granules_to_download}")
    
    
   



def reproject_raster_to_match_shapefile(src_path, dst_path, dst_crs):

    """
    Reproject a raster to match the coordinate reference system (CRS) of a shapefile.

    Parameters:
    ------------
    - src_path (str): Path to the source raster file that needs to be reprojected.
    - dst_path (str): Path to save the reprojected raster.
    - dst_crs (CRS or str): Target coordinate reference system.

    Returns:
    ---------
    None. The reprojected raster is written to dst_path.
    """

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

def create_vegetation_mask(red_band_path, nir_band_path, output_path, shapefile_path, threshold=0.3, save_plot=False, plot_path="plot.png"):
    
    """
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
    """

    # Read the AOI from the shapefile
    aoi_gdf = gpd.read_file(shapefile_path)
    geometry = aoi_gdf.geometry[0]  # Assuming only one geometry in the shapefile

    target_crs = aoi_gdf.crs

    # Reproject the rasters to match the CRS of the shapefile
    reprojected_red_band_path = "reprojected_red.tif"
    reprojected_nir_band_path = "reprojected_nir.tif"

    reproject_raster_to_match_shapefile(red_band_path, reprojected_red_band_path, target_crs)
    reproject_raster_to_match_shapefile(nir_band_path, reprojected_nir_band_path, target_crs)

    # Open and crop the reprojected red band using the AOI
    with rasterio.open(reprojected_red_band_path) as red_src:
        red_crop, red_transform = mask(red_src, [geometry], crop=True)

    # Open and crop the reprojected near-infrared band using the AOI
    with rasterio.open(reprojected_nir_band_path) as nir_src:
        nir_crop, _ = mask(nir_src, [geometry], crop=True)

    # Calculate NDVI
    # Using a small value (1e-8) in the denominator to prevent division by zero
    ndvi = (nir_crop - red_crop) / (nir_crop + red_crop )

   # Use the rescale_intensity function from skimage's exposure module
    #ndvi = exposure.rescale_intensity(ndvi, out_range=(-1, 1))


    # Create a binary vegetation mask
    vegetation_mask = (ndvi < threshold)
    
    

    # Prepare metadata for the output raster
    meta = red_src.meta.copy()
    meta.update({
        "height": red_crop.shape[1],
        "width": red_crop.shape[2],
        "transform": red_transform
    })

    # Save the vegetation mask raster
    with rasterio.open(output_path, 'w', **meta) as dest:
        dest.write(vegetation_mask)

    if save_plot:
        plt.colorbar(plt.imshow(vegetation_mask[0], cmap='gray'))
        plt.axis('off')
        plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
        plt.show()

    print(f"Vegetation mask created and saved to {output_path}")
    

def adjust_brightness(frame, brightness=50):
    """Adjust the brightness of an image.
    brightness > 0 will increase brightness
    brightness < 0 will decrease brightness
    """
    # Ensure the brightness adjustment does not go out of bounds
    adjusted_frame = frame + brightness
    adjusted_frame = np.clip(adjusted_frame, 0, 255).astype(np.uint8)
    return adjusted_frame

import cv2
import numpy as np
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
import subprocess
from skimage.metrics import structural_similarity as ssim
# import required libraries
#from vidgear.gears import CamGear
import time
from scipy.ndimage import shift as shift_cor

# Function to measure displacement using Dense Optical Flow
def measure_displacement_from_camera(hls_url, alpha=0.1, save_output=False, output_filename=None, ssim_threshold=0.4, 
                                     pyr_scale=0.5, levels=15, winsize=32,iterations= 3, poly_n=5,poly_sigma= 1.5, flags=1 , show_video=True, streamer_option='mag'):
    
    '''
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
    
    '''
    #import datetime
    
   
        # Suppress the Matplotlib GUI warning
    warnings.filterwarnings("ignore", category=UserWarning, message="Starting a Matplotlib GUI outside of the main thread will likely fail")

    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=48)
    backSub1 = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=48)
    cap = cv2.VideoCapture(hls_url)
    
    frame_rate = int(cap.get(5))
   
    
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return
    
    global frame_width, frame_height
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
  
    
    ret, prev_frame = cap.read()
    # Set the new dimensions for resizing
    new_width = 640
    new_height = 480
    # Resize the frame to the new dimensions
    #prev_frame = cv2.resize(prev_frame, (new_width, new_height))
   
    prev_frame_init=prev_frame
    # Check if the current directory has "mask.png"
    mask_path = Path("./mask.png")
    masked_poly=None
    if mask_path.exists():
        masked_poly=cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
        # Example for a 3D prev_frame (e.g., RGB image)
        prev_frame = cv2.bitwise_and(prev_frame, prev_frame, mask=masked_poly)
        
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
   

    frame_count = 0
    
    # Initialize lists to store frame datetimes and displacements
    frame_datetimes = []
    min_displacements = []
    max_displacements = []
    mean_displacement=[]

    # Lists to store processing times
    processing_times = []
    frame_numbers = []
    std_frames=[]
    shift_list_x=[]
    shift_list_y=[]
    
    # Initialize the lists to hold the frames and g frames
    profile_list = []
    imagescreenshot_list = []
    
    plot_folder='plots'
    data_folder='plot_data'
    os.makedirs(plot_folder, exist_ok=True)
    os.makedirs(data_folder, exist_ok=True)
    
     # Calculate frames to skip
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frames_between_saves = int(fps * 3600)
    #frame_suffix=1
    
    ##Save to video
    # Adjust 'XVID' with 'MP4V' or another codec if needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #MJPG
    
    
    #############################
    

    def detect_outliers(image, threshold=3):
        """Detect outliers based on deviation from local mean."""
        mean_filter = np.ones((3, 3)) / 9.0
        local_mean = convolve(image.astype(float), mean_filter)
        deviation = np.abs(image - local_mean)
        return deviation > threshold

    def simple_kriging(image, outliers):
        """A very simplified kriging-like interpolation for outliers."""
        new_image = image.copy()  # New image has the exact same dimensions as the original
        Y, X = np.indices(image.shape)
        for y, x in zip(*np.where(outliers)):
            # Consider a 5x5 neighborhood around the outlier
            x_min, x_max = max(0, x - 2), min(image.shape[1], x + 3)
            y_min, y_max = max(0, y - 2), min(image.shape[0], y + 3)
            
            # Exclude the outlier itself
            x_idx, y_idx = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            valid_mask = (x_idx != x) | (y_idx != y)
            
            # Calculate distances to the outlier
            distances = np.sqrt((x_idx - x)**2 + (y_idx - y)**2)
            
            # Inverse distance weighting
            weights = 1 / (distances + 1e-10)
            weights = weights[valid_mask]
            
            # Apply weights to compute new value for outlier
            new_value = np.average(image[y_min:y_max, x_min:x_max][valid_mask], weights=weights)
            new_image[y, x] = new_value
        
        return new_image
    
    
    ############################
    
    import logging

    def set_video_bitrate(input_video_path, output_video_path, bitrate_kbps):
        """
        Set the bitrate of a video using FFmpeg. If the output file already exists, a new file with a suffix is created instead.

        Parameters:
        - input_video_path: Path to the input video file.
        - output_video_path: Path where the output video with the new bitrate will be saved.
        - bitrate_kbps: Desired video bitrate in kbps.
        """
        try:
            # Check if the output file already exists
            original_output_path = output_video_path
            counter = 1
            while os.path.exists(output_video_path):
                # Split the output path into directory, base name, and extension
                dir_name, base_name = os.path.split(original_output_path)
                file_name, file_extension = os.path.splitext(base_name)
                
                # Append a suffix to make the file name unique
                output_video_path = os.path.join(dir_name, f"{file_name}_{counter}{file_extension}")
                counter += 1
            # Build the FFmpeg command
            command = [
                'ffmpeg',
                '-i', input_video_path,
                '-b:v', f'{bitrate_kbps}k',
                '-bufsize', f'{int(bitrate_kbps * 2)}k',  # Optional: Adjust buffer size
                output_video_path
            ]

            subprocess.run(command, check=True)
            logging.info(f"Video with bitrate {bitrate_kbps} kbps saved as {output_video_path}")
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to set bitrate: {e}")
            return False
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return False
    
    def delete_older_mp4(folder_path, bitrate_kbps):
        # Convert string path to Path object for easier manipulation
        folder = Path(folder_path)
        
        # Ensure the "converted" subfolder exists
        converted_folder = folder / "converted"
        converted_folder.mkdir(exist_ok=True)
        
        # Filter out all mp4 files within the specified folder
        mp4_files = list(folder.glob('*.mp4'))
        
        # Only proceed if there is at least one mp4 file
        if len(mp4_files) > 0:
            # Get the last modified times of the files
            file_mod_times = [(file, os.path.getmtime(file)) for file in mp4_files]
            
            # Sort the files based on modification time (most recent last)
            sorted_files = sorted(file_mod_times, key=lambda x: x[1], reverse=True)
            
            # Exclude the most recent file from conversion
            files_to_convert = sorted_files [1:]  # All except the most recent
            
            for file, _ in files_to_convert:
                # Define the output path for the converted file
                converted_output_path = converted_folder / file.name
                
                # Apply video conversion here; example function call (set_video_bitrate)
                # Assuming set_video_bitrate function is defined to take source path, output path, and bitrate
                #set_video_bitrate(file, converted_output_path, bitrate_kbps)
                attempt = 0
                success = False  # Flag to indicate success
                while attempt < 5:
                    try:
                        set_video_bitrate(file, converted_output_path, bitrate_kbps)
                        success = True  # Operation was successful
                        print(f"Operation successful on attempt {attempt+1}.")
                        break  # Operation succeeded, exit the loop
                    except OSError as e:
                        if e.winerror == 32:  # Only retry on "file in use" errors
                            print(f"Attempt {attempt+1} failed, file is in use. Retrying in {5} seconds...")
                            time.sleep(5)
                            attempt += 1
                        else:
                            print(f"Operation failed due to an unexpected error: {e}")
                            break  # Exit the loop if it's not a "file in use" error
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                        break  # Exit the loop on any other exception

                if not success:
                    print(f"Operation skipped after {5} attempts.")
                            
                    print(f"Converted and saved: {converted_output_path}")
                try:
                    os.remove(file)
                except Exception as e:  # Catching all exceptions
                    pass  # If an error occurs, just skip the deletion and continue the program


    def delete_files_except_last_n(directory, n=2):
        """Delete all files in the given directory except the last n files based on modification time."""
        files = list(Path(directory).glob('*'))  # List all items in the directory
        files = [f for f in files if f.is_file()]  # Filter out subdirectories
        files.sort(key=os.path.getmtime, reverse=True)  # Sort files by modification time, newest first

        
        # Delete all but the last n files
        if len(files) > n:
            for file in files[n:]:
                os.remove(file)
                #print(f"Deleted {file}")

    def find_and_clean_stats_folders(root_directory):
        """Find 'stats' folders in the given root_directory and its subdirectories, then clean them."""
        for dirpath, dirnames, filenames in os.walk(root_directory):
            for dirname in dirnames:
                if dirname == 'stats':
                    stats_path = os.path.join(dirpath, dirname)
                    #print(f"Cleaning 'stats' folder: {stats_path}")
                    delete_files_except_last_n(stats_path, 2)

    def find_polygons(image):
        """
        Finds regions with pixels of value 1 and identifies polygons around them.

        Parameters:
        - image: A binary image (numpy array) where pixels of interest have a value of 1.

        Returns:
        - A list of polygons, where each polygon is represented by a list of its vertex coordinates.
        """
        image = np.uint8(image)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            polygon = approx.reshape(-1, 2).tolist()
            polygons.append(polygon)
        return polygons

    def draw_circles_on_image(polygons, base_image):
        """
        Draws circles on the base_image based on the polygons provided.

        Parameters:
        - polygons: A list of polygons, each represented by a list of its vertex coordinates.
        - base_image: The image (numpy array) on which circles will be drawn.

        Returns:
        - The image with drawn circles.
        """
        for polygon in polygons:
            # Convert polygon points to a numpy array
            points = np.array(polygon, dtype=np.float32)
            
            # Calculate the minimal enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(points)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Draw the circle on the base image
            cv2.circle(base_image, center, radius, (0, 255, 0), 2)  # Change color and thickness as needed
            # Iterate over all contours and draw them on the image as red polylines with thickness of 5

        return base_image

    def replace_outliers_with_nan_fixed(data, z_threshold=5):
        """
        Replaces outliers in a 2D array with NaN based on a Z-score threshold, with a fix for data type issue.
        
        Parameters:
        - data: 2D NumPy array
        - z_threshold: Z-score threshold to identify outliers
        
        Returns:
        - A 2D array with outliers replaced by NaN
        """
        # Ensure data is a NumPy array and convert to float type to accommodate NaNs
        #data = np.asarray(data, dtype=float)
        
        # Calculate mean and standard deviation along columns, ignoring NaN values
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)

        # Safely calculate Z-scores, avoiding division by zero
        z_scores = np.where(stds > 0, (data - means) / stds, 0)

        # Identify outliers
        outliers = np.abs(z_scores) > z_threshold

        # Replace outliers with the respective column means
        for i in range(data.shape[1]):
            data[outliers[:, i], i] = means[i]
        
        return data
    
    def mask_outside_contour(contour, gray_img):
        """
        Masks everything outside the given contour in the grayscale image with NaN.
        
        Parameters:
        - contour: The contour (as obtained from cv2.findContours) to mask inside of.
        - gray_img: The grayscale image (2D numpy array).
        
        Returns:
        - The modified image with areas outside the contour masked as NaN.
        """
        # Create a mask with the same dimensions as the image, filled with zeros (black)
        mask = np.zeros_like(gray_img, dtype=np.uint8)
        
        # Fill the contour on the mask with white (255)
        cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)
        
        # Convert mask to boolean where true is the area inside the contour
        mask_bool = mask.astype(bool)
        
        # Create a copy of the image to modify
        masked_img = gray_img.copy().astype(np.float32)  # Convert to float to allow NaN
        
        # Set pixels outside of the contour to NaN
        masked_img[~mask_bool] = np.nan
        
        return masked_img
    # Variable to hold the last successful colorbar
    last_successful_colorbar  = None
    last_successful_image = None
    
    combined_frame=None
    # Initialize the background subtractor
    avg_frame = None
    alpha = 0.1  # Weight for the running average
    # Initialize the ORB detector
    #orb = cv2.ORB_create()
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    # Initialize variables for ORB feature tracking
    #prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_frame, None)
    prev_keypoints, prev_descriptors = sift.detectAndCompute(prev_frame, None)
    # The directory where you want to create subfolders
    base_directory = "plots/TimeLaps_Images"
    os.makedirs(base_directory, exist_ok=True)
    # Initially set this to the current hour
    last_hour = datetime.now().hour
    now=datetime.now()
    frame_suffix = "_0"
    release_flag=False
    while True:
        ret, frame = cap.read()
        ret2, frame2 = cap.read()
        # Resize the frame to the new dimensions
        #frame = cv2.resize(frame, (new_width, new_height))
        #if not ret:
            #break
        
        frame_init=frame
        
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        current_hour = current_time.hour
        # Create a folder name based on the current timestamp
        folder_name = current_time.strftime("%Y-%m-%d_%H")
        full_path_img = os.path.join(base_directory, folder_name)
        full_path_stats = os.path.join(full_path_img, 'stats')
          # Create the new folder
        # os.makedirs(full_path_img, exist_ok=True)
        # os.makedirs(full_path_stats, exist_ok=True)
        
        # Check if the hour has changed
        if  current_hour != last_hour:
            # Update last_hour to the current hour
            last_hour = current_hour
            frame_count=0
        #if  frame_count % 1000 == 0:
            # Capture the current timestamp
            now = datetime.now()
            release_flag=True
            ## reset lists #######3
            frame_datetimes.clear()
            min_displacements.clear()
            max_displacements.clear()
            mean_displacement.clear()
            # Lists to store processing times
            processing_times.clear()
            frame_numbers.clear()
            std_frames.clear()
            shift_list_x.clear()
            shift_list_y.clear()
            profile_list.clear()
            imagescreenshot_list.clear()
            delete_older_mp4('plots/Videos', 1500)
            
            frame_suffix = f"_{frame_count}"
        else:
            release_flag==False
        ###################################################################################################################
        label_plot = os.path.join(plot_folder, f"frame_{now.strftime('%Y-%m-%d')}.jpg")
        file_path_data_csv = os.path.join(data_folder, f"frame_{now.strftime('%Y-%m-%d')}.csv")
        
        image_sequence_label=os.path.join(full_path_img, f"frame_{current_time.strftime('%Y-%m-%d_%H-%M-%S')}_{frame_count}.jpg")
        stats_sequence_label=os.path.join(base_directory, f"frame_{current_time.strftime('%Y-%m-%d_%H')}.jpg")
        #stats_sequence_label=os.path.join(full_path_stats, f"frame_{current_time.strftime('%Y-%m-%d_%H')}.jpg")
        stats_sequence_label_H=os.path.join(full_path_stats, f"frame_{current_time.strftime('%Y-%m-%d_%H')}_{frame_count}.jpg")
        #find_and_clean_stats_folders(base_directory)
        
        # Check if the current directory has "mask.png"
        if mask_path.exists():
            frame = cv2.bitwise_and(frame, frame, mask=masked_poly)
            ref_mask=masked_poly.astype(bool)
       
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        #######Homography###############
        # Feature matching
        #keypoints, descriptors = orb.detectAndCompute(frame_gray, None)
        keypoints, descriptors = sift.detectAndCompute(frame_gray, None)
        #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Create a BFMatcher object with distance measurement cv2.NORM_L2 (appropriate for SIFT) and crossCheck=True for better matching
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(prev_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:  # Ensure we have enough matches to find a homography
            src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = frame_gray.shape
                frame_gray = cv2.warpPerspective(frame_gray, M, (w, h))
                
                # Calculate and display the shift in x and y directions
                dx, dy = M[0, 2], M[1, 2]
                #print(f"Shift in x: {dx:.2f}, Shift in y: {dy:.2f}")
        
         # Update keypoints and descriptors for the next frame
        prev_keypoints, prev_descriptors = keypoints, descriptors
        #######################
        frame_gray_ini=frame_gray
        prev_frame_initgray=prev_frame
         # Apply background subtraction
        np.seterr(divide='ignore', invalid='ignore')
    
        frame_gray=frame_gray.astype(np.uint8)
        
        # Apply MOG2 to get the foreground mask
        fgMask1 = backSub1.apply(frame_gray)
        
        fgMask1[fgMask1<1]=0
        #  # Initialize avg_frame if it has not been initialized
        if avg_frame is None:
            avg_frame = np.float32(frame_gray)

        # Calculate the weighted average
        cv2.accumulateWeighted(frame_gray, avg_frame, alpha)
        avg_gray = cv2.convertScaleAbs(avg_frame)

        # Thresholding the difference between the current frame and the running average
        _, avg_mask = cv2.threshold(cv2.absdiff(frame_gray, avg_gray), 30, 255, cv2.THRESH_BINARY)

        # Combine MOG2 mask and weighted average mask to refine the foreground mask
        combined_mask = cv2.bitwise_and(fgMask1, avg_mask)
        #combined_mask=fgMask1
        if mask_path.exists():
            combined_mask=cv2.bitwise_and(fgMask1, masked_poly)
        
        # prev_frame[combined_mask==0]=0
        # frame_gray[combined_mask==0]=0
        
        #  # pixel precision first
        # detected_shift, error, diffphase = phase_cross_correlation(prev_frame2, frame_gray2, upsample_factor = 20, normalization='phase' )
        # y_shift, x_shift = detected_shift

        shift_list_x.append(dx)
        shift_list_y.append(dy)
        
         # Start timer
        start_time = time.time()
        
        if mask_path.exists():
            poly_aoi = find_polygons(masked_poly)
            contours, _ = cv2.findContours(masked_poly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Calculate the area for each contour and find the largest
            largest_contour = max(contours, key=cv2.contourArea)
            # Find the bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            frame_gray_crop=frame_gray
            # Crop the image using array slicing
        
            frame_gray_crop=mask_outside_contour(largest_contour, frame_gray_crop)
            prev_frame_crop=mask_outside_contour(largest_contour, prev_frame)
            frame_gray_ini_crop=mask_outside_contour(largest_contour, frame_gray_ini)
            frame_gray_crop_noNan = np.where(np.isnan(frame_gray_crop), 0, frame_gray_crop)
            prev_frame_crop_noNan = np.where(np.isnan(prev_frame_crop), 0, prev_frame_crop)
            float_frame_gray_crop=frame_gray_crop.astype('float32')
            mask_nan=float_frame_gray_crop[float_frame_gray_crop==0]=np.nan
            
        else:
            ###Circle window
            min_val=0
            max_val=150
            min_loc=(0, 0)
            max_loc=(600, 400)
            
            # Define the cropping rectangle
            radius = int(max_val)
            top_left = (max(max_loc[0] - radius, 0), max(max_loc[1] - radius, 0))
            bottom_right = (min(max_loc[0] + radius, frame_gray.shape[1] - 1), min(max_loc[1] + radius, frame_gray.shape[0] - 1))
            
            frame_gray_crop=frame_gray[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            prev_frame_crop=prev_frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            frame_gray_ini_crop=frame_gray_ini[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            frame_gray_crop_noNan = frame_gray_crop
            prev_frame_crop_noNan = prev_frame_crop
            
            prev_frame_crop_noNan=prev_frame
            frame_gray_crop_noNan=frame_gray
            float_frame_gray_crop=frame_gray_crop.astype('float32')
            mask_nan=float_frame_gray_crop[float_frame_gray_crop==0]=np.nan

        ############################
        
        
        
        
        ####################
        
        
        #frame_gray = shift_cor(frame_gray, shift=detected_shift)
        
        similarity_index, ssim_map = ssim(prev_frame.astype('uint8'), frame_gray.astype('uint8'), full=True)
        
        fgMask = backSub.apply(frame_gray_crop_noNan)
        # Calculate dense optical flow
        # Apply background subtraction
    
        prev_frame_crop_noNan[fgMask==0]=0
        frame_gray_crop_noNan[fgMask==0]=0
        
        flow = cv2.calcOpticalFlowFarneback(prev_frame, frame_gray, None, pyr_scale=0.5, levels=15, winsize=148,iterations= 3, poly_n=5,poly_sigma= 1.5, flags=1)
        
       
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        magnitude[combined_mask==0]=0
        
        #magnitude[mask_nan==np.nan]=np.nan
        ##########
         # Set the magnitude to zero where fgMask is zero
        
        # if np.max(magnitude) < 0.02:
        #     alpha = 0.1  # Low motion; slower update
        # elif np.max(magnitude) < 0.05:
        #     alpha = 0.2  # Moderate motion; moderate update
        # else:
        #     alpha = 0.3  # High motion; faster update
                
        
        # fgMask[fgMask<1]=0
        # magnitude[fgMask == 0] = 0
        
        #magnitude=replace_outliers_with_nan_fixed(magnitude, z_threshold=5)
        
        # outliers = detect_outliers(magnitude)
        # magnitude = simple_kriging(magnitude, outliers)
       
        # Apply the mask to the magnitude image
        #magnitude = cv2.multiply(magnitude, fgMask1.astype('float32'))
        
        
        # # Get the current datetime
        current_datetime = datetime.now()
        if ssim_threshold is not None:
            magnitude[ssim_map > ssim_threshold]=0

        
        
        ##########
        
        
        
        if mask_path.exists():
            # Find and draw polygons around pixels of value 1
            poly_aoi = find_polygons(masked_poly)
            window=magnitude.copy()
            contours, _ = cv2.findContours(masked_poly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Calculate the area for each contour and find the largest
            largest_contour = max(contours, key=cv2.contourArea)
            # Find the bounding rectangle of the largest contour
            x, y, w, h = cv2.boundingRect(largest_contour)
        
            frame_gray_crop=frame_gray
            #Crop the image using array slicing
            
            frame_gray_crop=mask_outside_contour(largest_contour, frame_gray_crop)
            
            window=mask_outside_contour(largest_contour, window)
            vel_2d_crop=window
            
            # frame_gray_crop = frame_gray[y:y+h, x:x+w]
            # vel_2d_crop=vel_2d_crop[y:y+h, x:x+w]
        else:
            vel_2d_crop=magnitude[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            
            # Crop the mask
            window = vel_2d_crop   

       
        # if ssim_threshold is not None:
           
        #     magnitude[ssim_map > ssim_threshold]=0
        #     angle[ssim_map > ssim_threshold]=0
            # else:
                
            #     magnitude[ssim_map < ssim_threshold]=0
            #     angle[ssim_map < ssim_threshold]=0
        
        fgMask[fgMask<1]=0
        window[fgMask == 0] = 0
        window=window*0.8 ### one pixel equal to 80cm
        frame_numbers.append(frame_count)
        end_time = time.time()
        processing_time = (end_time - start_time)
        window=window/float(processing_time)
          # Calculate the mean pixel value of the window
        mean_pixel_value = np.nanmean(window)
        disp_max = np.nanmax(window)
        disp_min = np.nanmin(window)
        std_window=np.nanstd(window)
        # Append the datetime and displacements to the respective lists
        frame_datetimes.append(current_datetime)
        min_displacements.append(disp_min)
        max_displacements.append(disp_max)
        mean_displacement.append(mean_pixel_value)
        std_frames.append(std_window)
        
        processing_times.append(processing_time)
        
        
        frame_count += 1
        
        plot_timedelta=frame_datetimes[0] - frame_datetimes[-1]
        
        # Calculating total difference in minutes
        total_minutes = plot_timedelta.days * 24 * 60 + plot_timedelta.seconds / 60
        
        # Display x and y displacement as text on the frame
        cv2.putText(frame_init, f'Displacement-VEL MIN , MAX: {disp_min:.2f}, {disp_max:.2f}, processing_time:{processing_time:.2f} seconds/frame at Frame Number:{frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(frame, f'Y Displacement MIN , MAX: {y_disp_min:.2f} , {y_disp:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Normalize the magnitudes to the range [0, 1]
        normalized_magnitudes = cv2.normalize(magnitude, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Convert normalized magnitudes to colors using a colormap
        colormap = plt.get_cmap('hot')  # You can use other colormaps like 'viridis', 'plasma', etc.
        colors = (colormap(normalized_magnitudes)[:, :, :3] * 255).astype(np.uint8)
        ########################################################################3#
       
        plt.style.use('seaborn-notebook')
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

        # Display the grayscale image
        ax.imshow(frame_gray_ini_crop, cmap='gray')  # Specifying cmap='gray' for clarity

        # Display the velocity data with a professional color map
        dd = ax.imshow(vel_2d_crop, cmap='plasma_r', alpha=0.6, vmin=0, vmax=1)

        # Adding a color bar with a label, adjusting its position relative to the new layout
        fig.colorbar(dd, ax=ax, fraction=0.046, pad=0.04).set_label(f'Displacement (m)/{3}seconds', rotation=270, labelpad=15, fontsize=12, fontweight='bold')

        # Adding titles and labels with a professional appearance
        ax.set_title('Displacement AOI', fontsize=16, fontweight='bold', pad=20)

        # Enhancing the aesthetics of the tick labels
        ax.tick_params(axis='both', which='major', labelsize=12)

       
        if mask_path.exists():
            # Add a small margin for better visualization
            margin = 10
            ax.set_xlim(x , x+w )
            ax.set_ylim( y+h, y  )
        
        # Adjust layout for a cleaner look
        plt.tight_layout()  # Adjust 'pad' as needed to fit the color bar and prevent overlap

       
        
        
        try:  
            # Save with high quality
            fig.savefig(stats_sequence_label[:-4]+'_w.jpg', dpi=150) 
        
        except PermissionError as e:
            print(f"Permission denied when trying to save 'plot.png': {e}")
            # Handle the error, e.g., by using a different file name or directory
        except Exception as e:
            print(f"An unexpected error occurred when trying to save 'plot.png': {e}")
        
        try:   
            plt_img=cv2.imread(stats_sequence_label[:-4]+'_w.jpg', 1)
            
        except PermissionError as e:
            print(f"Permission denied when trying to save 'plot.png': {e}")
            # Handle the error, e.g., by using a different file name or directory
        except Exception as e:
            print(f"An unexpected error occurred when trying to save 'plot.png': {e}")
        
         
        if plt_img is not None: 
            try:
                
                # Attempt to resize the colorbar to match the image height
                    AOI_img = cv2.resize(plt_img, (frame_width  , frame_height), interpolation=cv2.INTER_AREA)
                    
                # Update the last successful colorbar
                    if AOI_img is None:
                        AOI_img=last_successful_colorbar
                    combined_frame=np.hstack((frame_init, AOI_img))
            except Exception as e:
                print("Resize operation failed:", str(e))
                # if last_successful_colorbar  is not None:
                #     AOI_img = last_successful_colorbar 
                # else:
                #     print("No successful previous resize operation to fallback on.")

            ##########################
        if AOI_img is not None:
            last_successful_colorbar  = AOI_img
        
        

        #Convert frame datetimes to matplotlib-compatible format
        datetimes_from_frame = matplotlib.dates.date2num(frame_datetimes)

       # Create a figure and two subplots for min and max displacements
        fig2, (ax2, ax1) = plt.subplots(2, 1, sharex=False, figsize=(15, 10), constrained_layout=False)
        
        ##################
        from scipy import stats
       
        X = frame_numbers

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(datetimes_from_frame, mean_displacement)

        # Generate Y values for the trendline
        Y_trendline = intercept + slope * datetimes_from_frame
        
        
        total_time=np.cumsum(processing_times)
        total_time=total_time[-1]
        ax1.plot(datetimes_from_frame, Y_trendline, label=f'(Mean-Displacement Rate: {slope:.4f}) m/{total_minutes:.4f}minutes', color='red', linestyle='-')
        
        # Calculate the mean square error (MSE) of the residuals
        residuals = mean_displacement - (slope * datetimes_from_frame + intercept)
        MSE = np.mean(residuals**2)

        # Calculate the standard deviation of the y-values
        std_dev = np.sqrt(MSE)

        # Calculate the confidence intervals around the trendline
        # Assuming a 95% confidence level, the t-value can be found using the t-distribution
        # Adjust degrees of freedom
        df = len(datetimes_from_frame) - 2
        # 95% confidence level
        confidence_level = 0.95
        t_value = stats.t.ppf((1 + confidence_level) / 2, df)

        # Standard error of the slope
        se_slope = std_err

        # Calculate the margin of error
        margin_of_error = t_value * se_slope * np.sqrt(1/len(datetimes_from_frame) + (datetimes_from_frame - np.mean(datetimes_from_frame))**2 / np.sum((datetimes_from_frame - np.mean(datetimes_from_frame))**2))

        # Upper and lower bounds of the confidence interval
        upper_bound = Y_trendline + margin_of_error
        lower_bound = Y_trendline - margin_of_error
       
        
        # Plot confidence interval
        ax1.fill_between(datetimes_from_frame, lower_bound, upper_bound, color='k', alpha=0.2, label='95% CI for Trendline')
        ###########
         # Plot maximum displacement with a red line
        ax1.plot(datetimes_from_frame, mean_displacement, label='Mean Displacement', color='k', linestyle='-', marker='o')
        #ax1.plot(datetimes_from_frame, std_frames, label=f'Standard-Deviation: {std_err:.4f}', color='green', linestyle='--', marker='x')
        # Plot minimum displacement with a blue line
        ax1.plot(datetimes_from_frame, min_displacements, label='Min Displacement', color='blue', linestyle='-', marker='^')

        # Plot maximum displacement with a red line
        #ax1.plot(datetimes_from_frame, max_displacements, label='Max Displacement', color='red', linestyle='--', marker='s')
        
       

        # Set labels for both y-axes
        ax1.set_ylabel('Displacement(m)', fontsize=12, fontweight='bold', color='darkred')
        ax1.set_xlabel('Time', fontsize=12, fontweight='bold', color='darkred')

        # Add a title
        ax1.set_title(f'Mean Displacement Over Time in m, timestamp:{timestamp}', fontsize=14, fontweight='bold', color='navy')

        # Add a legend in the top right corner
        #ax1.legend(loc='lower right', fontsize=12, framealpha=0.6)

        # Format the x-axis as datetime
        # Format the x-axis as datetime
        ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%d:%m:%y-%H:%M:%S'))
         # Adding minor ticks
        ax1.minorticks_on()
        # Rotate the x-axis labels for better readability
        #ax1.set_xticks(rotation=45)
        plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

        # Add grid lines
        #ax1.grid(True, linestyle='--', alpha=0.7)
        
        
        #####
        # Create a second y-axis that shares the same x-axis
        ax3 = ax1.twinx()
        colorx = 'tab:blue'
        ax3.plot(datetimes_from_frame, shift_list_x, color=colorx, label='Image-Shift-x(1px=80cm)', linestyle='dotted', marker='d')
        ax3.tick_params(axis='y', labelcolor=colorx)
        ax3.set_ylabel('Image-Shift', fontsize=12, fontweight='bold', color=colorx)

        colory = 'tab:purple'
        ax3.plot(datetimes_from_frame, shift_list_y, color=colory, label='Image-Shift-y(1px=80cm)', linestyle='dashed', marker='^')
        ax3.tick_params(axis='y', labelcolor=colory)
        ax3.set_ylabel('Image-Shift(pixel)', fontsize=12, fontweight='bold', color=colorx)
        
        #ax3.legend(loc='upper right')
        
        # Retrieving handles and labels for both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles3, labels3 = ax3.get_legend_handles_labels()

        # Combining handles and labels
        handles = handles1 + handles3
        labels = labels1 + labels3

        # Creating a single legend
        ax1.legend(handles, labels, loc='upper right', fontsize=12, framealpha=0.85)
        
        ###ax2##########
        cumulative_processing_times=np.cumsum(processing_times)
        # Adjusted plotting for latency
        target_processing_time = 0.0001  # Approximate real-time target
        latency_times = np.array(processing_times) - target_processing_time
        
        ax2.plot(frame_numbers, latency_times, label="Frame Processing Latency", color='tomato', linewidth=2)
        ax2.axhline(0, color='limegreen', linestyle='--', label='Real-time Threshold')
        
        # Annotation for total processing time
        total_proc_time = cumulative_processing_times[-1]
       
        
        ax2.set_title(f'Frame Processing Latency vs. Frame Number: Total Processing Time: {total_proc_time:.2f} seconds', fontsize=14, fontweight='bold', color='navy')
        ax2.set_xlabel('Frame Number', fontsize=12, fontweight='bold', color='darkred')
        ax2.set_ylabel('Latency Time (seconds)', fontsize=12, fontweight='bold', color='darkred')

        # Marking areas to indicate performance
        ax2.fill_between(frame_numbers, latency_times, 0, where=(latency_times > 0), color='red', alpha=0.3, label='Above Real-time Threshold')
        ax2.fill_between(frame_numbers, latency_times, 0, where=(latency_times <= 0), color='green', alpha=0.3, label='Within Real-time Threshold')

        # Adding a legend
        ax2.legend(loc="lower right", fontsize=12)

        # Styling the ticks
        ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, color='green', direction='in')
        ax2.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, color='green', direction='in')

        # Adding minor ticks
        ax2.minorticks_on()
        #ax2.grid(True)
        # Styling the plot background
        plt.style.use('seaborn-notebook')
        
        #########
        
       #######################
        
        ### Save data to csv files
        DD_list=[dd.strftime("%Y-%m-%d") for dd in frame_datetimes ]
        TT_list=[tt.strftime("%H:%M:%S") for tt in frame_datetimes ]
        
        column_names = ['Date', "Time", "frame_numbers","std_frames", "min_displacements", "max_displacements", "mean_displacement", "shift_list_x", "shift_list_y", "latency_times"]

        # Creating a DataFrame
        df = pd.DataFrame(list(zip(DD_list,TT_list, frame_numbers,std_frames, min_displacements, max_displacements, mean_displacement, shift_list_x, shift_list_y, latency_times)), columns=column_names)

        # Saving the DataFrame to a CSV file
        
        df.to_csv(file_path_data_csv)
        ########################3
       
        # Adjust the plot layout
        #plt.tight_layout(h_pad=0.1, w_pad=0.1)
        #fig2.canvas.draw()
    
        # if release_flag:
        #     import re            # Remove the 'stats' folder from the path using re.sub
        #     stats_sequence_label = re.sub(r'\\stats', '', stats_sequence_label)
        try:
          
            # Example: Attempting to save a plot
            #fig2.savefig(label_plot[:-4]+f"_{last_hour}.jpg", dpi=150)
            fig2.savefig(stats_sequence_label, dpi=72 )
        except PermissionError as e:
            print(f"Permission denied when trying to save 'plot.png': {e}")
            # Handle the error, e.g., by using a different file name or directory
        except Exception as e:
            print(f"An unexpected error occurred when trying to save 'plot.png': {e}")


        try:
            # Example: Attempting to save a plot
            image = cv2.imread(stats_sequence_label)
            if image is not None:
                minmax_plot = cv2.resize(image, (frame_width, frame_height))
                #minmax_plot = image
                last_successful_image=minmax_plot
            else:
                minmax_plot=last_successful_image   
        except PermissionError as e:
            print(f"Permission denied when trying to save 'plot.png': {e}")
            # Handle the error, e.g., by using a different file name or directory
        except Exception as e:
            print(f"An unexpected error occurred when trying to save 'plot.png': {e}")
        
        #  # Resize the colorbar to match the image height
        
        
        
        plt.close('all')
        
        #####################################
       
                
        # Display motion vectors as arrows
        step = 20  # Arrow spacing
        for y in range(0, frame_init.shape[0], step):
            for x in range(0, frame_init.shape[1], step):
                fx, fy = flow[y, x]
                # Calculate the magnitude of the flow vector
                flow_magnitude = np.sqrt(fx**2 + fy**2)

                # Calculate the end point of the arrow based on flow magnitude
                end_x = int(x + fx) 
                end_y = int(y + fy)

                # Scale the line length based on the flow magnitude
                scaled_end_x = int(x + fx * flow_magnitude)
                scaled_end_y = int(y + fy * flow_magnitude)
                color = tuple(map(int, colors[y, x]))
                # Draw the arrowed line
                cv2.arrowedLine(frame_init, (x, y), (scaled_end_x, scaled_end_y), color, 1)
        
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                
        
        
        # Combine the image with the colorbar
        # if plt_img is not None:
        #     combined_frame=np.hstack((frame_init, AOI_img))
            #combined_frame=np.hstack((combined_frame, colored_magnitude))
            #combined_mag = np.hstack((colored_magnitude, AOI_img))
            #combined_mag1 = np.hstack((combined_mag, minmax_plot))
        
        # Draw the window on the frame
        if mask_path.exists():
            #draw_circles_on_image(poly_aoi, combined_frame)
            #draw_circles_on_image(poly_aoi, combined_mag1)
            # The function cv2.findContours is used here with the arguments to find external contours and approximate them.
            contours, _ = cv2.findContours(masked_poly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(combined_frame, contours, -1, (0, 0, 255), 5)  # Draw white contours with thickness of 1
           

        else:  
            # cv2.rectangle(combined_frame, (window_x, window_y), (window_x+window_size[0], window_y+window_size[1]), (0, 0, 255), 5, cv2.LINE_AA)
            # # Draw the window on the frame
            # cv2.rectangle(combined_mag1, (window_x, window_y), (window_x+window_size[0], window_y+window_size[1]), (0, 0, 255), 5, cv2.LINE_AA)
            #cv2.rectangle(combined_frame, (y,y+h), (x,x+w), (0, 0, 255), 5, cv2.LINE_AA)
            #cv2.rectangle(combined_mag1, top_left, bottom_right, (0, 0, 255), 5, cv2.LINE_AA)
            cv2.rectangle(combined_frame, top_left, bottom_right, (0, 0, 255), 5, cv2.LINE_AA)
        
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        text_size = cv2.getTextSize(current_datetime.strftime("%Y-%m-%d %H:%M:%S"), font, font_scale, font_thickness)[0]
        
        # Position at the bottom left
        text_x = 10  # 10 pixels from the left edge for a little margin
        text_y = frame.shape[0] - 10  # 10 pixels from the bottom edge

        # Add the timestamp to the frame
        cv2.putText(combined_frame, current_datetime.strftime("%Y-%m-%d %H:%M:%S"), (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)     
        #combined_frame = np.hstack((combined_frame, colorbar))
        os.makedirs('plots/screenshots', exist_ok=True)
        os.makedirs('plots/Videos', exist_ok=True)
        os.makedirs('plots/Videos/converted', exist_ok=True)
        os.makedirs('plots/Videos/temp', exist_ok=True)
        
        save_image='plots/screenshots/'+ label_plot[5:-4]+f'_{last_hour}.jpg'
        #save_gif_statsprofile='plots/Videos/'+ label_plot[5:-4]+'_profile.gif'
        save_gif_image='plots/Videos/'+label_plot[5:-4]+ f'_{last_hour}_image.mp4'
        #save_gif_converted='plots/Videos/converted/'+os.path.splitext(os.path.basename(image_sequence_label))[0]+'_image.mp4'
        
        
        # Calculate the new dimensions
        width_f = int(combined_frame.shape[1] *1)
        height_f = int(combined_frame.shape[0] * 1)
        new_dimensions = (1280, 400)

      
        imagescreenshot_list.append(combined_frame)
        
        if save_output:
        
            # # Create a temporary file and close it immediately to use its name
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4',  dir='plots/Videos/temp') as temp_file:
                temp_file_name = temp_file.name
                
            fps = 30  # or any other frame rate you want
            frame_size = (imagescreenshot_list[0].shape[1], imagescreenshot_list[0].shape[0])  # width, height from the first image
            out_v = cv2.VideoWriter(temp_file_name, fourcc, fps, frame_size)

            # temp_count=0 
            # if release_flag== True : 
                
            for img in imagescreenshot_list:
                # Assuming the images are in BGR format, if not, convert them
                out_v.write(img)
                
            out_v.release()   
            # Release everything when done
            
            
            os.replace(temp_file_name, save_gif_image)
           
            
            #if current_hour!=last_hour:
               # delete_older_mp4('plots/Videos', 1500)
                
                #shutil.move(temp_file_name, save_gif_image)
                
            # if not ret:  
            #     for img in imagescreenshot_list:
            #         # Assuming the images are in BGR format, if not, convert them
            #         out_v.write(img)
            #     out_v.release()
                
            #     os.replace(temp_file_name, save_gif_image)
                
               
                #shutil.move(temp_file_name, image_sequence_label[:-4]+'.mp4')

            
                 
               # delete_older_mp4('plots/Videos', 1500)
            
           
            
        ######
       
       
       # Path to the custom directory
        custom_dir = 'plots/Videos/temp'

        try:
            # List all files in the directory
            files = [os.path.join(custom_dir, f) for f in os.listdir(custom_dir) if os.path.isfile(os.path.join(custom_dir, f))]
            
            # Sort the files by modification time, most recent last
            files.sort(key=lambda x: os.path.getmtime(x))
            
            # Remove the last file from the list to exclude it from deletion
            files_to_delete = files[:-5]
            
            for file in files_to_delete:
                try:
                    os.remove(file)
                except OSError as e:
                    # Check if the error is WinError 32
                    if e.winerror == 32:
                        continue
                    else:
                        raise  # Re-raise the exception if it's not related to the file being in use
        except Exception as e:
            print(f"Error processing directory cleanup: {e}")
            
       
        cv2.imwrite(save_image,combined_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80] )
       
        
        ######################
        
        # # Display frames and motion information
        # if show_video==True:
        #     cv2.imshow('Frames', combined_frame)
        #     vid=cv2.imshow('Optical Flow', combined_mag1)
        
        
       
        
        # # Check if saving output is enabled
        # if save_output:
        #     out.write(combined_mag1)
        #     out1.write(combined_frame)
            
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
        
        
        yield [minmax_plot , combined_frame]
         
        
    
    
    cap.release()
    cv2.destroyAllWindows()
    
    
    del fig
    del fig2
     
############################################

############################## The below program works to save the processing video into video file#############

# Function to measure displacement using Dense Optical Flow
def measure_displacement_from_camera_toFile(hls_url, alpha=0.1, save_output=False, output_filename=None, ssim_threshold=0.4, 
                                     pyr_scale=0.5, levels=15, winsize=32,iterations= 3, poly_n=5,poly_sigma= 1.5, flags=1 , show_video=True):
    
    '''
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
    
    '''
    
    # Set the new dimensions for resizing
    new_width = 1000
    new_height = 600
    
    cap = cv2.VideoCapture(hls_url)
    
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return
    
    # Check if saving output is enabled
    if save_output:
        # Get the frames' width, height, and frame rate
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_rate = int(cap.get(5))
        
        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_filename, fourcc, frame_rate, (frame_width + frame_width//10, frame_height))
        out1 = cv2.VideoWriter(output_filename[:-4]+"_1.mp4", fourcc, frame_rate, (frame_width +frame_width//10, frame_height))
    
    ret, prev_frame = cap.read()
    
    # Resize the frame to the new dimensions
    #prev_frame = cv2.resize(prev_frame, (new_width, new_height))
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    total_displacement = 0.0
    
    # Initialize the background model as the first frame
    background_model = prev_frame.astype(np.float32)
    
    x_displacement = []
    y_displacement = []
    
    frame_count = 0
    
    prev_displacement = None
    
    # Lists to store processing times
    processing_times = []
    frame_numbers = []

   
    
    while True:
        
        ret, frame = cap.read()
        # Resize the frame to the new dimensions
        #frame = cv2.resize(frame, (new_width, new_height))
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
         # Start timer
        start_time = time.time()
        
        similarity_index, ssim_map = ssim(prev_frame, frame_gray, full=True)
        
        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_frame, frame_gray, None, pyr_scale=0.5, levels=15, winsize=120,iterations= 3, poly_n=5,poly_sigma= 1.5, flags=1)
        
        # Calculate displacement magnitude
        displacement = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        
        # Adjust ksize and sigmaX based on the level of noise in your image
        #displacement = cv2.GaussianBlur(displacement, (5, 5), 0)
        
        
        
        ##########
        
        # End timer and calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        frame_numbers.append(frame_count)
        
        prev_frame = frame_gray
        
        frame_count += 1
        
        # Create an HSV image for visualizing the flow
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        
        # Calculate angle and magnitude for flow visualization
        #angle = np.arctan2(flow[..., 1], flow[..., 0])
        #magnitude = cv2.normalize(displacement, None, 0, 255, cv2.NORM_MINMAX)
        #magnitude=displacement
        
        #magnitude, angle = cv2.cartToPolar(cv2.medianBlur(flow[..., 0], 3), cv2.medianBlur(flow[..., 1], 3))
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate x and y displacement components
        x_disp = np.max(flow[..., 0])
        y_disp = np.max(flow[..., 1])
        
        x_disp_min = np.min(flow[..., 0])
        y_disp_min = np.min(flow[..., 1])
        
        disp_min=np.min(magnitude)
        disp_max=np.max(magnitude)
        
        # Append the displacement components to the lists
        x_displacement.append(x_disp)
        y_displacement.append(y_disp)
        
        # Sum up the displacement in the entire frame
        total_displacement += np.sum(displacement)
        if ssim_threshold is not None:
           
            magnitude[ssim_map > ssim_threshold]=0
            angle[ssim_map > ssim_threshold]=0
            # else:
                
            #     magnitude[ssim_map < ssim_threshold]=0
            #     angle[ssim_map < ssim_threshold]=0
        
        
        # Set hue according to the direction of motion
        hsv[..., 0] = (angle * 180 / np.pi / 2).astype(np.uint8)
        
        # Set value (brightness) according to the magnitude of motion
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV to BGR for display
        flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Apply Gaussian blur to reduce noise
        flow_vis = cv2.GaussianBlur(flow_vis, (0, 0), 3)
        
        if alpha is not None:
            # Update the background model using running average
            cv2.accumulateWeighted(frame_gray, background_model, alpha)
            background_mask = cv2.convertScaleAbs(background_model)
            
            # Invert the background mask to create a foreground mask
            foreground_mask = cv2.absdiff(frame_gray, background_mask)
            
            # Threshold the foreground mask to create a binary mask
            _, binary_mask = cv2.threshold(foreground_mask, 30, 255, cv2.THRESH_BINARY)
            
            # Use the binary_mask to refine the flow visualization
            #
            
            flow_vis = cv2.bitwise_and(flow_vis, flow_vis, mask=binary_mask)
        
        # Display x and y displacement as text on the frame
        cv2.putText(flow_vis, f'Displacement-VEL MIN , MAX: {disp_min:.2f}, {disp_max:.2f}, processing_time:{processing_time:.2f} seconds/frame at Frame Number:{frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(flow_vis, f'Y Displacement MIN , MAX: {y_disp_min:.2f} , {y_disp:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display x and y displacement as text on the frame
        cv2.putText(frame, f'Displacement-VEL MIN , MAX: {disp_min:.2f}, {disp_max:.2f}, processing_time:{processing_time:.2f} seconds/frame at Frame Number:{frame_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        #cv2.putText(frame, f'Y Displacement MIN , MAX: {y_disp_min:.2f} , {y_disp:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Normalize the magnitudes to the range [0, 1]
        normalized_magnitudes = cv2.normalize(magnitude, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Convert normalized magnitudes to colors using a colormap
        colormap = plt.get_cmap('hot')  # You can use other colormaps like 'viridis', 'plasma', etc.
        colors = (colormap(normalized_magnitudes)[:, :, :3] * 255).astype(np.uint8)
        ########################################################################3#
        # Create a colorbar using matplotlib
        fig, ax = plt.subplots(figsize=(0.5, 4))
        fig.subplots_adjust(bottom=0.5, top=0.99, left=0.01, right=0.3)

        # Use the same colormap and normalization range as the image
        norm = plt.Normalize(vmin=magnitude.min(), vmax=magnitude.max())
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('hot')), cax=ax)
        # Add label and tick marks to the colorbar
        #cb.set_label('Displacement', labelpad=-32, y=0.5, rotation=-90)
        #ticks = [magnitude.min(), magnitude.max()/2, magnitude.max()/4, magnitude.max()/6, magnitude.max()]
        #cb.set_ticks(ticks)
        #plt.axis('off')

        # Convert plt figure to numpy array and then to a cv2 image
        fig.canvas.draw()
        plt_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        plt_img = plt_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt_img = cv2.cvtColor(plt_img, cv2.COLOR_RGB2BGR)

        # Resize the colorbar to match the image height
        colorbar = cv2.resize(plt_img, (magnitude.shape[1]//10, magnitude.shape[0]))

        #####################################
        # Display motion vectors as arrows
        step = 20  # Arrow spacing
        for y in range(0, frame.shape[0], step):
            for x in range(0, frame.shape[1], step):
                fx, fy = flow[y, x]
                # Calculate the magnitude of the flow vector
                flow_magnitude = np.sqrt(fx**2 + fy**2)

                # Calculate the end point of the arrow based on flow magnitude
                end_x = int(x + fx)
                end_y = int(y + fy)

                # Scale the line length based on the flow magnitude
                scaled_end_x = int(x + fx * flow_magnitude)
                scaled_end_y = int(y + fy * flow_magnitude)
                
                color = tuple(map(int, colors[y, x]))

                # Draw the arrowed line
                cv2.arrowedLine(flow_vis, (x, y), (scaled_end_x, scaled_end_y), color, 1)
                
        # Display motion vectors as arrows
        step = 20  # Arrow spacing
        for y in range(0, frame.shape[0], step):
            for x in range(0, frame.shape[1], step):
                fx, fy = flow[y, x]
                # Calculate the magnitude of the flow vector
                flow_magnitude = np.sqrt(fx**2 + fy**2)

                # Calculate the end point of the arrow based on flow magnitude
                end_x = int(x + fx) 
                end_y = int(y + fy)

                # Scale the line length based on the flow magnitude
                scaled_end_x = int(x + fx * flow_magnitude)
                scaled_end_y = int(y + fy * flow_magnitude)
                color = tuple(map(int, colors[y, x]))
                # Draw the arrowed line
                cv2.arrowedLine(frame, (x, y), (scaled_end_x, scaled_end_y), color, 1)
        
        magnitude = cv2.normalize(displacement, None, 0, 255, cv2.NORM_MINMAX)
        # Convert the float data to uint8
        gray_image = magnitude.astype('uint8')
        
        colored_magnitude = cv2.applyColorMap(flow_vis[..., 2], cv2.COLORMAP_HOT)
        
        # Combine the image with the colorbar
        combined_mag = np.hstack((colored_magnitude, colorbar))
        combined_frame = np.hstack((frame, colorbar))
        
        # Display frames and motion information
        if show_video:
            cv2.imshow('Frames', combined_frame)
            vid=cv2.imshow('Optical Flow', combined_mag)
        
        plt.close()
        
        # Check if saving output is enabled
        if save_output:
            out.write(combined_mag)
            out1.write(combined_frame)
            
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    
    # Release the VideoWriter object if it was created
    if save_output:
        out.release()
    
    # # Apply the mask to both x_displacement and y_displacement
    # masked_x_displacement = [x_disp * binary_mask[frame_index // 2] for frame_index, x_disp in enumerate(x_displacement)]
    # masked_y_displacement = [y_disp * binary_mask[frame_index // 2] for frame_index, y_disp in enumerate(y_displacement)]
    
    cumulative_processing_times=np.cumsum(processing_times)
    # Adjusted plotting for latency
    target_processing_time = 0.033  # Approximate real-time target
    latency_times = np.array(processing_times) - target_processing_time
    
    plt.figure(figsize=(12, 8))
    plt.plot(frame_numbers, latency_times, label="Frame Processing Latency", color='tomato', linewidth=2)
    plt.axhline(0, color='limegreen', linestyle='--', label='Real-time Threshold')
    
    # Annotation for total processing time
    total_proc_time = cumulative_processing_times[-1]
    # total_proc_time=f'Total Processing Time: {total_proc_time:.2f} s'
    # plt.annotate(f'Total Processing Time: {total_proc_time:.2f} s',
    #             xy=(1, total_proc_time), xycoords=('axes fraction', 'data'),
    #             xytext=(-10, 10), textcoords='offset points',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    #             fontsize=12, color='darkgreen')
    
    plt.xlabel('Frame Number', fontsize=14)
    plt.ylabel('Latency Time (seconds)', fontsize=14)
    plt.title(f'Frame Processing Latency vs. Frame Number: Total Processing Time: {total_proc_time:.2f} seconds', fontsize=16, fontweight='bold', color='navy')
    
    plt.xlabel('Frame Number', fontsize=14, fontweight='bold', color='darkred')
    plt.ylabel('Latency Time (seconds)', fontsize=14, fontweight='bold', color='darkred')

    # Marking areas to indicate performance
    plt.fill_between(frame_numbers, latency_times, 0, where=(latency_times > 0), color='red', alpha=0.3, label='Above Real-time Threshold')
    plt.fill_between(frame_numbers, latency_times, 0, where=(latency_times <= 0), color='green', alpha=0.3, label='Within Real-time Threshold')

    # Adding a legend
    plt.legend(loc="upper right", fontsize=12)

    # Styling the ticks
    plt.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True, color='green', direction='in')
    plt.tick_params(axis='y', which='both', left=True, right=False, labelleft=True, color='green', direction='in')

    # Adding minor ticks
    plt.minorticks_on()
    plt.grid(True)
    # Styling the plot background
    plt.style.use('seaborn-darkgrid')
    if output_filename is None:
        save_fig_frames='ProcessingTime_perFrame.png'
    else:
        save_fig_frames=output_filename+".png"
    plt.savefig(save_fig_frames)
    
    return vid


##########################################


def calculate_new_geotransform(overlap_box, src_transform):
    """
    Calculate a new geotransform matrix based on the overlap of image boundaries.

    Parameters:
    ------------
    - overlap_box (shapely.geometry.Polygon): The overlapping bounding box of all valid images.
    - src_transform (tuple): The source geotransform matrix to be modified.

    Returns:
    ---------
    - tuple: A new geotransform matrix corresponding to the overlap_box.
    """
    xmin, ymin, xmax, ymax = overlap_box.bounds
    new_transform = (xmin, src_transform[1], src_transform[2], ymax, src_transform[4], src_transform[5])
    return new_transform

def crop_to_overlap(folder_path):
    """
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
    """
    image_files = sorted(os.listdir(folder_path))
    valid_extensions = ['.tif', '.jpg', '.png', '.bmp']
    image_path_list = []
    overlap_box = None

    for file in image_files:
        if os.path.splitext(file)[1].lower() in valid_extensions:
            image_path = os.path.join(folder_path, file)
            dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
            if dataset:
                geo_transform = dataset.GetGeoTransform()
                minx = geo_transform[0]
                maxy = geo_transform[3]
                maxx = minx + geo_transform[1] * dataset.RasterXSize
                miny = maxy + geo_transform[5] * dataset.RasterYSize
                image_box = box(minx, miny, maxx, maxy)

                if overlap_box is None:
                    overlap_box = image_box
                else:
                    overlap_box = overlap_box.intersection(image_box)

                image_path_list.append(image_path)
            dataset = None  # Close the file

    if not image_path_list:
        print("No valid image files found.")
        return

    for image_path in image_path_list:
        dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
        if dataset:
            geo_transform = dataset.GetGeoTransform()
            projection = dataset.GetProjection()
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(projection)

            new_transform = calculate_new_geotransform(overlap_box, geo_transform)
            x_offset = int((new_transform[0] - geo_transform[0]) / geo_transform[1])
            y_offset = int((new_transform[3] - geo_transform[3]) / geo_transform[5])
            x_size = int((overlap_box.bounds[2] - overlap_box.bounds[0]) / geo_transform[1])
            y_size = int((overlap_box.bounds[3] - overlap_box.bounds[1]) / -geo_transform[5])

            band = dataset.GetRasterBand(1)  # Adjust this if your dataset has more bands
            data = band.ReadAsArray(x_offset, y_offset, x_size, y_size)

            temp_output_path = os.path.join(folder_path, f"temp_{os.path.basename(image_path)}")
            print(f"Attempting to create temporary file: {temp_output_path}")

            driver = gdal.GetDriverByName('GTiff')
            out_dataset = driver.Create(temp_output_path, x_size, y_size, 1, band.DataType)
            out_dataset.SetGeoTransform(new_transform)
            out_dataset.SetProjection(src_srs.ExportToWkt())

            out_band = out_dataset.GetRasterBand(1)
            out_band.WriteArray(data)

            out_band.FlushCache()
            out_dataset = None  # Close the file

            dataset = None  # Ensure the dataset is closed and released
            gc.collect()

            print(f"Checking existence of temporary file: {temp_output_path}")
            time.sleep(2)  # Adjust the sleep time if needed
            
            try:
                if os.path.exists(temp_output_path):  # Check if temporary file exists
                    print(f"Temporary file {temp_output_path} found.")
                    os.chmod(image_path, 0o777)  # Change the permission to read/write for all
                    os.remove(image_path)
                    os.rename(temp_output_path, image_path)
                    print(f"File overwritten at {image_path}")
                else:
                    print(f"Temporary file {temp_output_path} not found.")
            except PermissionError as e:
                print(f"Permission error while overwriting {image_path}: {e}")
            except Exception as e:
                print(f"Unexpected error while overwriting {image_path}: {e}")

# # Test the function with your paths
# input_folder = 'geo/flowx'
# crop_to_overlap(input_folder)



def crop_point_shapefile_with_aoi(point_shapefile, aoi_shapefile, output_folder):
    
    """
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
    
    """

    
    # Read point shapefile
    points = gpd.read_file(point_shapefile)
   

    # Read AOI polygon shapefile
    aoi = gpd.read_file(aoi_shapefile)

    # Dissolve the AOI polygon to a single geometry without specifying 'by'
    dissolved_aoi = aoi.dissolve()
    
     # Ensure both GeoDataFrames have the same CRS
    dissolved_aoi = dissolved_aoi.to_crs(points.crs)

    # Use geopandas.overlay to clip points by AOI polygon
    points_in_aoi = gpd.overlay(points, dissolved_aoi, how='intersection')

    # Check if the resulting GeoDataFrame is empty
    if not points_in_aoi.empty:
        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Construct the output shapefile path
        output_filename = os.path.basename(point_shapefile).replace(".shp", "_clipped.shp")
        output_shapefile = os.path.join(output_folder, output_filename)

        # Save the clipped points to the output shapefile
        points_in_aoi.to_file(output_shapefile)

       
    else:
        print("The resulting GeoDataFrame is empty. No file will be written.")
        

####################


    

#################