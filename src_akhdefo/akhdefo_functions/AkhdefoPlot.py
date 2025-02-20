
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.transforms as mtransforms
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio
import numpy as np
from rasterio.plot import plotting_extent
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.patches as patches
import os
import rasterio
import glob
import seaborn as sb
import pandas as pd
import matplotlib.dates as mdates
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
from matplotlib.collections import LineCollection
# Import needed packages
import geopandas as gpd
import rioxarray as rxr
import earthpy as et
import earthpy.spatial as es
import rasterio as rio
import cmocean
import plotly.graph_objects as go
import plotly.express as px
import plotly.express as px_temp
import seaborn as sns  
import plotly.offline as py_offline
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from datetime import datetime
import math
from ipywidgets import interact
from ipywidgets import widgets
import plotly.io as pio
import re
from matplotlib_scalebar.scalebar import ScaleBar

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import rasterio
import rasterio.plot
import earthpy.spatial as es
import earthpy.plot as ep
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
from datetime import datetime

def create_kmz_with_overlay(image_path, raster_path, output_path, bbox=None):
    """
    Create a KMZ file with an image overlay based on the geographic bounds of a GeoTIFF file.

    This function reads a GeoTIFF raster file to determine its geographic bounds and coordinate reference system (CRS).
    It then uses these bounds to overlay an image on a KML file, which is subsequently saved as a KMZ file.
    If the CRS of the raster is not EPSG:4326 (latitude and longitude), the function will transform the coordinates
    to EPSG:4326 using pyproj.

    Parameters
    ----------
    image_path : str
        Path to the image file that will be overlaid on the KML. This image should ideally be a PNG or JPEG file.
    raster_path : str
        Path to the GeoTIFF raster file from which geographic bounds are derived.
    output_path : str
        Path where the output KMZ file will be saved.
    bbox: list
        [N,E,S,W]
    Returns
    -------
    None
        The function does not return any value but saves the KMZ file at the specified output location.

    Example
    -------
    >>> create_kmz_with_overlay('./Figure_Analysis/East_universal.png', 
                                './las_cropped_aoi/Sep_13th2017_lidar_crop.tif', 
                                'output_overlay.kmz')
    """
    
    import rasterio
    from rasterio.crs import CRS
    from pyproj import Transformer, CRS as PyProjCRS
    import simplekml
    # Open the GeoTIFF file to read its bounds and CRS
    
    with rasterio.open(raster_path) as dataset:
        bounds = dataset.bounds
        src_crs = dataset.crs  # Source CRS

        # Manually create a CRS object for EPSG:4326 if from_epsg fails
        epsg_4326_crs = PyProjCRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")

        # Check if the source CRS is not EPSG:4326
        if src_crs != epsg_4326_crs:
            # Create a transformer to convert from source CRS to EPSG:4326
            transformer = Transformer.from_crs(src_crs, epsg_4326_crs, always_xy=True)
            
            # Transform each corner of the bounding box
            west, south = transformer.transform(bounds.left, bounds.bottom)
            east, north = transformer.transform(bounds.right, bounds.top)
        else:
            # Use the bounds directly if they are already in EPSG:4326
            north = bounds.top
            south = bounds.bottom
            east = bounds.right
            west = bounds.left
    
    # import rasterio
    # import re

    
   
    
    if bbox is not None:
        # Calculate coordinates for each corner
        north, east= bbox[0], bbox[1]    
        south, west =  bbox[2], bbox[3]
        # # Check if the source CRS is not EPSG:4326
        # if src_crs != epsg_4326_crs:
        #    east, north=convert_utm_to_latlon(east, north, crs_info['UTM Zone'], crs_info['Hemisphere'] )
        #    west, south=convert_utm_to_latlon(west, south, crs_info['UTM Zone'], crs_info['Hemisphere'] )
        
    # Create a KML object
    kml = simplekml.Kml()
     


    # Create a ground overlay
    ground = kml.newgroundoverlay(name='Sample Overlay')
    ground.icon.href = image_path
    ground.latlonbox.north = north
    ground.latlonbox.south = south
    ground.latlonbox.east = east
    ground.latlonbox.west = west
    
   
    # Save to KMZ
    kml.savekmz(output_path)


def get_crs_info(raster_path):
        # Open the raster file
        with rasterio.open(raster_path) as dataset:
            # Get CRS
            crs = dataset.crs
            
            if not crs:
                return "CRS not found."

            # Initialize info dictionary
            crs_info = {'EPSG': None, 'UTM Zone': None, 'Hemisphere': None}
            
            # Extract EPSG code
            if crs.is_epsg_code:
                crs_info['EPSG'] = crs.to_epsg()
            
            # Convert CRS to WKT for easier parsing
            crs_wkt = crs.to_wkt()
            
            # Find UTM zone and hemisphere from WKT
            utm_match = re.search(r'UTM zone (\d+)(N|S)', crs_wkt)
            if utm_match:
                crs_info['UTM Zone'] = int(utm_match.group(1))
                crs_info['Hemisphere'] = 'north' if utm_match.group(2) == 'N' else 'south'

            return crs_info
from pyproj import Proj, transform

def convert_utm_to_latlon(easting, northing, zone, hemisphere):
    # Define the UTM projection string
    utm_proj = Proj(proj='utm', zone=zone, ellps='WGS84', south=hemisphere==hemisphere)
    
    # Define the latitude/longitude projection string
    latlon_proj = Proj(proj='latlong', ellps='WGS84')
    
    # Convert UTM to latitude and longitude
    longitude, latitude = transform(utm_proj, latlon_proj, easting, northing)
    return latitude, longitude


import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box
from rasterio.errors import CRSError
from pyproj import Transformer, CRS as PyProjCRS
from rasterio.transform import from_origin
import akhdefo_functions
import shutil
import tempfile
import os

#############################################################
def save_images_to_temp_folder(image_path1, image_path2):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Define the destination paths within the temporary directory
    dest_path1 = os.path.join(temp_dir, os.path.basename(image_path1))
    dest_path2 = os.path.join(temp_dir, os.path.basename(image_path2))
    
    # Copy images to the temporary directory
    shutil.copy(image_path1, dest_path1)
    shutil.copy(image_path2, dest_path2)
    
    akhdefo_functions.crop_to_overlap(temp_dir)
    
    # Print the paths of the saved images
    #print("Image 1 saved at:", dest_path1)
    #print("Image 2 saved at:", dest_path2)
    
    return dest_path1, dest_path2, temp_dir

###################################################3



################################################

def delete_temp_folder(temp_folder_path):
    # Check if the folder exists
    if os.path.exists(temp_folder_path):
        # Remove the directory and all its contents
        shutil.rmtree(temp_folder_path)

def crop_rasters_and_return_info(path1, path2):
    
    path1, path2, temp_dir=save_images_to_temp_folder(path1, path2)
    
    
    
    #try:
    with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
        # Use EPSG code directly for both sources if they are the same and well-defined.
        epsg_code = 'EPSG:32610'  # Standard EPSG code for WGS 84 / UTM zone 10N
        src_crs=src2.crs
        bbox1 = box(*src1.bounds)
        bbox2 = box(*src2.bounds)

        intersection = bbox1.intersection(bbox2)
        if intersection.is_empty:
            return "No overlapping area found."

        window1 = rasterio.windows.from_bounds(*intersection.bounds, src1.transform)
        window2 = rasterio.windows.from_bounds(*intersection.bounds, src2.transform)

        data1 = src1.read(1, window=window1, masked=True)
        data2 = src2.read(1, window=window2, masked=True)
       
        
            # Mask no data values
        if src1.nodata is not None:
            data1 = np.where(data1 == src1.nodata, np.nan, data1)
        else:
            data1 = data1  # Use original data if no no_data_value defined
            
              # Mask no data values
        if src2.nodata is not None:
            data2 = np.where(data2 == src2.nodata, np.nan, data2)
        else:
            data2 = data2  # Use original data if no no_data_value defined
        
        
        
        crs_info = get_crs_info(path2)
        utm_zone=crs_info['UTM Zone']
        hemisphere=crs_info['Hemisphere']
        
        # Convert intersection bounds to geographic coordinates
        west, south = src1.xy(intersection.bounds[1], intersection.bounds[0])
        east, north = src1.xy(intersection.bounds[3], intersection.bounds[2])

        # Calculate the width and height of the bounding box
        width = intersection.bounds[2] - intersection.bounds[0]
        height = intersection.bounds[3] - intersection.bounds[1]
        
        #Transform of the subset intersection
        subset_transform=src1.window_transform(window1)
        # Extracting values from the transform
        west = subset_transform.c
        north = subset_transform.f
        pixel_size_x = subset_transform.a
        pixel_size_y = -subset_transform.e 
        
        # Calculating the east and south boundaries
        east = west + (width * pixel_size_x)
        south = north - (height * pixel_size_y)

        #print(north, south, east, west)
        
        # east, north=convert_utm_to_latlon(east, north, crs_info['UTM Zone'], crs_info['Hemisphere'] )
        # west, south=convert_utm_to_latlon(west, south, crs_info['UTM Zone'], crs_info['Hemisphere'] )
        
            # Manually create a CRS object for EPSG:4326 if from_epsg fails
        epsg_4326_crs = PyProjCRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")

        # Check if the source CRS is not EPSG:4326
        if src_crs != epsg_4326_crs:
            # # Create a transformer to convert from source CRS to EPSG:4326
            # transformer = Transformer.from_crs(src_crs, epsg_4326_crs, always_xy=True)
            
            # # Transform each corner of the bounding box
            # west, south = transformer.transform(intersection.bounds[2], intersection.bounds[0])
            # east, north = transformer.transform(intersection.bounds[3], intersection.bounds[2])
            # Assume transformation is needed to WGS84
            target_crs = PyProjCRS.from_epsg(4326)
            # transformer = Transformer.from_crs(src1.crs, target_crs, always_xy=True)
            # # Perform the transformation
            # west, south = transformer.transform(west, south)
            # east, north = transformer.transform(east, north)
            west, south, east, north = transform_bounds(src1.crs, target_crs, *intersection.bounds)
            ####################
            import math
            # Constants
            meters_per_degree_latitude = 111320  # meters per degree of latitude
            latitude = north
            # Conversion formulas
            pixel_size_y_in_degrees = pixel_size_y / meters_per_degree_latitude
            pixel_size_x_in_degrees = pixel_size_x / (meters_per_degree_latitude * math.cos(math.radians(latitude)))
            pixel_size_x_in_degrees, pixel_size_y_in_degrees
            subset_transform_updated=from_origin(west, north, pixel_size_x_in_degrees, pixel_size_y_in_degrees)
        else:
            west, south , east, north 
            subset_transform_updated=subset_transform
        
        #print(north, south, east, west)
        from rasterio.coords import BoundingBox
        
        
        
        import math
        # Constants
        meters_per_degree_latitude = 111320  # meters per degree of latitude
        latitude = north
        # Conversion formulas
        pixel_size_y_in_degrees = pixel_size_y / meters_per_degree_latitude
        pixel_size_x_in_degrees = pixel_size_x / (meters_per_degree_latitude * math.cos(math.radians(latitude)))
        pixel_size_x_in_degrees, pixel_size_y_in_degrees
        subset_transform_updated=from_origin(west, north, pixel_size_x_in_degrees, pixel_size_y_in_degrees)
        
        
        
        
        # Define the custom bounds
        bounds = BoundingBox(left=west, bottom=south, right=east, top=north)
        print('bounds: ', bounds)

        from scipy.ndimage import zoom
        data2 = np.where(data2 == -32767, np.nan, data2)
        data1 = np.where(data1 == -32767, np.nan, data1)
        # Calculate the zoom factor
        zoom_factor = len(data1) / len(data2)
         # Get the no-data value from the dataset's metadata
        
        # Use the zoom function to resize array1
        data2 = zoom(data2, zoom_factor, order=1)
        data2 = np.where(data2 == src2.nodata, np.nan, data2)
        
        
        
        
        
            # Prepare the return information including pixel dimensions and bounding box size
        info1 = {
            'array': data1,
            'transform': src1.window_transform(window1),
            'crs': src1.crs,
            'resolution': (src1.res[0], src1.res[0]),
            'bbox_size': {'width': width, 'height': height},
            'bounds_latlong': {'north': north, 'south': south, 'east': east, 'west': west} , 'utm_zone':utm_zone,
            'trasnform_latlon':subset_transform_updated, 'resolution_latlon':(pixel_size_x_in_degrees, pixel_size_y_in_degrees)
        }
        info2 = {
            'array': data2,
            'transform': src2.window_transform(window2),
            'crs':src2.crs,
            'resolution': (src1.res[0], src1.res[1]),
            'bbox_size': {'width': width, 'height': height},
            'bounds_latlong': {'north': north, 'south': south, 'east': east, 'west': west}, 'utm_zone':utm_zone, 
            'trasnform_latlon':subset_transform_updated, 'resolution_latlon':(pixel_size_x_in_degrees, pixel_size_y_in_degrees)
        }
        
    
        
    src1.close()
    src2.close()
    delete_temp_folder(temp_dir)
    return info1, info2
        
        
        
    # except CRSError as e:
    #     raise RuntimeError(f"CRS transformation error: {e}")
    # except Exception as e:
    #     raise RuntimeError(f"An unexpected error occurred: {e}")
    
    

# Usage
# info1, info2 = crop_rasters_and_return_info(path_to_raster1, path_to_raster2)














def akhdefo_viewer(path_to_dem_file, raster_file, output_folder, title='', 
                   pixel_resolution_meters=3.125, output_file_name="", 
                   alpha=0.5, unit_conversion=None, no_data_mask=False, 
                   colormap='jet', min_value=None, max_value=None, 
                   normalize=False, colorbar_label=None, show_figure=True , aspect_raster=None, cmap_aspect=None, step=10):
    """
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
    """
    try:
        
        # with rasterio.open(path_to_dem_file) as src_dem:
        #     # Number of bands
        #     band_count = src_dem.count
        #     xres, yres=src_dem.res
        #     if band_count >2:
        #         dem = src_dem.read(masked=True)
        #         dem_transform = src_dem.transform
        #         hillshade=dem
                
        #     else:
        #         dem = src_dem.read(1, masked=True)
        #         dem_transform = src_dem.transform
            
        #         hillshade = es.hillshade(dem)
                

        # with rasterio.open(raster_file) as src_raster:
        #     raster = src_raster.read(1, masked=True)
        #     raster_transform = src_raster.transform
        #     raster_crs = src_raster.crs
    

        dem_data, raster_data=crop_rasters_and_return_info(path_to_dem_file, raster_file)
        
        if aspect_raster is not None:
            
            _, aspect_data=crop_rasters_and_return_info(path_to_dem_file, aspect_raster)
            aspect_arr=aspect_data['array']
            aspect_transform=aspect_data['trasnform_latlon'] 
        else:
            aspect_arr=None
            aspect_transform=None
            
        dem_arr=dem_data['array']
        dem_transform=dem_data['transform'] 
        dem_res=dem_data['resolution']
        dem_crs=dem_data['crs']
        utm_zone=dem_data['utm_zone']
        raster=raster_data['array']
        raster_transform=raster_data['transform']
        raster_res=raster_data['resolution']
        raster_crs=raster_data['crs']
        xres=dem_res[0]
        hillshade = es.hillshade(dem_arr)
        
        with rasterio.open(path_to_dem_file) as src_dem:
            # Number of bands
            band_count = src_dem.count
            if band_count >2:
                hillshade=dem_arr
        
        # Assuming you've already executed the function and have the results stored in cropped_info1 and cropped_info2
        width = dem_data['bbox_size']['width']
        height = dem_data['bbox_size']['height']
        
        west=dem_data['bounds_latlong']['west']
        south=dem_data['bounds_latlong']['south']
        east=dem_data['bounds_latlong']['east']
        north=dem_data['bounds_latlong']['north']
        
        if no_data_mask:
            raster = np.ma.masked_where(raster == 0, raster)

        if unit_conversion:
            unit_type, unit_factor = _separate_floats_letters(unit_conversion)
            raster *= float(unit_factor)

        if pixel_resolution_meters is None:
            pixel_resolution_meters=xres

        # Set the output file name if it's not provided
        if not output_file_name:
            output_file_name = os.path.splitext(os.path.basename(raster_file))[0] + ".png"

        _create_plot(hillshade=hillshade, raster=raster, dem_transform=dem_transform, raster_transform=raster_transform, raster_crs=raster_crs, alpha=alpha, colormap=colormap, normalize=normalize,
                     title=title, output_folder=output_folder, output_file_name=output_file_name, colorbar_label=colorbar_label, pixel_resolution_meters=pixel_resolution_meters,
                     min_value=min_value, max_value=max_value, aspect_raster=aspect_arr, cmap_aspect=cmap_aspect, step=step, UTM_zone=utm_zone)
        
        
        create_kmz_overlay(hillshade=hillshade, raster=raster, colormap=colormap, alpha=alpha, west=west,south=south,east=east,north=north,output_folder= output_folder, 
                           output_file_name=output_file_name+'.kmz', colorbar_label=colorbar_label, normalize=normalize, min_value=min_value, max_value=max_value, title=title, 
                           aspect_raster=aspect_arr, step=step, cmap_aspect=cmap_aspect, transform=aspect_transform)

        
       
       
    
        
        

        if show_figure:
            plt.show()
        else:
            plt.close()
            
            
        # create_kmz_with_overlay(image_path=os.path.join(output_folder, output_file_name),raster_path=raster_file, output_path=os.path.join(output_folder, output_file_name[:-4]+'.kmz'),
        #                         bbox= None )

    except Exception as e:
        raise RuntimeError(f"An error occurred: {e}")






def _separate_floats_letters(input_string):
    """
    Separates floats and letters from a string.
    """
    floats = re.findall(r'\d+\.\d+|\d+', input_string)
    letters = re.findall(r'[a-zA-Z]+', input_string)
    if not floats or not letters:
        raise ValueError("Invalid input string for unit conversion.")
    return letters[0], floats[0]

def _normalize_raster(raster, min_value, max_value):
    """
    Normalizes raster values between given minimum and maximum values.
    """
    if min_value is None and max_value is None:
        min_value = np.nanmin(raster)
        max_value = np.nanmax(raster)
    else:
        min_value=min_value
        max_value=max_value
       

    if np.nanmin(raster) < 0 and np.nanmax(raster) >0:
        norm = TwoSlopeNorm(vmin=min_value, vcenter=0, vmax=max_value)
    else:
        norm = Normalize(vmin=min_value, vmax=max_value)
    
    
    
    return raster, norm


def _create_plot(hillshade, raster, dem_transform, raster_transform, raster_crs, alpha, colormap, normalize,
                 title, output_folder, output_file_name, colorbar_label, pixel_resolution_meters, min_value, max_value , aspect_raster=None, cmap_aspect=None, step=10 , UTM_zone=None):
    """
    Creates and saves a plot of hillshade and raster overlay.
    """
    
    basemap_dimensions = hillshade.ndim
  
        
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot the hillshade layer using dem_transform for its extent
    if basemap_dimensions > 2:
        #ep.plot_rgb(hillshade, rgb=(0, 1, 2), str_clip=2, ax=ax, extent=rasterio.plot.plotting_extent(hillshade, transform=dem_transform))
        rasterio.plot.show(hillshade, transform=dem_transform, ax=ax)
    else:
        
           
        #rasterio.plot.show(hillshade,extent=rasterio.plot.plotting_extent(hillshade, transform=dem_transform), ax=ax , cmap='gray')
        ax.imshow(hillshade,extent=rasterio.plot.plotting_extent(hillshade, transform=dem_transform), cmap='gray')
        # ep.plot_bands(
        #     hillshade,
        #     ax=ax,
        #     cmap='gray',
        #     scale=True,
        #     cbar=False,
        #     extent=rasterio.plot.plotting_extent(raster, transform=raster_transform))  # ensure correct transform
        # # )

    if aspect_raster is not None:
        alpha_basemap=0.45
    else:
        alpha_basemap=alpha
    
    if normalize==True:
        raster, norm = _normalize_raster(raster, min_value, max_value)
        
    #if normalize==True:
    # Overlay the raster with alpha for transparency using raster_transform for its extent
      
        
        img = ax.imshow(raster, alpha=alpha_basemap, cmap=colormap, norm=norm, 
                    extent=rasterio.plot.plotting_extent(hillshade, transform=dem_transform))  # ensure correct transform
    if normalize==False:
        
        if min_value is None and max_value is None:
            min_value = np.nanmin(raster)
            max_value = np.nanmax(raster)
        else:
            min_value=min_value
            max_value=max_value
       
      
                 # Overlay the raster with alpha for transparency using raster_transform for its extent
        img = ax.imshow(raster, alpha=alpha_basemap, cmap=colormap, vmin=min_value , vmax=max_value, 
                        extent=rasterio.plot.plotting_extent(hillshade, transform=dem_transform))  # ensure correct transform
    if aspect_raster is not None:
        def aspect_to_uv(aspect):
            """
            Convert aspect data to U and V components for arrows.
            """
            aspect_rad = np.deg2rad(aspect)
            u = np.sin(aspect_rad)
            v = np.cos(aspect_rad)
            return u, v
        # # Load the raster file
        # with rasterio.open(aspect_raster) as dataset:
        #     # Read the aspect data
        #     aspect_data = dataset.read(1)
            
        #     # Get the geotransformation data
        #     transform = dataset.transform

        #     # Get the shape of the data
        #     data_shape = aspect_data.shape
        aspect_data=aspect_raster
        data_shape=aspect_data.shape
        # Generate a grid of points every 10 pixels
        step=step
        x_positions = np.arange(0, data_shape[1], step)
        y_positions = np.arange(0, data_shape[0], step)
        x_grid, y_grid = np.meshgrid(x_positions, y_positions)

        # Subset the aspect data to match the grid
        aspect_subset = aspect_data[y_positions[:, None], x_positions]
        aspect_subset, norm = _normalize_raster(aspect_subset, min_value=None, max_value=None)
        u_subset, v_subset = aspect_to_uv(aspect_subset)

        # Convert grid positions to real world coordinates
        x_grid_world, y_grid_world = dem_transform * (x_grid, y_grid)
        if cmap_aspect is None:
            cmap_aspect='hsv'
        quiver = ax.quiver(x_grid_world, y_grid_world, u_subset, v_subset, aspect_subset, scale=20, cmap=cmap_aspect , angles='xy', norm=norm, alpha=alpha)
        # Adding a second colorbar in a horizontal position
        cbar_ax2 = fig.add_axes([0.25, 0.04, 0.5, 0.02])  # Position for the horizontal colorbar
        cbar2 = fig.colorbar(quiver, cax=cbar_ax2, orientation='horizontal',  extend='both')  # Using the ScalarMappable created earlier
        cbar2.set_label('Aspect-Colorbar(degrees)')
            

    # Add colorbar
    if colorbar_label:
        cbar_ax = fig.add_axes([0.92, 0.22, 0.02, 0.5])
        fig.colorbar(img, cax=cbar_ax, label=colorbar_label, extend='both')

    # Set axis labels based on CRS
    if raster_crs.is_geographic:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    else:
        ax.set_xlabel('Easting')
        ax.set_ylabel('Northing')

    #ax.grid(True, which='major')
    ax.set_title(title)

   
        
    scalebar = ScaleBar(1, location='lower right', units='m',
                        frameon=True, scale_loc='bottom', dimension='si-length', box_color='white', color='k', border_pad=1, box_alpha=0.65)  # Adjust parameters as needed
   
    ax.add_artist(scalebar)
    

    # Save the plot
    plt.savefig(os.path.join(output_folder, output_file_name), dpi=100, bbox_inches='tight')





from simplekml import Kml, OverlayXY, ScreenXY, Units, RotationXY, AltitudeMode, Camera

def gearth_fig(west, south, east, north, pixels=1024):
    aspect = np.cos(np.mean([south, north]) * np.pi/180.0)
    xsize = np.ptp([east, west]) * aspect
    ysize = np.ptp([north, south])
    aspect = ysize / xsize

    if aspect > 1.0:
        figsize = (10.0 / aspect, 10.0)
    else:
        figsize = (10.0, 10.0 * aspect)

    fig = plt.figure(figsize=figsize, frameon=False, dpi=pixels // 10)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    return fig, ax

def make_kml(west, south, east, north,
             figs, colorbar=None, **kw):
    """TODO: LatLon bbox, list of figs, optional colorbar figure,
    and several simplekml kw..."""

    kml = Kml()
    altitude = kw.pop('altitude', 2e7)
    roll = kw.pop('roll', 0)
    tilt = kw.pop('tilt', 0)
    altitudemode = kw.pop('altitudemode', AltitudeMode.relativetoground)
    camera = Camera(latitude=np.mean([north, south]),
                    longitude=np.mean([east, west]),
                    altitude=altitude, roll=roll, tilt=tilt,
                    altitudemode=altitudemode)

    #kml.document.camera = camera
    draworder = 0
    for fig in figs:  # NOTE: Overlays are limited to the same bbox.
        draworder += 1
        ground = kml.newgroundoverlay(name='GroundOverlay')
        ground.draworder = draworder
        # ground.visibility = kw.pop('visibility', 1)
        # #ground.name = kw.pop('name', 'overlay')
        ground.color = kw.pop('color', '9effffff')
        ground.atomauthor = kw.pop('author', 'ocefpaf')
        ground.latlonbox.rotation = kw.pop('rotation', 0)
        ground.description = kw.pop('description', 'Matplotlib figure')
        ground.gxaltitudemode = kw.pop('gxaltitudemode',
                                       'clampToSeaFloor')
        ground.icon.href = fig
        ground.latlonbox.east = east
        ground.latlonbox.south = south
        ground.latlonbox.north = north
        ground.latlonbox.west = west

    if colorbar:  # Options for colorbar are hard-coded (to avoid a big mess).
        screen = kml.newscreenoverlay(name='Legend')
        screen.icon.href = colorbar
        screen.overlayxy = OverlayXY(x=0, y=0,
                                     xunits=Units.fraction,
                                     yunits=Units.fraction)
        screen.screenxy = ScreenXY(x=0.015, y=0.075,
                                   xunits=Units.fraction,
                                   yunits=Units.fraction)
        screen.rotationXY = RotationXY(x=0.5, y=0.5,
                                       xunits=Units.fraction,
                                       yunits=Units.fraction)
        screen.size.x = 0
        screen.size.y = 0
        screen.size.xunits = Units.fraction
        screen.size.yunits = Units.fraction
        screen.visibility = 1

    kmzfile = kw.pop('kmzfile', 'overlay.kmz')
    
    
    kml.savekmz(kmzfile)
  

def create_kmz_overlay(hillshade, raster, colormap, alpha, west, south, east, north, output_folder, output_file_name, pixels=2030, colorbar_label='colorbar_label',
                       normalize=True, min_value= None, max_value=None, title='title', aspect_raster=None, step=10, cmap_aspect='hsv', transform=None):
    
    fig_ov, ax = gearth_fig(west, south, east, north, pixels)
    ax.imshow(hillshade, alpha=0.85,cmap='gray', extent=[west, east, south, north])
    
   
    if normalize:
        raster, norm = _normalize_raster(raster, min_value, max_value)
        img = ax.imshow(raster, cmap=colormap, alpha=0.75, extent=[west, east, south, north], norm=norm)
    
    img = ax.imshow(raster, cmap=colormap, alpha=0.75, extent=[west, east, south, north])
    ax.set_axis_off()
    fig_path = os.path.join(output_folder, 'overlay' + '.jpg')
    
    
    ########ADD qquiver plot############
    if aspect_raster is not None:
        def aspect_to_uv(aspect):
            """
            Convert aspect data to U and V components for arrows.
            """
            aspect_rad = np.deg2rad(aspect)
            u = np.sin(aspect_rad)
            v = np.cos(aspect_rad)
            return u, v
        aspect_data=aspect_raster
        data_shape=aspect_data.shape
        # Generate a grid of points every 10 pixels
        step=step
        x_positions = np.arange(0, data_shape[1], step)
        y_positions = np.arange(0, data_shape[0], step)
        x_grid, y_grid = np.meshgrid(x_positions, y_positions)

        # Subset the aspect data to match the grid
        aspect_subset = aspect_data[y_positions[:, None], x_positions]
        aspect_subset, norm = _normalize_raster(aspect_subset, min_value=None, max_value=None)
        u_subset, v_subset = aspect_to_uv(aspect_subset)

        # Convert grid positions to real world coordinates
        x_grid_world, y_grid_world = transform * (x_grid, y_grid)
        
        if cmap_aspect is None:
            cmap_aspect='hsv'
    if aspect_raster is not None:
        quiver = ax.quiver(x_grid_world, y_grid_world, u_subset, v_subset, aspect_subset, scale=20, cmap=cmap_aspect , angles='xy', norm=norm, alpha=1)
        
    
    fig_ov.savefig(fig_path, dpi=fig_ov.dpi, transparent=False, bbox_inches='tight')
    plt.close(fig_ov)
        ##############################

    if aspect_raster is not None:
        figsize=(4.0, 2.0)
        colorbar_path=os.path.join(output_folder, 'legend' + '.jpg')
    else:
        figsize=(4.0, 1.0)
        colorbar_path=os.path.join(output_folder, 'legend' + '.jpg')
        
        
        
    if aspect_raster is not None:
        
        fig_cb = plt.figure(figsize=figsize, facecolor='white', frameon=True)

        # First Colorbar
        ax_cb1 = fig_cb.add_axes([0.05, 0.8, 0.9, 0.1])  # Adjusted the position of the axes for the first colorbar
        cb1 = plt.colorbar(img, cax=ax_cb1, orientation='horizontal', extend='both')  # First colorbar
        cb1.set_label(colorbar_label, color='k', labelpad=1)  # Adjusted labelpad
        
        # Second Colorbar
        ax_cb2 = fig_cb.add_axes([0.05, 0.3, 0.9, 0.1])  # Adjusted the position of the axes for the second colorbar
        cb2 = plt.colorbar(quiver, cax=ax_cb2, orientation='horizontal', extend='both')  # Second colorbar
        cb2.set_label('Aspect-Colorbar(degrees)', color='k', labelpad=1)  # Adjusted labelpad
        #plt.tight_layout()  # Ensures all elements are included with tight layout
        fig_cb.savefig(colorbar_path, transparent=False)
        plt.close(fig_cb)
        make_kml(west, south, east, north, [fig_path], kmzfile=os.path.join(output_folder, output_file_name[:-4] + '.kmz'), colorbar=colorbar_path, name=title)
    
    else:
        fig_cb = plt.figure(figsize=figsize, facecolor='white', frameon=True)
        # First Colorbar
        ax_cb1 = fig_cb.add_axes([0.05, 0.6, 0.9, 0.2])  # Adjusted the position of the axes for the first colorbar
        cb1 = plt.colorbar(img, cax=ax_cb1, orientation='horizontal', extend='both')  # First colorbar
        cb1.set_label(colorbar_label, color='k', labelpad=1)  # Adjusted labelpad
        #plt.tight_layout()  # Ensures all elements are included with tight layout
        fig_cb.savefig(colorbar_path, transparent=False) 
        plt.close(fig_cb)
        
        make_kml(west, south, east, north, [fig_path], kmzfile=os.path.join(output_folder, output_file_name[:-4] + '.kmz'), colorbar=colorbar_path, name=title)
        
    #Delete temp files  
    os.remove(fig_path)
    os.remove(colorbar_path)
        
        
        

    
       
    
    
    
    
    
    
    # fig_cb = plt.figure(figsize=(4.0, 1.0), facecolor='white', frameon=True)
    # ax_cb = fig_cb.add_axes([0.05, 0.6, 0.9, 0.2])  # Adjusted the position of the axes
    # cb = plt.colorbar(img, cax=ax_cb, orientation='horizontal', extend='both')  # Changed orientation and extended both ends
    # cb.set_label(colorbar_label, color='k', labelpad=10)  # Adjusted labelpad
    
    # ###second colorbar##
    # if aspect_raster is not None:
    #     # Adding a second colorbar in a horizontal position
    #     cbar_ax2 = fig_cb.add_axes([0.25, 0.2, 0.5, 0.02])  # Position for the horizontal colorbar
    #     cbar2 = fig_cb.colorbar(quiver, cax=cbar_ax2, orientation='horizontal',  extend='both')  # Using the ScalarMappable created earlier
    #     cbar2.set_label('Aspect-Colorbar(degrees)')
        
    # plt.tight_layout()
    # fig_cb.savefig(colorbar_path, transparent=False)
    # plt.close(fig_cb)
   
    # fig_cb = plt.figure(figsize=(1.0, 4.0), facecolor='white', frameon=True)
    # ax_cb = fig_cb.add_axes([0.0, 0.05, 0.2, 0.9])
    # cb = plt.colorbar(img, cax=ax_cb)
    # cb.set_label(colorbar_label, rotation=90, color='k', labelpad=20)
    
    # fig_cb.savefig(colorbar_path, transparent=False)
    # plt.close(fig_cb)

   

    
    
   


def plot_stackNetwork(src_folder="", output_folder="", cmap='tab20', date_plot_interval=(5, 30), marker_size=15):
    
    """
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
    
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    path_folder = sorted([os.path.join(src_folder, file) for file in os.listdir(src_folder) if file.endswith('.tif')])
    data = []
    for filepath in path_folder:
        filename = os.path.basename(filepath)

        # Using regular expression to find dates in filename
        multi_date_pattern = re.compile(r'(\d{8})')
        single_date_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}|\d{8})')
        
        multi_dates = multi_date_pattern.findall(filename)
        if len(multi_dates) >= 2:
            # Convert dates to datetime objects and calculate delta_dd for the first and last date
            date_objs = [pd.to_datetime(date, format='%Y%m%d') for date in multi_dates]
            delta_dd = (date_objs[-1] - date_objs[0]).days
            data.append({'Time': date_objs[-1], 'Delta_DD': delta_dd})
        else:
            single_dates = single_date_pattern.findall(filename)
            if single_dates:
                # Convert single found date to datetime object
                date_format = '%Y-%m-%d' if '-' in single_dates[0] else '%Y%m%d'
                data.append({'Time': pd.to_datetime(single_dates[0], format=date_format), 'Delta_DD': None})

    # Calculate delta_dd for single dates
    sorted_data = sorted(data, key=lambda x: x['Time'])
    for i in range(len(sorted_data) - 1):
        if sorted_data[i]['Delta_DD'] is None:
            sorted_data[i]['Delta_DD'] = (sorted_data[i + 1]['Time'] - sorted_data[i]['Time']).days

    # Preparing data for plotting
    df = pd.DataFrame(sorted_data)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.autofmt_xdate()
    sc = ax.scatter(df['Time'], df['Delta_DD'], c=mdates.date2num(df['Time']), cmap=cmap, s=marker_size)
    ax.plot(df['Time'], df['Delta_DD'], color='k', alpha=0.5)
    cb = fig.colorbar(sc, ax=ax, orientation='horizontal', pad=0.02)

    loc_major = mdates.AutoDateLocator(minticks=date_plot_interval[0], maxticks=date_plot_interval[1])
    cb.ax.xaxis.set_major_locator(loc_major)
    cb.ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc_major))

    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=date_plot_interval[1]))
    plt.xlim(df['Time'].min(), df['Time'].max())
    plt.ylabel("Days Between Dates")
    cb.ax.tick_params(rotation=90)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Dates')
    plt.xticks(rotation=85)
    ax.grid(True)
    plt.savefig(os.path.join(output_folder, "Stack_Network.jpg"), dpi=300, bbox_inches='tight')




    

def akhdefo_ts_plot(path_to_shapefile=r"", dem_path=r"", point_size=1.0, opacity=0.75, cmap="turbo",
                    Set_fig_MinMax=True, MinMaxRange=[-50,50] , color_field='VEL', user_data_points="", 
                    path_saveData_points="" , save_plot=False, Fig_outputDir='' , VEL_Scale='year', filename_dates=''):
    '''
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
    
    '''
     #####################################################################
   
    
    #################################################
    def ts_plot(df, plot_number, save_plot=False , output_dir="", plot_filename="" , VEL_Scale=VEL_Scale):
    
       
        pio.renderers.default = 'plotly_mimetype'
        
        df=pd.read_csv("temp.csv")
        

        df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
        df['dd_str']=df['dd'].astype(str)
        df['dd_str'] = df['dd_str'].astype(str)
        df.rename(columns={ df.columns[1]: "val" }, inplace = True)
        df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y%m%d')
        
        df=df.set_index('dd')
        
        ########################
        df=df.dropna()
        # Make index pd.DatetimeIndex
        df.index = pd.DatetimeIndex(df.index)
        # Make new index
        idx = pd.date_range(df.index.min(), df.index.max())
        # Replace original index with idx
        df = df.reindex(index = idx)
        # Insert row count
        df.insert(df.shape[1],
                'row_count',
                df.index.value_counts().sort_index().cumsum())

        df=df.dropna()
        
        #df=df.set_index(df['row_count'], inplace=True)

        df.sort_index(ascending=True, inplace=True)
    
    
    
    
        
        
        #####start building slider
        widgets.SelectionRangeSlider(
        options=df.index,
        description='Dates',
        orientation='horizontal',
        layout={'width': '1000px'})
        
        
        ############
        def ts_helper(df, VEL_Scale=VEL_Scale, plot_number=plot_number):
            
            mean_vel=df['val'].mean()
            mean_vel_std=df['val'].std()
            def best_fit_slope_and_intercept(xs,ys):
                from statistics import mean
                xs = np.array(xs, dtype=np.float64)
                ys = np.array(ys, dtype=np.float64)
                m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
                    ((mean(xs)*mean(xs)) - mean(xs*xs)))
                
                b = mean(ys) - m*mean(xs)
                
                return m, b

            #convert dattime to number of days per year
            
            
            dates_list=([datetime.strptime(x, '%Y%m%d') for x in df.dd_str])
            #days_num=[( ((x) - (pd.Timestamp(year=x.year, month=1, day=1))).days + 1) for x in dates_list]
            dd_days=dates_list[len(dates_list)-1]- dates_list[0]
            print(dates_list[0], dates_list[len(dates_list)-1] , dd_days)
            dd_days=str(dd_days)
            dd_days=dd_days.removesuffix('days, 0:00:00')
            delta=int(dd_days)
            m, b = best_fit_slope_and_intercept(df.row_count, df.val)
            print("m:", math.ceil(m*100)/100, "b:",math.ceil(b*100)/100)
            regression_model = LinearRegression()
            val_dates_res = regression_model.fit(np.array(df.row_count).reshape(-1,1), np.array(df.val))
            y_predicted = regression_model.predict(np.array(df.row_count).reshape(-1,1))
        
            if VEL_Scale=='year':
                rate_change=regression_model.coef_[0]/delta * 365.0
                std=np.std(y_predicted)/delta * 365
            elif VEL_Scale=='month':
                rate_change=regression_model.coef_[0]/delta * 30.0
                std=np.std(y_predicted)/delta * 30
            else:
                rate_change=regression_model.coef_[0]  #delta is number of days
                std=np.std(y_predicted)
                
            # model evaluation
            mse=mean_squared_error(np.array(df.val),y_predicted)
            rmse = np.sqrt(mean_squared_error(np.array(df.val), y_predicted))
            r2 = r2_score(np.array(df.val), y_predicted)
            
            # printing values
            slope= ('Slope(linear deformation rate):' + str(math.ceil((regression_model.coef_[0]/delta)*100)/100) + " mm/day")
            Intercept=('Intercept:'+ str(math.ceil(b*100)/100))
            #print('MSE:',mse)
            rmse=('Root mean squared error: '+ str(math.ceil(rmse*100)/100))
            r2=('R2 score: '+ str(r2))
            
            std=("STD: "+ str(math.ceil(std*100)/100)) 
            # Create figure
            #fig = go.Figure()
            
            return y_predicted, rate_change, slope, Intercept, rmse, r2, std, plot_number, print(len(df)), dd_days, mean_vel, mean_vel_std
        
        
        @interact
        def read_values(
            slider = widgets.SelectionRangeSlider(
            options=df.index,
            index=(0, len(df.index) - 1),
            description='Dates',
            orientation='horizontal',
            layout={'width': '500px'},
            continuous_update=True) ):
            
            #df=pd.read_csv("temp.csv")
            df=pd.read_csv("temp.csv")

            df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
            df['dd_str']=df['dd'].astype(str)
            df['dd_str'] = df['dd_str'].astype(str)
            df.rename(columns={ df.columns[1]: "val" }, inplace = True)
            df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y%m%d')
            
            df=df.set_index('dd')
            
            ########################
            df=df.dropna()
            # Make index pd.DatetimeIndex
            df.index = pd.DatetimeIndex(df.index)
            # Make new index
            idx = pd.date_range(df.index.min(), df.index.max())
            # Replace original index with idx
            df = df.reindex(index = idx)
            # Insert row count
            df.insert(df.shape[1],
                    'row_count',
                    df.index.value_counts().sort_index().cumsum())

            df=df.dropna()
            
            #df=df.set_index(df['row_count'], inplace=True)

            df.sort_index(ascending=True, inplace=True)
            
            
            
            df=df.loc[slider[0]: slider[1]]
            
            
             
            
            helper=ts_helper(df, VEL_Scale=VEL_Scale)
            
            y_predicted=helper[0]
            rate_change=helper[1]
            slope=helper[2]
            Intercept=helper[3]
            rmse=helper[4]
            r2=helper[5]
            std=helper[6]
            plot_number=helper[7]
            Mean_VEL=helper[10]
            Mean_VEL_STD=helper[11]
            
            print(rate_change)
            print(slope)
            print(rmse)
            print(std)
            print(Intercept)
            print(f'Mean_VEL: {Mean_VEL}')
            print(f'Mean_VEL: {Mean_VEL_STD}')
            
        

            fig = go.Figure()
            fig.update_xaxes(range=[slider[0], slider[1]])
            trace1 = go.Scatter(x=(df.index), y=(y_predicted), mode='lines', name='Trendline')
            fig.add_trace(trace1)
            trace2 = go.Scatter(x=(df.index), y=(df.val), mode='markers', name='Data-Points')
            fig.add_trace(trace2)
            trace3 = go.Scatter(x=(df.index), y=(df.val), mode='lines', name='Draw-line', visible='legendonly')
            fig.add_trace(trace3)
            
            fig.update_layout(xaxis_title="Date", yaxis_title="millimeter")
            
            unit=helper[9]+ "days"
            if VEL_Scale=="year":
                unit="year"
            elif VEL_Scale=="month":
                unit="month"
            else:
                unit=unit
            
            tt= "Linear-VEL:"+str(round(rate_change,2))+"mm/"+unit+":"+ "Linear-VEL-STD:"+str(round(np.std(y_predicted), 2))+ f' Mean-VEL: {round(Mean_VEL, 3)} mm/{unit} , Mean-VEL-STD: {round(Mean_VEL_STD, 3)}' + " : Plot ID-" + str(plot_number)
            
            fig.update_layout(
                
            title_text=tt, title_font_family="Sitka Small",
            title_font_color="red", title_x=0.5 , legend_title="Legend",
            font=dict(
                family="Courier New, monospace",
                size=15,
                color="RebeccaPurple" ))
            
            fig.update_layout(font_family="Sitka Small")
            
            # fig.update_layout(legend=dict(
            # yanchor="top",
            # y=-0,
            # xanchor="left",
            # x=1.01))
            fig.update_xaxes(tickformat='%Y.%m.%d')
            fig.update_layout(xaxis = go.layout.XAxis( tickangle = 45))
            
            
            fig.update_layout(hovermode="x unified")
            
            f2=go.FigureWidget(fig.to_dict()).show()
           
            
            if save_plot==True:
            
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)

                path_profile=fig.write_html(output_dir + "/" + plot_filename + ".html" )
                fig.write_image(output_dir + "/" + plot_filename + ".jpeg", scale=1, width=1080, height=300 )
            
            f2
     
    
   #######################################################################3
   # Define the file name to check
    # filename_temp_dates = filename_dates 
    # # Check if the file exists in the current working directory
    # if os.path.exists(filename_temp_dates):
    #     filename_dates=filename_temp_dates
    #     #print(f'The file "{filename_dates}" exists in the current directory.')
    # else:
    #     print(f'The file "{filename_dates}" does not exist in the current directory.\n Please enter path to Names.txt to filename_dates variable')
    
    # dnames=[]
    # with open(filename_dates, 'r') as fp:
    #     for line in fp:
    #         # remove linebreak from a current name
    #         # linebreak is the last character of each line
    #         x =  'D' + line[:-1]

    #         # add current item to the list
    #         dnames.append(x[:-18])

   
    
    # Import shapfilepath
    shapefile_path = os.path.join(path_to_shapefile)
    basename = os.path.basename(shapefile_path[:-4])
    # Open shapefile data with geopandas
    gdf = gpd.read_file(shapefile_path)
    gdf.crs
    dnames = [col for col in gdf.columns if col.startswith('D')]
    # Define path to dem data
    #dem_path = 'dem.tif'

    with rio.open(dem_path) as src:
        elevation = src.read(1)
        elevation = elevation.astype('float32')
        xres, yres=src.res
        # Set masked values to np.nan
        elevation[elevation < 0.0] = np.nan
        
    # Create and plot the hillshade with earthpy
    hillshade = es.hillshade(elevation, azimuth=270, altitude=45)

    dem = rxr.open_rasterio(dem_path, masked=True)
    dem_plotting_extent = plotting_extent(dem[0], dem.rio.transform())

    # Getting the crs of the raster data
    dem_crs = es.crs_check(dem_path)
    

    # Transforming the shapefile to the dem data crs
    gdf = gdf.to_crs(dem_crs)
    
    
    min=gdf[color_field].min()
    max=gdf[color_field].max()
    import matplotlib.colors as mcolors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    
    if min < 0 and max >0 :
        if Set_fig_MinMax==True:
            min_n=MinMaxRange[0]
            max_n=MinMaxRange[1]
            min=min_n
            max=max_n
            offset = mcolors.TwoSlopeNorm(vmin=min,
                        vcenter=0., vmax=max)  
        else:
            offset = mcolors.TwoSlopeNorm(vmin=min,
                        vcenter=0., vmax=max)
            
    else  : 
        if Set_fig_MinMax==True:
            min_n=0
            max_n=100
            min=MinMaxRange[0]
            max=MinMaxRange[1]
            offset=mcolors.Normalize(vmin=min, vmax=max)
        else:
            offset=mcolors.Normalize(vmin=min, vmax=max)


    if user_data_points!="":
        print("Click Anywhere on the Figure to plot Your Time-Series Profile and plot Point Locations")


    fig = plt.figure(figsize=(7,7))
    ax1 = fig.add_subplot(111)
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('bottom', size='5%', pad=0.5)
    
    #fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10,5))
    ep.plot_bands( hillshade,cbar=False,title=basename,extent=dem_plotting_extent,ax=ax1, scale=False)
    img_main=ax1.scatter(gdf.x, gdf.y, c=gdf[color_field], alpha=opacity, s=point_size, picker=1, cmap=cmap, norm=offset)
    #scalebar = ScaleBar(dx=xres, units='m', location='lower right',frameon=False, scale_loc='bottom')
    #scalebar = ScaleBar(xres, "m", scale_loc="right",border_pad=1,pad=0.5, box_color='white', box_alpha=0.5, location='lower right', ax=ax1)
    #ax1.add_artist(scalebar)
    plt.grid(True)
    #ax.scatter(gdf.x, gdf.y, s= 0.5, c=gdf.VEL_MEAN ,picker=1)
    cb=fig.colorbar(img_main, ax=ax1, cax=cax, extend='both', orientation='horizontal')
    cb.set_label('mm/year', labelpad=2, x=0.5, rotation=0)
    
    global count
    count=0
    
    x_list=[]
    y_list=[]
    label_ID_list=[]
    df_filt1_list=[]
    
    
    def onclick(event):
        global count
        count+=1
        
        global ix, iy
        ix, iy = event.xdata, event.ydata
        print('button=%d, Figure Coordinates: x=%d, y=%d, : Geographic Coordinates: xdata=%f, ydata=%f' % 
            (event.button, event.x, event.y, event.xdata, event.ydata))
        #plt.plot(event.xdata, event.ydata, ',')
        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return array[idx]
        def filter_rowwise(gdf,find_nearest, ix, iy):
            # Calculate the nearest x and y values
            nearest_x = find_nearest(gdf['x'], ix)
            nearest_y = find_nearest(gdf['y'], iy)

            # Select the row where both x and y are closest to the specified values
            selected_row = gdf[(gdf['x'] == nearest_x) & (gdf['y'] == nearest_y)].head(1)

            if len(selected_row) == 0:
                # If no exact match is found, select rows with closest x and y values separately
                x_match = gdf[gdf['x'] == nearest_x]
                y_match = gdf[gdf['y'] == nearest_y]

                if not x_match.empty and not y_match.empty:
                    # Select the row with the closest x and y values
                    selected_row = pd.concat([x_match, y_match]).drop_duplicates().head(1)

            if not selected_row.empty:
                # Reset the index and drop the 'CODE' column
                selected_row = selected_row.reset_index(drop=True)
                selected_row = selected_row.drop(columns=['CODE'])

                # Create a new 'Code' column based on the index
                selected_row['CODE'] = selected_row.index
            
                # Extract columns starting with 'D' and rename them
                dnames = [col for col in selected_row.columns if col.startswith('D')]
                selected_row=selected_row[dnames]
                selected_row.columns = selected_row.columns.str.replace(r"D", "")

                # Transpose the selected row and save it to a CSV file
                selected_row = selected_row.T
                selected_row.to_csv('temp.csv')

            return nearest_x, nearest_y
        
          
        if user_data_points!="":
            df_poi=pd.read_csv(user_data_points)
            for idx, row in df_poi.iterrows():
                #print("Index:", idx, ": ",  row['x'], row['y'])
                
                s1, s2=filter_rowwise(gdf,find_nearest, row['x'], row['y'] )
                
                
                output_dir=Fig_outputDir 
                plot_filename=basename+"_"+str(idx+1)
                    
                df=pd.read_csv('temp.csv')
                ps=ts_plot(df, idx+1, save_plot=save_plot, output_dir=Fig_outputDir, plot_filename=basename+"_"+str(idx+1), VEL_Scale=VEL_Scale)
                ax1.scatter(s1, s2,  marker=idx+1,  label=idx+1, s=100)
                ax1.text( s1, s2, idx+1, fontdict=dict(color="black"),
                bbox=dict(facecolor="white",alpha=0.75))
                
        else:
            s1, s2=filter_rowwise(gdf,find_nearest, ix, iy )
            
        
            df=pd.read_csv('temp.csv')
            
            output_dir=Fig_outputDir
            plot_filename=basename+"_"+str(count)
            
            ps=ts_plot(df, count, save_plot=save_plot, output_dir=Fig_outputDir, plot_filename=basename+"_"+str(count), VEL_Scale=VEL_Scale)
            
            print("count: " , count)

    
        ########################33
            
            #os.unlink("temp.csv")
            
            
            x_list.append(s1)
            y_list.append(s2)
            label_ID_list.append(count)
            
            
            ax1.scatter(event.xdata, event.ydata,  marker=count,  label=count, s=100)
            ax1.text( event.xdata, event.ydata, count, fontdict=dict(color="black"),
                bbox=dict(facecolor="white",alpha=0.75))
        
        
        
        if path_saveData_points!="": 
            if not os.path.exists(path_saveData_points):
                os.mkdir(path_saveData_points)
            df_filt1=pd.concat(df_filt1_list)
            df_filt1['x']=x_list
            df_filt1['y']=y_list
            df_filt1['ID']=label_ID_list
            
            
            df_filt1 = df_filt1.loc[:, ~df_filt1.columns.str.contains('Unnamed')]
            cols = list(df_filt1.columns)
            cols = [cols[-1]] + cols[:-1]
            cols = [cols[-1]] + cols[:-1]
            cols = [cols[-1]] + cols[:-1]
            df_filt1 = df_filt1[cols]
            
            df_filt1.to_csv(path_saveData_points + "/" + "POI.csv")
            
        ax1.legend(loc='upper left') 
        if save_plot==True:
    
            if not os.path.exists(Fig_outputDir):
                os.mkdir(Fig_outputDir)
            
            path_Fig= plt.savefig(Fig_outputDir + "/" + basename + ".png" )
                
        #os.unlink("temp.csv")
         
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    
    
    #plt.show()
   
   
   

def MeanProducts_plot_ts(path_to_shapefile="", dem_path="" , out_folder="Figs_analysis", color_field="", Set_fig_MinMax=False, MinMaxRange=[-100,100],
                   opacity=0.5, cmap="jet" , point_size=1, cbar_label="mm/year" , batch_plot=False, plot_inverse_Vel=False ):
    
    """
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
    """
   ###########################
   
   #Inverse Velocity
    
    def inverse_Velocity(path_to_shapefile, point_size, cmap):
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from sklearn.linear_model import LinearRegression
        from datetime import datetime

        # Load the shapefile
        gdf = gpd.read_file(path_to_shapefile)

        # Identify columns that start with 'D'
        d_columns = [col for col in gdf.columns if col.startswith('D')]
        #d_columns = d_columns[1:]  # Adjust according to your specific needs

        # Calculate inverse velocity for each row of the 'D' columns
        inverse_velocity_d_columns = 1 / gdf[d_columns].replace(0, np.nan)  # Replace zeros with NaN

        # Helper function to convert date columns to ordinal
        def date_to_ordinal(date_str):
            return pd.to_datetime(date_str[1:], format='%Y%m%d').toordinal()

        # Convert 'D' column names to ordinals for regression
        date_ordinals = np.array([date_to_ordinal(date) for date in d_columns])

        # Initialize lists to store zero crossing points
        x_coords_regression = []
        y_coords_regression = []
        dates_regression = []

        for index, row in inverse_velocity_d_columns.iterrows():
            valid_indices = ~row.isna()
            valid_inverse_velocities = row[valid_indices].values
            valid_date_ordinals = date_ordinals[valid_indices]

            if len(valid_inverse_velocities) >= 2:  # Ensure enough data points for regression
                model = LinearRegression()
                model.fit(valid_date_ordinals.reshape(-1, 1), valid_inverse_velocities)

                slope = model.coef_[0]
                intercept = model.intercept_

                if slope != 0:  # Avoid division by zero
                    zero_crossing_ordinal = -intercept / slope
                    # Ensure zero_crossing_ordinal is within a valid range
                    if zero_crossing_ordinal >= 1:
                        zero_crossing_date = datetime.fromordinal(int(zero_crossing_ordinal))
                        if valid_date_ordinals.min() <= zero_crossing_ordinal <= valid_date_ordinals.max() + 3000:  # Check within range
                            point = gdf.at[index, 'geometry']  # Get the geometry object at the given index
                            x_coords_regression.append(point.x)  # Append the x coordinate
                            y_coords_regression.append(point.y)  # Append the y coordinate
                            dates_regression.append(zero_crossing_date)
                    else:
                        continue
                        #print(f"Invalid zero crossing ordinal {zero_crossing_ordinal} for index {index}")

        # Convert dates for plotting
        dates_regression_num = mdates.date2num(dates_regression)

        # Plotting the zero crossings from linear regression model
        fig = plt.figure(figsize=(20, 8))
        ax = fig.add_subplot(122)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('bottom', size='3%', pad=0.1)
        
        scatter_regression = ax.scatter(x_coords_regression, y_coords_regression, c=dates_regression_num, cmap=cmap, s=point_size)

        # Setup the colorbar with correct formatting
        cbar_regression = fig.colorbar(scatter_regression, ax=ax, orientation='horizontal', cax=cax, extend='both')
       
        cbar_regression.set_label('Date of Zero Crossing')
        
         # Rotate colorbar labels
        for label in cbar_regression.ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment('right')

        cbar_regression.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # ax.set_xlabel('X Coordinate')
        # ax.set_ylabel('Y Coordinate')
        ax.set_title('Zero Crossing Dates (Linear Regression Extrapolation)')
        plt.grid(True)
        
        

        return fig, ax

        # Example usage:
        # path_to_shapefile = 'path/to/your/shapefile.shp'
        # fig, ax = inverse_Velocity(path_to_shapefile)
        # plt.show()

        #########################################
   
   
   
    import xml.etree.ElementTree as ET
    
    
    def get_values_from_xml(filename):
        try:
            # Parse the XML file
            tree = ET.parse(filename)
            root = tree.getroot()

            # Extracting x and y coordinates
            coordinates = root.find('ReferencePoint/Coordinates')
            if coordinates is not None:
                # Parsing the coordinate string
                coordinates_text = coordinates.text if coordinates.text is not None else ""
                # Assuming the format is "(x, y)"
                coordinates_text = coordinates_text.strip("() ")
                x_str, y_str = coordinates_text.split(", ")
                x = float(x_str)
                y = float(y_str)
            else:
                x, y = None, None

            # Extracting VEL and VEL_STD
            veld_data = root.find('VELDATA')
            if veld_data is not None:
                vel = veld_data.find('VEL').text if veld_data.find('VEL') is not None else None
                vel_std = veld_data.find('VEL_STD').text if veld_data.find('VEL_STD') is not None else None
            else:
                vel, vel_std = None, None

            # Convert to float if not None
            x = float(x) if x is not None else None
            y = float(y) if y is not None else None
            vel = float(vel) if vel is not None else None
            vel_std = float(vel_std) if vel_std is not None else None

            return x, y, vel, vel_std

        except ET.ParseError as pe:
            print(f"XML Parse Error: {pe}")
            return None, None, None, None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None, None, None
        
    x_ref=None
    import xml.etree.ElementTree as ET
    xml_file_path=path_to_shapefile+'.xml'
    if os.path.exists(xml_file_path):
        if xml_file_path.endswith('.xml'):
            x_ref, y_ref, vel, vel_std = get_values_from_xml(xml_file_path)
            # tree = ET.parse(xml_file_path)
            # root = tree.getroot()

            # x_elem = root.find(".//ReferencePoint/x")
            # y_elem = root.find(".//ReferencePoint/y")

            # if x_elem is not None and y_elem is not None:
            #     x_ref = float(x_elem.text)
            #     y_ref = float(y_elem.text)
            # else:
            #     pass
   
   
   
   ###################
    
    # Import shapfilepath
    shapefile_path = os.path.join(path_to_shapefile)
    basename = os.path.basename(shapefile_path[:-4])
    # Open shapefile data with geopandas
    gdf = gpd.read_file(shapefile_path)
    gdf.crs
    # Define path to dem data
    #dem_path = 'dem.tif'

    with rio.open(dem_path) as src:
        elevation = src.read(1)
        xres, yres=src.res
        elevation = elevation.astype('float32')
        # Set masked values to np.nan
        elevation[elevation < 0.0] = np.nan
    # Create and plot the hillshade with earthpy
    hillshade = es.hillshade(elevation, azimuth=275, altitude=30)

    dem = rxr.open_rasterio(dem_path, masked=True)
    dem_plotting_extent = plotting_extent(dem[0], dem.rio.transform())

    # Getting the crs of the raster data
    dem_crs = es.crs_check(dem_path)
    

    # Transforming the shapefile to the dem data crs
    gdf = gdf.to_crs(dem_crs)
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    if batch_plot==False:
        min=gdf[color_field].min()
        max=gdf[color_field].max()
        import matplotlib.colors as mcolors
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        
        if min < 0 and max >0 :
            if Set_fig_MinMax==True:
                min_n=MinMaxRange[0]
                max_n=MinMaxRange[1]
                min=min_n
                max=max_n
                offset = mcolors.TwoSlopeNorm(vmin=min,
                            vcenter=0., vmax=max)  
            else:
                offset = mcolors.TwoSlopeNorm(vmin=min,
                            vcenter=0., vmax=max)
                
        else  : 
            if Set_fig_MinMax==True:
                min_n=0
                max_n=100
                min=MinMaxRange[0]
                max=MinMaxRange[1]
                offset=mcolors.Normalize(vmin=min, vmax=max)
            else:
                offset=mcolors.Normalize(vmin=min, vmax=max)
                
        
        
        if plot_inverse_Vel==True:
            fig, ax2 =inverse_Velocity(path_to_shapefile=path_to_shapefile, point_size=point_size , cmap=cmap)
            
            ax1=fig.add_subplot(121)
        else:
            fig = plt.figure(figsize=(7,7))
            ax1 = fig.add_subplot(111)
        
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('bottom', size='3%', pad=0.1)
        
        def has_xy_columns(gdf):
            return 'x' in gdf.columns and 'y' in gdf.columns
        
        if has_xy_columns(gdf):
            x=gdf.x
            y=gdf.y
        else:
            x=gdf.geometry.x
            y=gdf.geometry.y
        #fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10,5))
        ep.plot_bands( hillshade,cbar=False,title=color_field,extent=dem_plotting_extent,ax=ax1, scale=False)
        
        img_main=ax1.scatter(x, y, c=gdf[color_field], alpha=opacity, s=point_size, picker=1, cmap=cmap, norm=offset)
        #scalebar = ScaleBar(xres, "m", scale_loc="right",border_pad=1,pad=0.5, box_color='white', box_alpha=0.5, location='lower right', ax=ax1)
        #scalebar = ScaleBar(dx=xres, units='m', location='lower right',frameon=False, scale_loc='bottom')
        #ax1.add_artist(scalebar)
        ax1.grid(True)
        #ax.scatter(gdf.x, gdf.y, s= 0.5, c=gdf.VEL_MEAN ,picker=1)
        cb=fig.colorbar(img_main, ax=ax1, cax=cax, extend='both', orientation='horizontal')
        cb.set_label(cbar_label, labelpad=2, x=0.5, rotation=0)
         # Rotate colorbar labels
        for label in cb.ax.get_xticklabels():
            label.set_rotation(25)
            label.set_horizontalalignment('right')
        if os.path.exists(xml_file_path):
            if x_ref is not None:
                ax1.scatter(x_ref, y_ref, label=f"Reference VEL,VEL_STD: {vel:.2f},{vel_std:.2f}", color='k', marker='s')
                ax1.legend()
       
        
        if plot_inverse_Vel==True:
            #fig, ax2 =inverse_Velocity(path_to_shapefile)
            
            ep.plot_bands( hillshade,cbar=False,title=color_field,extent=dem_plotting_extent,ax=ax2, scale=False)
            #scalebar = ScaleBar(xres, "m", scale_loc="right",border_pad=1,pad=0.5, box_color='white', box_alpha=0.5, location='lower right', ax=ax2)
            #scalebar = ScaleBar(dx=xres, units='m', location='lower right',frameon=False, scale_loc='bottom')
            #ax2.add_artist(scalebar)
            ax2.set_title('Inverse Velocity')
            
            if os.path.exists(xml_file_path):
                if x_ref is not None:
                    ax2.scatter(x_ref, y_ref, label=f"Reference VEL,VEL_STD: {vel:.2f},{vel_std:.2f}", color='k', marker='s')
                    ax2.legend()
            
        
        plt.tight_layout()
        
        
        plt.savefig(out_folder+"/"+color_field+".png")
        
        plt.show()
            
            
            
    ##########################################
    
    if batch_plot==True:
        dnames = [col for col in gdf.columns if col.startswith('D')]
       
            
        for nd in gdf[dnames]:
            min=gdf[nd].min()
            max=gdf[nd].max()
            import matplotlib.colors as mcolors
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            
            
            if min < 0 and max >0 :
                if Set_fig_MinMax==True:
                    min_n=MinMaxRange[0]
                    max_n=MinMaxRange[1]
                    min=min_n
                    max=max_n
                    offset = mcolors.Normalize(vmin=min,
                                    vmax=max)  
                else:
                    
                    offset = mcolors.Normalize(vmin=min,
                                    vmax=max)
            elif min==0 and max==0:
                offset = mcolors.Normalize(vmin=0,
                                    vmax=1)
            elif min <0:
                if abs(min)>max:
                    
                    offset = mcolors.TwoSlopeNorm(vmin=min, vcenter=0,
                                    vmax=-min) 
                else:
                    offset = mcolors.TwoSlopeNorm(vmin=-max, vcenter=0,
                                    vmax=max) 
                            
            else  : 
                if Set_fig_MinMax==True:
                    min_n=0
                    max_n=100
                    min=MinMaxRange[0]
                    max=MinMaxRange[1]
                    offset=mcolors.Normalize(vmin=min, vmax=max)
                else:
                    offset=mcolors.Normalize(vmin=min, vmax=max)
            fig = plt.figure(figsize=(7,7))
            ax1 = fig.add_subplot(111)
            
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('bottom', size='5%', pad=0.05)
            def has_xy_columns(gdf):
                return 'x' in gdf.columns and 'y' in gdf.columns
            
            if has_xy_columns(gdf):
                x=gdf.x
                y=gdf.y
            else:
                x=gdf.geometry.x
                y=gdf.geometry.y
            #fig, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(10,5))
            ep.plot_bands( hillshade,cbar=False,title=nd,extent=dem_plotting_extent,ax=ax1, scale=False)
            img_main=ax1.scatter(x, y, c=gdf[nd], alpha=opacity, s=point_size, picker=1, cmap=cmap, norm=offset)
            #scalebar = ScaleBar(xres, "m", scale_loc="right",border_pad=1,pad=0.5, box_color='white', box_alpha=0.5, location='lower right', ax=ax1)
            #scalebar = ScaleBar(dx=xres, units='m', location='lower right',frameon=False, scale_loc='bottom')
            #ax1.add_artist(scalebar)
            plt.grid(True)
            #ax.scatter(gdf.x, gdf.y, s= 0.5, c=gdf.VEL_MEAN ,picker=1)
            cb=fig.colorbar(img_main, ax=ax1, cax=cax, extend='both', orientation='horizontal')
            cb.set_label(cbar_label, labelpad=2, x=0.5, rotation=0)
            
            if os.path.exists(xml_file_path):
                if x_ref is not None:
                    ax1.scatter(x_ref, y_ref, label="REF", color='b', marker='s')
                    ax1.legend()
            plt.tight_layout()
            
            plt.savefig(out_folder+"/"+nd+".png")
            
            if len(dnames)>10:
                print("akhdefo is plotting more than 10 figures to avoid crushing python kernel we skip displaying figures. \n Please see figures inside provided out_folder path")
                plt.close()
            else:
                
                plt.show()
                
            
            