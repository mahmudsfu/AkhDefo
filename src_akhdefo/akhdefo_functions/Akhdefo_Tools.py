
import os
import rasterio
import geopandas as gpd
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
from rasterio.warp import reproject

import numpy as np

def resample_raster(src_array, src_transform, src_crs, dst_transform, dst_crs, dst_shape, resampling_method=rasterio.enums.Resampling.nearest):
    """
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
    """
    
    # Create an empty array with the destination shape
    resampled_array = np.empty(dst_shape, np.float32)
    
    # Define the source and destination transformations and arrays for reproject
    reproject(
        source=src_array,
        destination=resampled_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling_method
    )

    return resampled_array



def scatter_area_mask(input_folder, output_folder, plot_folder, shapefile_path, scatter_Area_threshold=1.1, vegetation_mask_path=None):
    """
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
    """
    
    # Ensure the output and plot folders exist
    for folder in [output_folder, plot_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # Read the AOI from the shapefile
    aoi_gdf = gpd.read_file(shapefile_path)
    geometry = aoi_gdf.geometry[0]  # Assuming only one geometry in the shapefile

    accumulated_mask = None  # This will store the final aggregated mask

    # Process each raster
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):
            input_path = os.path.join(input_folder, filename)
            
            with rasterio.open(input_path) as src:
                # Crop raster based on AOI
                cropped_image, cropped_transform = mask(src, [geometry], crop=True)
                cropped_image = cropped_image[0]  # As it returns in (bands, row, col) format
                
                # Normalize the image
                min_val = np.min(cropped_image)
                max_val = np.max(cropped_image)
                
                print('min_initial: ', min_val)
                print('max_initial: ' , max_val)
                normalized_image = (cropped_image - min_val) / (max_val - min_val) *100
                
                 # Normalize the image
                min_val = np.min(normalized_image)
                max_val = np.max(normalized_image)
                
                print('min_normalized: ', min_val)
                print('max_normalized: ' , max_val)
                
                # Convert the normalized image to binary
                
                binary_image = np.where(normalized_image < scatter_Area_threshold, 0, 1).astype(rasterio.uint8)

                # Accumulate the mask
                if accumulated_mask is None:
                    accumulated_mask = binary_image
                else:
                    accumulated_mask *= binary_image

    # Save the final accumulated mask
    meta = src.meta.copy()
    meta.update({
        'dtype': rasterio.uint8,
        'height': accumulated_mask.shape[0],
        'width': accumulated_mask.shape[1],
        'transform': cropped_transform
    })

    if vegetation_mask_path is not None:
        with rasterio.open(vegetation_mask_path) as src:
            vegi_mask = src.read(1)
            vegi_mask = np.where(vegi_mask < 1, 0, 1).astype(rasterio.uint8)
            vegi_transform = src.transform
            vegi_crs = src.crs

        # Resample vegi_mask to match accumulated_mask
        resampled_vegi_mask = resample_raster(
            src_array=vegi_mask,
            src_transform=vegi_transform,
            src_crs=vegi_crs,
            dst_transform=cropped_transform,  # This should be the transform of accumulated_mask
            dst_crs=meta['crs'],  # This should be the CRS of accumulated_mask
            dst_shape=accumulated_mask.shape
        )

        combined_mask=resampled_vegi_mask+ accumulated_mask
        combined_mask = np.where(resampled_vegi_mask > 0, 1, 0).astype(rasterio.uint8)
        accumulated_mask = combined_mask
        
        # Plot and save the final accumulated mask
        plt.colorbar(plt.imshow(accumulated_mask, cmap='gray'))
        plt.axis('off')
        plt.savefig(os.path.join(plot_folder, "SAR_scatterArea_Vegetation_masks.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        
        
        with rasterio.open(os.path.join(output_folder, "SAR_scatterArea_Vegetation_masks.tif"), 'w', **meta) as dest:
            dest.write(accumulated_mask, 1)
    else:
        with rasterio.open(os.path.join(output_folder, "SAR_scatterArea_mask.tif"), 'w', **meta) as dest:
            dest.write(accumulated_mask, 1)

        # Plot and save the final accumulated mask
        plt.colorbar( plt.imshow(accumulated_mask, cmap='gray'))
        plt.axis('off')
        plt.savefig(os.path.join(plot_folder, "SAR_scatterArea_mask.png"), dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        


def utm_to_latlon(easting, northing, zone_number, zone_letter):
    '''
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

    '''
    import geopandas as gpd
    import utm
    easting = easting
    northing = northing
    lon, lat=utm.to_latlon(easting, northing, zone_number, zone_letter)
    
    return [lon, lat]

import os

import numpy as np
import rasterio
from osgeo import gdal, osr
from rasterio.transform import Affine


def flip_geotiff_180(directory):
    # List all files in the directory
    for filename in os.listdir(directory):
        # Only process files with the .tif extension
        if filename.endswith(".tif"):
            filepath = os.path.join(directory, filename)

            # Open the file
            with rasterio.open(filepath) as src:
                # Read the image data
                data = src.read()
                # Define the transform
                transform = src.transform

            # Flip the data array upside down (180 degree rotation)
            data = np.flipud(data)

            # Update the transform
            transform = Affine(transform.a, transform.b, transform.c, transform.d, -transform.e, src.height * transform.e + transform.f)

            # Write the data to the same file, overwriting the original
            with rasterio.open(filepath, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2], count=data.shape[0], dtype=data.dtype, crs=src.crs, transform=transform) as dst:
                dst.write(data)


def assign_fake_projection(input_dir, output_dir):
    '''
    Note
    ====

    This program assigns fake latlon geographic coordinates to ground-based images 
    so that it can be ingest using gdal and rasterio geospatial libraries for further processing
    
    input_dir: str
        path to image directories without projection info
    
    output_dir: str
        output path image directory for images included projection info

    
    '''
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List of valid extensions
    valid_extensions = ['.tif', '.jpg', '.png', '.bmp']

    # # Create a "fake" Spatial Reference object for source
    # source_srs = osr.SpatialReference()
    # source_srs.SetWellKnownGeogCS('LOCAL_CS')  # 'LOCAL_CS' is a placeholder coordinate system
    # Create a Spatial Reference object for source
    # source_srs = osr.SpatialReference()
    # source_srs.SetWellKnownGeogCS('LOCAL_CS')
    #source_srs.SetWellKnownGeogCS('WGS84')  # WGS84 is a commonly used geodetic coordinate system

    #######

    # Create a SpatialReference object
    source_srs = osr.SpatialReference()

    # Set the UTM Zone 10N coordinate system
    source_srs.SetUTM(10, 1)  # Zone 10, Northern Hemisphere


    ########
    from scipy import ndimage

    # Iterate over all files in the directory
    for filename in os.listdir(input_dir):
        # Check if the file has a valid extension
        if os.path.splitext(filename)[1] in valid_extensions:
            # Define the full path to the input raster
            input_raster_path = os.path.join(input_dir, filename)

            # Open the raster
            ds = gdal.Open(input_raster_path, gdal.GA_ReadOnly)

            # Read the raster data
            data = ds.ReadAsArray()

            # Rotate array by 45 degrees
            #data = ndimage.rotate(data, 180)

            # Define the full path to the output raster
            # We keep the original filename but put it into the output_dir
            output_raster_path = os.path.join(output_dir, filename[:-4]+".tif")

            # Create a new raster dataset with the same dimensions
            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(output_raster_path, ds.RasterXSize, ds.RasterYSize, ds.RasterCount, ds.GetRasterBand(1).DataType)

            # Assign the "fake" projection and the same geotransform
            out_ds.SetProjection(source_srs.ExportToWkt())
            out_ds.SetGeoTransform(ds.GetGeoTransform())

             # Assign the WGS84 projection and the same geotransform
            out_ds.SetProjection(source_srs.ExportToWkt())
            out_ds.SetGeoTransform(ds.GetGeoTransform())

            # Write the data to the new raster
            for i in range(ds.RasterCount):
                out_band = out_ds.GetRasterBand(i+1)
                out_band.WriteArray(data[i])

            # Close the datasets
            ds = None
            out_ds = None
    for filename in os.listdir(output_dir):
        # Only process files with the .tif extension
        if filename.endswith(".tif"):
            filepath = os.path.join(output_dir, filename)

            # Open the file
            with rasterio.open(filepath) as src:
                # Read the image data
                data = src.read()
                # Define the transform
                transform = src.transform

            # Flip the data array upside down (180 degree rotation)
            data = np.flipud(data)

            # Update the transform
            transform = Affine(transform.a, transform.b, transform.c, transform.d, -transform.e, src.height * transform.e + transform.f)

            # Write the data to the same file, overwriting the original
            with rasterio.open(filepath, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2], count=data.shape[0], dtype=data.dtype, crs=src.crs, transform=transform) as dst:
                dst.write(data)  
import os
import re
import shutil


def move_files(base_directory):
    """
    This function reorganizes files in the specified directory. 
    It searches for timestamps in filenames, creates subdirectories based on the hour part of the timestamp,
    and moves files to the appropriate subdirectories. The files are renamed based on the year, month, and day of the timestamp.
    
    Args:
        base_directory (str): Path of the directory containing the files to be reorganized.

    """

    # List of regex patterns for different timestamp formats
    timestamp_patterns = [
        r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\.',  # yyyymmddhhmmss
        r'(?P<year>\d{2})(?P<month>\d{2})(?P<day>\d{2})(?P<hour>\d{2})\.',  # yymmddhh
        r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})\.',  # yyyymmdd
        r'(?P<hour>\d{2})(?P<minute>\d{2})(?P<second>\d{2})\.',  # hhmmss
        r'(?P<hour>\d{2})\.'  # hh
        # Add more patterns as necessary
    ]

    for filename in os.listdir(base_directory):
        # If the file is not a file, skip
        if not os.path.isfile(os.path.join(base_directory, filename)):
            continue

        # Extract the timestamp from the filename
        for pattern in timestamp_patterns:
            match = re.search(pattern, filename)
            if match:
                year = match.groupdict().get('year', '0000')
                month = match.groupdict().get('month', '00')
                day = match.groupdict().get('day', '00')
                hour = match.group('hour')
                break
        else:
            print(f"No timestamp found in file {filename}.")
            continue

        # Construct new filename based on date and existing extension
        base, extension = os.path.splitext(filename)
        new_filename = f"{year}-{month}-{day}{extension}"

        # Make directory for this hour if it doesn't exist
        hour_dir = os.path.join(base_directory, hour)
        if not os.path.exists(hour_dir):
            os.makedirs(hour_dir)

        # Move and rename file to the corresponding hour folder
        shutil.move(os.path.join(base_directory, filename), os.path.join(hour_dir, new_filename))




import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import Affine


def calculate_volume(elevation_map, slope_map, cell_size, output_file, plot_map=False, plot_file=None):
    """
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

    """
    # Read elevation map raster using rasterio
    src= rasterio.open(elevation_map)
    # Get CRS from elevation_map
    crs = src.crs

    # Get transform from elevation_map
    transform = src.transform

     # Read elevation map data
    elevation_data = src.read(1)

     # Read elevation map raster using rasterio
    src1= rasterio.open(slope_map)
        
     # Read slope_map data
    slope_data = src.read(1)

    # Calculate the dimensions of the maps
    rows, cols = elevation_data.shape

    # Initialize the volume map
    volume_map = np.zeros_like(elevation_data, dtype=float)

    # Iterate over each cell in the maps
    for i in range(rows):
        for j in range(cols):
            # Calculate the cell area
            area = cell_size ** 2

            # Calculate the cell volume contribution
            volume_map[i, j] += elevation_data[i, j] * area

            # Calculate the slope gradient
            slope_gradient = np.tan(np.radians(slope_data[i, j]))

            # Calculate the additional volume due to the slope
            volume_map[i, j] += 0.5 * slope_gradient * area

    # Save volume map as GeoTIFF
    with rasterio.open(output_file, "w", driver="GTiff", height=volume_map.shape[0], width=volume_map.shape[1], count=1, dtype=volume_map.dtype, crs=crs, transform=transform) as dst:
        dst.write(volume_map, 1)

    # Plot and save volume map if desired
    if plot_map:
        plt.imshow(volume_map, cmap='viridis')
        plt.colorbar(label='Volume')
        plt.title('Volume Map')
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.show()

    return volume_map




# def ts_plot(df, plot_number, save_plot=False , output_dir="", plot_filename="" , VEL_Scale='year'):


#     import plotly.graph_objects as go
#     import plotly.express as px
#     import plotly.express as px_temp
#     import pandas as pd
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import geopandas as gpd 
#     import pandas as pd  
#     import seaborn as sns  
#     import plotly.offline as py_offline
#     import os   
#     import statsmodels.api as sm
#     from sklearn.metrics import mean_squared_error, r2_score
#     import numpy as np
#     from sklearn.linear_model import LinearRegression
#     from datetime import datetime
#     import math
    
#     py_offline.init_notebook_mode()
#     #%matplotlib widget
#     #df=pd.read_csv("temp.csv")
#     df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
#     df['dd_str']=df['dd'].astype(str)
#     df['dd_str'] = df['dd_str'].astype(str)
#     df.rename(columns={ df.columns[1]: "val" }, inplace = True)
#     df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y%m%d')
    
#     df=df.set_index('dd')
    
#     ########################
#     df=df.dropna()
#     # Make index pd.DatetimeIndex
#     df.index = pd.DatetimeIndex(df.index)
#     # Make new index
#     idx = pd.date_range(df.index.min(), df.index.max())
#     # Replace original index with idx
#     df = df.reindex(index = idx)
#     # Insert row count
#     df.insert(df.shape[1],
#             'row_count',
#             df.index.value_counts().sort_index().cumsum())

#     df=df.dropna()
    
#     #df=df.set_index(df['row_count'], inplace=True)

#     df.sort_index(ascending=True, inplace=True)
    

#     def best_fit_slope_and_intercept(xs,ys):
#         from statistics import mean
#         xs = np.array(xs, dtype=np.float64)
#         ys = np.array(ys, dtype=np.float64)
#         m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
#             ((mean(xs)*mean(xs)) - mean(xs*xs)))
        
#         b = mean(ys) - m*mean(xs)
        
#         return m, b

    

#     #convert dattime to number of days per year
    
    
    

#     dates_list=([datetime.strptime(x, '%Y%m%d') for x in df.dd_str])
#     days_num=[( ((x) - (pd.Timestamp(year=x.year, month=1, day=1))).days + 1) for x in dates_list]
#     time2=days_num[len(days_num)-1]
#     time1=days_num[0]
#     delta=time2-time1
#     delta=float(delta)
#     print(days_num, delta)
    
#     m, b = best_fit_slope_and_intercept(df.row_count, df.val)
#     print("m:", math.ceil(m*100)/100, "b:",math.ceil(b*100)/100)
#     regression_model = LinearRegression()
#     val_dates_res = regression_model.fit(np.array(days_num).reshape(-1,1), np.array(df.val))
#     y_predicted = regression_model.predict(np.array(days_num).reshape(-1,1))
    
#     if VEL_Scale=='year':
#         rate_change=regression_model.coef_[0]/delta * 365.0
#     elif VEL_Scale=='month':
#         rate_change=regression_model.coef_[0]/delta * 30
        
#     # model evaluation
#     mse=mean_squared_error(np.array(df.val),y_predicted)
#     rmse = np.sqrt(mean_squared_error(np.array(df.val), y_predicted))
#     r2 = r2_score(np.array(df.val), y_predicted)
    
#     # printing values
#     print('Slope(linear deformation rate):' + str(math.ceil(regression_model.coef_[0]*100)/100/delta) + " mm/day")
#     print('Intercept:', math.ceil(b*100)/100)
#     #print('MSE:',mse)
#     print('Root mean squared error: ', math.ceil(rmse*100)/100)
#     print('R2 score: ', r2)
#     print("STD: ",math.ceil(np.std(y_predicted)*100)/100) 
#     # Create figure
#     #fig = go.Figure()
    
#     fig = go.FigureWidget()
    
#     plot_number="Plot Number:"+str(plot_number)

#     fig.add_trace(go.Scatter(x=list(df.index), y=list(df.val)))
#     fig = px.scatter(df, x=list(df.index), y=list(df.val),
#                 color="val", hover_name="val"
#                     , labels=dict(x="Dates", y="mm/"+VEL_Scale , color="mm/"+VEL_Scale))
    
#     # fig.add_trace(
#     # go.Scatter(x=list(df.index), y=list(val_fit), mode = "lines",name="trendline", marker_color = "red"))
    
    
    
#     fig.add_trace(go.Scatter(x=list(df.index), y=list(df.val),mode = 'lines',
#                             name = 'draw lines', line = dict(shape = 'linear', color = 'rgb(0, 0, 0)', dash = 'dash'), connectgaps = True))
    
#     fig.add_trace(
#         go.Scatter(x=list(df.index), y=list(y_predicted), mode = "lines",name="trendline", marker_color = "black", line_color='red'))
    
    

#     # Add range slider
#     fig.update_layout(
#         xaxis=dict(
#             rangeselector=dict(
#                 buttons=list([
#                     dict(count=1,
#                         label="1m",
#                         step="month",
#                         stepmode="backward"),
#                     dict(count=6,
#                         label="6m",
#                         step="month",
#                         stepmode="backward"),
#                     dict(count=1,
#                         label="YTD",
#                         step="year",
#                         stepmode="todate"),
#                     dict(count=1,
#                         label="1y",
#                         step="year",
#                         stepmode="backward"),
#                     dict(step="all")
#                 ])
#             ),
#             rangeslider=dict(
#                 visible=True
#             ),
#             type="date"
#         ) 
#     )
#     fig.update_xaxes(rangeslider_thickness = 0.05)
#     #fig.update_layout(showlegend=True)

#     #fig.data[0].update(line_color='black')
#     tt= "Defo-Rate:"+str(round(rate_change,2))+":"+ "Defo-Rate-STD:"+str(round(np.std(y_predicted), 2))+ ":" +plot_number
    
#     # make space for explanation / annotation
#     fig.update_layout(margin=dict(l=20, r=20, t=20, b=60),paper_bgcolor="LightSteelBlue")

    
#     fig.update_layout(
        
#     title_text=tt, title_font_family="Sitka Small",
#     title_font_color="red", title_x=0.5 , legend_title="Legend",
#     font=dict(
#         family="Courier New, monospace",
#         size=15,
#         color="RebeccaPurple" ))
    
#     fig.update_layout(legend=dict(
#     yanchor="top",
#     y=-0,
#     xanchor="left",
#     x=1.01
# ))

#     # fig.update_layout(
#     # updatemenus=[
#     #     dict(
#     #         type="buttons",
#     #         direction="right",
#     #         active=0,
#     #         x=0.57,
#     #         y=1.2,
#     #         buttons=list([
#     #             dict(
#     #                 args=["colorscale", "Viridis"],
#     #                 label="Viridis",
#     #                 method="restyle"
#     #             ),
#     #             dict(
#     #                 args=["colorscale", "turbo"],
#     #                 label="turbo",
#     #                 method="restyle"
#     #             )
#     #         ]),
#     #     )
#     # ])

    
#     fig.update_xaxes(showspikes=True, spikemode='toaxis' , spikesnap='cursor', spikedash='dot', spikecolor='blue', scaleanchor='y', title_font_family="Arial", 
#                     title_font=dict(size=15))
#     fig.update_yaxes(showspikes=True, spikemode='toaxis' , spikesnap='cursor', spikedash='dot', spikecolor='blue', scaleanchor='x', title_font_family="Arial",
#                     title_font=dict(size=15))

    
    
#     if save_plot==True:
    
#         if not os.path.exists(output_dir):
#             os.mkdir(output_dir)

#         fig.write_html(output_dir + "/" + plot_filename + ".html" )
#         fig.write_image(output_dir + "/" + plot_filename + ".jpeg", scale=1, width=1080, height=300 )
        
    
#     def zoom(layout, xrange):
#         in_view = df.loc[fig.layout.xaxis.range[0]:fig.layout.xaxis.range[1]]
#         fig.layout.yaxis.range = [in_view.High.min() - 10, in_view.High.max() + 10]

#     fig.layout.on_change(zoom, 'xaxis.range')
    
#     fig.show()
    
    




    
#     start=int(start.timestamp() * 1000)
#     end=int(end.timestamp() * 1000)

#     #df=pd.read_csv('temp2.csv')
    
#     df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
#     df['dd_str']=df['dd'].astype(str)
#     df['dd_str'] = df['dd_str'].astype(str)
#     df.rename(columns={ df.columns[1]: "val" }, inplace = True)
#     df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y-%m-%d')
#     df.insert(df.shape[1],
#             'row_count',
#             df.index.value_counts().sort_index().cumsum())
#     #df=df.set_index('dd')
#     #df.index = pd.DatetimeIndex(df.index)
#     df.dd_str = pd.DatetimeIndex(df.dd_str)
#     df['dd_int'] = [int(i.timestamp()*1000) for i in df.dd_str]
#     import numpy as np 
#     def find_nearest(array, value):
#         array = np.asarray(array)
#         idx = (np.abs(array - value)).argmin()
#         return array[idx]
#     s=find_nearest(np.array(df.dd_int), start)
#     e=find_nearest(np.array(df.dd_int), end)

#     s=(df[df['dd_int']==s].index)
#     e=(df[df['dd_int']==e].index)

#     df_filter=df[s[0]:e[0]]
#     print(df_filter)

#     df=df_filter  
    
# import pandas as pd
# import ipywidgets as widgets
# from IPython.display import display

# class DateRangePicker(object):
#     def __init__(self,start,end,freq='D',fmt='%Y-%m-%d'):
#         """
#         Parameters
#         ----------
#         start : string or datetime-like
#             Left bound of the period
#         end : string or datetime-like
#             Left bound of the period
#         freq : string or pandas.DateOffset, default='D'
#             Frequency strings can have multiples, e.g. '5H' 
#         fmt : string, defauly = '%Y-%m-%d'
#             Format to use to display the selected period

#         """
#         self.date_range=pd.date_range(start=start,end=end,freq=freq)
#         options = [(item.strftime(fmt),item) for item in self.date_range]
#         self.slider_start = widgets.SelectionSlider(
#             description='start',
#             options=options,
#             continuous_update=False
#         )
#         self.slider_end = widgets.SelectionSlider(
#             description='end',
#             options=options,
#             continuous_update=False,
#             value=options[-1][1]
#         )

#         self.slider_start.on_trait_change(self.slider_start_changed, 'value')
#         self.slider_end.on_trait_change(self.slider_end_changed, 'value')

#         self.widget = widgets.Box(children=[self.slider_start,self.slider_end])

#     def slider_start_changed(self,key,value):
#         self.slider_end.value=max(self.slider_start.value,self.slider_end.value)
#         self._observe(start=self.slider_start.value,end=self.slider_end.value)

#     def slider_end_changed(self,key,value):
#         self.slider_start.value=min(self.slider_start.value,self.slider_end.value)
#         self._observe(start=self.slider_start.value,end=self.slider_end.value)

#     def display(self):
#         display(self.slider_start,self.slider_end)

#     def _observe(self,**kwargs):
#         if hasattr(self,'observe'):
#             self.observe(**kwargs)

# def fct(start,end):
#     print (start,end)
    
#     start=int(start.timestamp() * 1000)
#     end=int(end.timestamp() * 1000)

#     df=pd.read_csv('temp2.csv')

#     df.rename(columns={ df.columns[0]: "dd" }, inplace = True)
#     df['dd_str']=df['dd'].astype(str)
#     df['dd_str'] = df['dd_str'].astype(str)
#     df.rename(columns={ df.columns[1]: "val" }, inplace = True)
#     df['dd']= pd.to_datetime(df['dd'].astype(str), format='%Y-%m-%d')
#     df.insert(df.shape[1],
#             'row_count',
#             df.index.value_counts().sort_index().cumsum())
#     #df=df.set_index('dd')
#     #df.index = pd.DatetimeIndex(df.index)
#     df.dd_str = pd.DatetimeIndex(df.dd_str)
#     df['dd_int'] = [int(i.timestamp()*1000) for i in df.dd_str]
#     import numpy as np 
#     def find_nearest(array, value):
#         array = np.asarray(array)
#         idx = (np.abs(array - value)).argmin()
#         return array[idx]
#     s=find_nearest(np.array(df.dd_int), start)
#     e=find_nearest(np.array(df.dd_int), end)

#     s=(df[df['dd_int']==s].index)
#     e=(df[df['dd_int']==e].index)

#     df_filter=df[s[0]:e[0]]
#     print(df_filter)
#     return (start, end)
    
# w=DateRangePicker(start='2022-08-02',end="2022-09-02",freq='D',fmt='%Y-%m-%d')
# w.observe=fct
# w.display()

# #a=fct[0]
# print(w.observe[0])

############################################################


def akhdefo_fitPlane(dem_data='', line_shapefile=None , out_planeFolder='Planes_out'):
    """
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
        
    """
    
    global dip_angle_list, dip_direction_list , fig1_cmap, plane_colors
    
    dip_angle_list=[]
    dip_direction_list=[]
    plane_colors=[]
    
    # def cmap_plot(plane_colors,  label='Color'):
        
    #     from matplotlib.colors import ListedColormap
    #     import numpy as np
    #     try:
    #         cmap = ListedColormap(plane_colors)
    #         fig, ax = plt.subplots(figsize=(6, 1))
    #         fig.subplots_adjust(bottom=0.5)
    #         cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=ax, orientation='horizontal')
    #         cbar.set_label(label)
    #         plt.show()
    #     except Exception as ex:
    #         print("")
        
    #     return fig
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from osgeo import gdal
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.colors import LightSource
    import earthpy.spatial as es
    import geopandas as gpd
    import random
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
    from shapely.geometry import LineString
    
    
    if not os.path.exists(out_planeFolder):
        os.makedirs(out_planeFolder)
    
    def color_from_dip_and_direction(dip, direction):
        cmap_dip = plt.get_cmap('hsv')
        cmap_direction = plt.get_cmap('hsv')

        col_dip = cmap_dip(dip / 90.0)
        col_direction = cmap_direction(direction / 360.0)

        color = [0.5*(col_dip[i] + col_direction[i]) for i in range(3)]
    
        return color

    def random_color():
        """Generate a random color."""
        return (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

    def read_dem(dem_file):
        # Open the DEM file using GDAL
        ds = gdal.Open(dem_file)
        if ds is None:
            raise FileNotFoundError(f"File not found: {dem_file}")

        # Access the first raster band from the dataset
        band = ds.GetRasterBand(1)

        # Retrieve the no data value from the band
        no_data_value = band.GetNoDataValue()

        # Read the data into a NumPy array
        data = band.ReadAsArray()

        # Check if there is a no data value defined and replace it with NaN
        if no_data_value is not None:
            data[data == no_data_value] = np.nan

        # Get geotransformation and projection information
        transform = ds.GetGeoTransform()
        projection = ds.GetProjection()

        # Clean up by closing the dataset
        ds = None
        return data, transform, projection
    
    data, transform, projection = read_dem(dem_data)
    
    import math

    def calculate_dip_angle_and_direction(A, B):
        # Calculate the dip angle (inclination)
        dip_angle = math.degrees(math.atan(math.sqrt(A**2 + B**2)))
       
        
        # Calculate the dip direction (azimuth)
        dip_direction = math.degrees(math.atan2(-B, -A))
        
        # Ensure dip_direction is in the range [0, 360)
        dip_direction = (dip_direction + 90) % 360
        
        return dip_angle, dip_direction
    
    def fit_plane(points, xx, yy):
        min_range = np.nanmin(data)
        max_range = np.nanmax(data)
        
        # xx = (xx - np.nanmin(xx)) / (np.nanmax(xx) - np.nanmin(xx))
        # xx = xx * (max_range - min_range) + min_range
        
        # yy = (yy - np.nanmin(yy)) / (np.nanmax(yy) - np.nanmin(yy))
        # yy = yy * (max_range - min_range) + min_range
        
        x_vals, y_vals, z_vals = zip(*points)
    
        A = np.vstack([x_vals, y_vals, np.ones(len(x_vals))]).T
        a, b, c = np.linalg.lstsq(A, z_vals, rcond=None)[0]
        
        print(f"Plane equation: z = {a}x + {b}y + {c}")  # Debugging line
        
        zz = a * xx + b * yy + c
        ##################################
        # points = np.array(points) 
        # # Compute the centroid of the points
        # centroid = np.mean(points, axis=0)

        # # Subtract the centroid from the points to center them
        # centered_points = points - centroid

        # # Perform SVD on the centered points
        # U, S, Vt = np.linalg.svd(centered_points)

        # # The normal vector of the plane is the last row of Vt
        # normal_vector = Vt[-1, :]

        # # Normalize the normal vector
        # normal_vector /= np.linalg.norm(normal_vector)

        # # The equation of the plane is: ax + by + cz + d = 0, where [a, b, c] is the normal vector
        # a, b, c = normal_vector

        # # Calculate d using the plane equation and the centroid
        # d = -np.dot(normal_vector, centroid)
        # zz = (-a * xx - b * yy - d) / c  # Solve for z
        
       
        dip_angle, dip_direction = calculate_dip_angle_and_direction(a, b)

        print(f"Dip Angle (Inclination): {dip_angle} degrees")
        print(f"Dip Direction (Azimuth): {dip_direction} degrees")
        
        
        # def normalize_array(arr, new_min, new_max):
        #     min_val = np.min(arr)
        #     max_val = np.max(arr)
        #     normalized = new_min + (arr - min_val) * (new_max - new_min) / (max_val - min_val)
        #     return normalized
        # zz = normalize_array(zz, min_range, max_range)
        
        #  # Clip the values to the desired range
        #zz = np.clip(zz, min_range, max_range)
        
        zz[zz < min_range] = np.nan
        zz[zz > max_range] = np.nan
       
       
        # # Define the desired range based on your 'data'
        # if limit_extend==False:
           
        # # # # Normalize 'zz' to match the 'min_range' and 'max_range' of 'data'
        # zz = (zz - np.nanmin(zz)) / (np.nanmax(zz) - np.nanmin(zz))
        # zz = zz * (max_range - min_range) + min_range
        
               
        # print(zz.shape)
        # print(zz)
        return zz, a, b, c, dip_angle, dip_direction
    
   
   
   

    def onclick(event, points, data, transform):
        x, y = event.xdata, event.ydata
        col = int((x - transform[0]) / transform[1])
        row = int((y - transform[3]) / transform[5])

        if 0 <= col < data.shape[1] and 0 <= row < data.shape[0]:
            z = data[row, col]
            points.append([x, y, z])
            ax_2d.plot(x, y, 'o', color='black')
            canvas_2d.draw()

            point_str = f"x={x:.2f}, y={y:.2f}, z={z:.2f}"
            points_listbox.insert(tk.END, point_str)

    def on_point_double_click(event):
        global points, ax_2d, points_listbox, hs
        idx = points_listbox.curselection()[0]  # Index of selected point in the listbox
        point = points[idx]
        
        # Remove the point from our data
        points.pop(idx)
        points_listbox.delete(idx)

        # Identify and remove the corresponding point from the ax_2d plot
        for line in ax_2d.lines:
            xdata, ydata = line.get_xdata(), line.get_ydata()
            if xdata[0] == point[0] and ydata[0] == point[1]:
                line.remove()
                break

        canvas_2d.draw()
        
        
    
    
    def plot_planes(dip_list, azimuth_list, rgb_colors, out_planeFolder=out_planeFolder):
        #from mplstereonet import StereonetAxes
         # Create a stereo net plot
        fig = plt.figure(figsize=(10, 10), dpi=300)
        #ax = fig.add_subplot(121, projection='stereonet')
        ax=fig.add_subplot(121, polar=True)

        for dip_angles, dip_directions, color in zip(dip_list, azimuth_list, rgb_colors):
            # Convert dip directions and dip angles to radians
            #dip_directions = (dip_directions + 90) % 360 #converted dip direction to strike
            dip_angles = (90 - np.array(dip_angles))  # Convert dip angles to complementary angles
             # Convert dip azimuths to radians
            dip_azimuths_rad = np.radians(dip_directions)
            # Plot the poles of planes as dots
            ax.scatter(dip_azimuths_rad, dip_angles, color=color)   

        ax.grid(True)
        ax.set_theta_zero_location('N')  # Set 0 degrees at the top
        ax.set_theta_direction(-1)  # Reverse the direction of the angles

        # Set labels for cardinal directions
        ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])

        # Set the radial grid and labels
        ax.set_rlabel_position(90)
        ax.set_yticks([15, 30, 45, 60, 75])
        ax.set_rmax(90)  # Set the maximum radial distance
        ax.set_title("Poles to Planes")
        # Customize tick marks - specify angles in radians
        # ax.set_xticks(np.radians(range(0, 360, 10)))  # 10-degree intervals
        # ax.set_yticks(np.radians(range(-90, 90, 10)))

        strikes=[]
        all_colors=[]
        for dip_angles, dip_directions, color in zip(dip_list, azimuth_list, rgb_colors):
            dip_azimuths_rad = np.radians(dip_directions)
            #dip_directions = (dip_directions + 90) % 360
            dip_angles = (90 - np.array(dip_angles))  # Convert dip angles to complementary angles
            strikes.append((dip_azimuths_rad + 90) % 360)
            all_colors.append(color)
        

         # Calculate the number of bins using Scott's Rule
        n = len(strikes)

        # Calculate the standard deviation of the data
        std_dev = np.nanstd(strikes)
        # Calculate max and min, ignoring NaN values
        max_strike = np.nanmax(strikes)
        min_strike = np.nanmin(strikes)
        # Calculate the bin width using Scott's Rule formula, ensuring it's not zero
        if std_dev != 0:
            bin_width = 0.5 * std_dev / (n**(1/3))  # Scott's Rule formula
            #num_bins = max(int((max(strikes) - min(strikes)) / bin_width), 1)  # Ensure num_bins is at least 1
            num_bins = max(int((max_strike - min_strike) / bin_width), 1)  # Ensure num_bins is at least 1
        else:
            num_bins = 12  # A default number of bins if std_dev is zero
        # Create a rose diagram (circular histogram)
        ax_rose = fig.add_subplot(122, polar=True)

        for dip, strike, color, az in zip(dip_list, strikes, all_colors, azimuth_list):
            # Convert dip azimuths to radians
            dip_azimuths_rad = np.radians(az)
            dip_angles = (90 - np.array(dip_angles))  # Convert dip angles to complementary angles
            
            # Plot dip angles as radii and dip azimuths as angles
            #ax_rose.scatter(dip_azimuths_rad, dip_angles, c=color, alpha=0.5)
            # Plot the orientation data as a rose diagram
            n, bins, patches=ax_rose.hist(strike, bins=num_bins, color=color, alpha=0.5)
             # Annotate the number of data points and bin number
            ax_rose.text(0, -0.1, f"Data Points: {len(strikes)}", ha='center', va='center', transform=ax_rose.transAxes)
            ax_rose.text(0, -0.15, f"Bin Count: {num_bins}", ha='center', va='center', transform=ax_rose.transAxes)

        # # Customize the rose diagram
        # ax_rose.set_theta_zero_location("N")
        # ax_rose.set_theta_direction(-1)
        # Customize the polar plot
        ax_rose.set_theta_zero_location('N')  # Set 0 degrees at the top
        ax_rose.set_theta_direction(-1)  # Reverse the direction of the angles

        # Set labels for cardinal directions
        ax_rose.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
        ax_rose.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
       

        # Set the radial grid and labels
        # ax_rose.set_rlabel_position(90)
        # ax_rose.set_yticks([15, 30, 45, 60, 75])
        # ax_rose.set_rmax(90)  # Set the maximum radial distance
        ax_rose.set_title("Rose Diagram Strike")

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.2)

        plt.savefig(out_planeFolder + "/" + "stereoplot.png", dpi=300)
        plt.tight_layout
        #plt.close()
        plt.close()
        
    
    
        
    def draw_plane(xx,yy):
        
        if len(points) >= 3:
            fitted_plane, a, b, c , dip_angle, dip_direction = fit_plane(points, xx, yy)
            
            dip_angle_list.append(dip_angle)
            dip_direction_list.append(dip_direction)
            
            # if limit_extend:
            #  #Get bounding box of the selected points
            #     xmin, xmax, ymin, ymax = bounding_box(points)
                
            #     mask = (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
                
            #     for i in range(xx.shape[0]):
            #         for j in range(xx.shape[1]):
            #             if not mask[i, j]:
            #                 fitted_plane[i, j] = np.nan  # Set out-of-bounds values to NaN
        
            planes.append(fitted_plane)
            
            #plane_color = random_color()
            plane_color=color_from_dip_and_direction(dip_angle, dip_direction)
            print('plane color: ', plane_color)
            plane_colors.append(plane_color)

            for point  in points:
                ax_2d.plot(point[0], point[1], 'o', color=plane_color , alpha=0.7)
          
            
        
            canvas_2d.draw()
            from matplotlib.colors import LightSource
            ls = LightSource(270, 45)
            # To use a custom hillshading mode, override the built-in shading and pass
            # in the rgb colors of the shaded surface calculated from "shade".
            rgb = ls.shade(data, cmap=plt.cm.gray, vert_exag=1, blend_mode='overlay')

            #ax_3d.clear()
            ax_3d.set_title('3D DEM with Fitted Planes')
            ax_3d.plot_surface(xx, yy, data,cmap='gray',
                       linewidth=0, antialiased=True, shade=False, alpha=0.4, facecolors=rgb)
            #for plane in planes:
            surf=ax_3d.plot_surface(xx, yy, fitted_plane, alpha=0.6, linewidth=0, antialiased=True, color=plane_color)
            
                #plot points in 3D
            for point in points:
                ax_3d.plot(point[0], point[1], 'o', color=plane_color)
                
            canvas_3d.draw()
            
            
            

            points_listbox.delete(0, tk.END)
            points.clear()
            
            plot_planes(dip_angle_list, dip_direction_list, plane_colors)
            
            # Assuming you have a 'planes' list, you can check its format
            # for plane in planes:
            #     print(type(plane))
            #     print(len(plane))  # This will give you the length or number of elements in the plane

            #     # If it's a NumPy array, you can print its shape to see the dimensions
            #     if isinstance(plane, np.ndarray):
            #         print(plane.shape)
                
            #     # Print the first element (assuming it's a list or NumPy array)
            #     print(plane[0])

            # # Print the length of the 'planes' list
            # print(len(planes))
            
            
            
        

    def bounding_box(points):
        """Find the bounding box for a set of points."""
        x_vals, y_vals, _ = zip(*points)
        
        
        return min(x_vals) , max(x_vals), min(y_vals), max(y_vals)

    ########

   
    
    import ezdxf
    from scipy.spatial import Delaunay
    import numpy as np
    import os

    def list_xyz_files_in_folder(folder_path):
        xyz_files = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".xyz"):
                xyz_files.append(os.path.join(folder_path, filename))
        return xyz_files

    def create_dxf_from_xyz_files(xyz_files, output_path, single=None):
        #import open3d as o3d
        import pymeshlab

        for xyz_file in xyz_files:
            
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(xyz_file)
            ms.compute_normal_for_point_clouds()
            #ms.generate_surface_reconstruction_ball_pivoting()
            ms.generate_surface_reconstruction_screened_poisson()
            ms.meshing_remove_unreferenced_vertices()
            ms_1=ms
            #ms.save_current_mesh(output_path[:-4]+'.obj')
            ms_1.save_current_mesh(output_path[:-4]+'.ply', binary=True)
            # pcd = o3d.io.read_point_cloud(xyz_file)
            # pcd.estimate_normals()

            # # to obtain a consistent normal orientation
            # pcd.orient_normals_towards_camera_location(pcd.get_center())

            # # or you might want to flip the normals to make them point outward, not mandatory
            # pcd.normals = o3d.utility.Vector3dVector( - np.asarray(pcd.normals))

            # # surface reconstruction using Poisson reconstruction
            # mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

            # # paint uniform color to better visualize, not mandatory
            # mesh.paint_uniform_color(np.array([0.7, 0.7, 0.7]))

            # o3d.io.write_triangle_mesh(output_path[:-4]+'.ply', mesh)
        # doc = None
        # msp = None

        # # Function to initialize DXF document
        # def initialize_dxf():
        #     nonlocal doc, msp
        #     doc = ezdxf.new('R2010')  # Specify the DXF format, e.g., 'R2010' for compatibility
        #     msp = doc.modelspace()

        # if single is None:
        #     initialize_dxf()

        # for xyz_file in xyz_files:
        #     if single is not None:
        #         initialize_dxf()

        #     data = np.loadtxt(xyz_file)
        #     if data.size == 0 or data.shape[1] < 3:
        #         print(f"Skipping {xyz_file} due to insufficient data")
        #         continue
        #     if data.ndim == 1:
        #         data = data[np.newaxis, :]

        #     x, y, z = data[:, 0], data[:, 1], data[:, 2]
        #     points = np.column_stack((x, y, z))

        #     try:
        #         tri = Delaunay(points, furthest_site=False, incremental=True, qhull_options='Qc')
        #         for simplex in tri.simplices:
        #             vertices = [tuple(points[i]) for i in simplex]  # Ensure vertices are tuples
        #             msp.add_3dface(vertices)
        #     except Exception as e:
        #         print(f"Failed to process {xyz_file}: {e}")
        #         continue

        #     if single is not None:
        #         filename = os.path.join(output_path, f"{os.path.splitext(os.path.basename(xyz_file))[0]}.dxf")
        #         try:
        #             doc.saveas(filename)
        #             print(f"Saved: {filename}")
        #         except Exception as e:
        #             print(f"Error saving {filename}: {e}")

        # if single is None and doc is not None:
        #     try:
        #         doc.saveas(output_path)
        #         print(f"Saved combined DXF file: {output_path}")
        #     except Exception as e:
        #         print(f"Error saving combined DXF file: {e}")
        
    
    def save_planes_to_obj():
        x_coords = np.linspace(left, right, data.shape[1])
        y_coords = np.linspace(top, bottom, data.shape[0])
        xx, yy = np.meshgrid(x_coords, y_coords)
       
        if not planes:
            print("No planes to save.")
            return
        
        
        output_filename_dxf = f"{out_planeFolder}/planes.dxf"
        for idx, plane in enumerate(planes):
            output_filename = f"{out_planeFolder}/planes_{idx}.xyz"
            output_filename_idx = f"{out_planeFolder}/planes_{idx}.dxf"
            
            # Initialize lists to store vertices and faces
            vertices = []
            
            # Sampled point interval
            sample_interval = 200
            
            # Open the XYZ file for writing
            with open(output_filename, 'w') as xyz_file:
                for i in range(0, plane.shape[0], sample_interval):
                    for j in range(0, plane.shape[1], sample_interval):
                        if i < xx.shape[0] and j < xx.shape[1]:
                            x = xx[i, j]
                            y = yy[i, j]
                            z = plane[i, j]

                            # Check for NaN values
                            if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                                # Add the vertex to the list
                                vertices.append((x, y, z))
                               
                                # Write vertex data to the XYZ file
                                xyz_file.write(f'{x} {y} {z}\n')
            
            create_dxf_from_xyz_files(list_xyz_files_in_folder(out_planeFolder), output_filename_idx, single=idx)
            
                  
        print(f'zz values saved to {output_filename}')
            
    
    # import open3d as o3d

    # def save_planes_to_obj():
    #     x_coords = np.linspace(left, right, data.shape[1])
    #     y_coords = np.linspace(top, bottom, data.shape[0])
    #     xx, yy = np.meshgrid(x_coords, y_coords)
        
    #     if not planes:
    #         print("No planes to save.")
    #         return
        
    #     output_filename_dxf = f"{out_planeFolder}/planes.dxf"
    #     for idx, plane in enumerate(planes):
    #         output_filename_xyz = f"{out_planeFolder}/planes_{idx}.xyz"
    #         output_filename_obj = f"{out_planeFolder}/planes_{idx}.obj"
            
    #         # Initialize lists to store vertices and faces
    #         vertices = []
            
    #         # Sampled point interval
    #         sample_interval = 200
            
    #         # Open the XYZ file for writing
    #         with open(output_filename_xyz, 'w') as xyz_file:
    #             for i in range(0, plane.shape[0], sample_interval):
    #                 for j in range(0, plane.shape[1], sample_interval):
    #                     if i < xx.shape[0] and j < xx.shape[1]:
    #                         x = xx[i, j]
    #                         y = yy[i, j]
    #                         z = plane[i, j]

    #                         # Check for NaN values
    #                         if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
    #                             # Add the vertex to the list
    #                             vertices.append((x, y, z))
                            
    #                             # Write vertex data to the XYZ file
    #                             xyz_file.write(f'{x} {y} {z}\n')
            
    #         # Load the XYZ file as a point cloud
    #         pcd = o3d.io.read_point_cloud(output_filename_xyz, format='xyz')
            
    #         # Optionally estimate normals
    #         pcd.estimate_normals()
            
    #         # Create a mesh using the point cloud (using Poisson reconstruction as an example)
    #         radius = 10  # you may need to adjust this based on your data scale
    #         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    #             pcd, o3d.utility.DoubleVector([radius, radius * 2]))
            
    #         # Save mesh as OBJ
    #     o3d.io.write_triangle_mesh(output_filename_obj, mesh)
        
        
    def collect_points_from_line(gdf):
        """Collect start, middle, and end points from each line feature."""
        collected_points = []

        for geometry in gdf.geometry:
            if geometry.geom_type == "LineString":
                x, y = geometry.xy
                start = (x[0], y[0])
                end = (x[-1], y[-1])
                
                #print('start: ' ,start)
                
                # # Create a LineString from coordinates to easily get a point at 50% distance
                line = LineString(zip(x, y))
                # if len(line.coords) >= 3:
                #     # If the line has more than 2 vertices, collect points at vertices
                #     vertices = list(line.coords)
                #     #print('vertics: ', vertices)
                #     vertex_list=[i for i in vertices]
                #     for kj in vertex_list:
                #         collected_points.append(kj)
                #     #collected_points.extend(vertices)
                # else:
                    
                middle = line.interpolate(0.5, normalized=True).coords[0]
                p2 = line.interpolate(0.25, normalized=True).coords[0]
                p1 = line.interpolate(0.75, normalized=True).coords[0]

                collected_points.append([start, middle, end, p1,p2])

        return collected_points

    # def collect_points_from_line(gdf):
    #     # Initialize an empty list to store the coordinates
    #     collected_points = []

    #     # Iterate through the GeoDataFrame
    #     for geometry in gdf['geometry']:
    #         if isinstance(geometry, LineString):
    #             # Count the number of vertices in the LineString
    #             num_vertices = len(geometry.coords)
                
    #             # Check if the LineString is straight (3 or fewer vertices)
    #             if num_vertices <= 3:
    #                 # For straight lines, take coordinates of start, end, and middle
    #                 start_point = geometry.coords[0]
    #                 end_point = geometry.coords[-1]
    #                 middle_point = geometry.interpolate(0.5, normalized=True).coords[0]
                    
    #                 collected_points.extend([start_point, middle_point, end_point])
    #             else:
    #                 # For curved lines, take coordinates of all vertices
    #                 for point in geometry.coords:
    #                     x, y = point
    #                     collected_points.append((x, y))
        
    #     return collected_points

# Now, coordinates_list contains the coordinates of start, end, middle, and all vertices
# based on whether the line is straight (3 or fewer vertices) or curved
        
    

    def draw_planes_from_lines(xx,yy):
        global ax_2d, ax_3d, data, transform, left, right, bottom, top, planes 
        
        data, transform, projection = read_dem(dem_data)

        gdf = gpd.read_file(line_shapefile)
        if gdf.crs.to_string() != projection:
            gdf = gdf.to_crs(projection)

        point_sets = collect_points_from_line(gdf)
        # x_coords = np.linspace(left, right, data.shape[1])
        # y_coords = np.linspace(top, bottom, data.shape[0])
        # xx, yy = np.meshgrid(x_coords, y_coords)
        
        from tqdm import tqdm
        planes=[]
        for points in tqdm(point_sets, desc="Processing fitted planes"):
            #print('points: ', points)
            #z_values = [data[int((point[1] - top) / abs(transform[5])), int((point[0] - left) / transform[1])] for point in points]
            
            z_values = []

            for point in points:
                y_index = int((point[1] - top) / abs(transform[5]))
                # y_index = int((point[1] - top) / abs(transform[5])) if isinstance(point[1], (int, float)) and isinstance(transform[5], (int, float)) else None
                # x_index = int((point[0] - top) / abs(transform[1])) if isinstance(point[0], (int, float)) and isinstance(transform[1], (int, float)) else None

                x_index = int((point[0] - left) / transform[1])
                
                z_value = data[y_index][x_index]
                
                z_values.append(z_value)

            xyz_points = [points[i] + (z_values[i],) for i in range(len(z_values))]

            if len(xyz_points) >= 3:
                fitted_plane, a, b, c, dip_angle, dip_direction = fit_plane(xyz_points, xx, yy)
                
                dip_angle_list.append(dip_angle)
                dip_direction_list.append(dip_direction)
            
                # if limit_extend:
                #     # Get bounding box of the selected points
                #     xmin, xmax, ymin, ymax = bounding_box(xyz_points)
                    
                #     mask = (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)
                    
                #     for i in range(xx.shape[0]):
                #         for j in range(xx.shape[1]):
                #             if not mask[i, j]:
                #                 fitted_plane[i, j] = np.nan  # Set out-of-bounds values to NaN
                ###########
                
                #########
            
                planes.append(fitted_plane)
                #plane_color = random_color()
                plane_color=color_from_dip_and_direction(dip_angle, dip_direction)
                plane_colors.append(plane_color)
                
                
        
            ####
                for pointn in xyz_points:
                    ax_2d.plot(pointn[0], pointn[1], 'o', color=plane_color, alpha=0.7)
                canvas_2d.draw()

                #ax_3d.clear()
                ax_3d.set_title('3D DEM with Fitted Planes')
                ax_3d.plot_surface(xx, yy, data, cmap='terrain', linewidth=0, antialiased=True, alpha=0.5)
                #for plane in planes:
                ax_3d.plot_surface(xx, yy, fitted_plane, alpha=0.6, linewidth=0, antialiased=True, color=plane_color)
                canvas_3d.draw()
                
                plot_planes(dip_angle_list, dip_direction_list, plane_colors)
            
        
    

    def plot_dem(data, transform, projection, path_to_shapefile):
        global ax_2d, ax_3d, points, canvas_2d, canvas_3d, points_listbox, planes, plane_colors, hs, left, right, bottom, top
        points = []
        planes = []
        plane_colors = []
        
        

        hs = es.hillshade(data)

        root = tk.Tk()
        root.title("Akhdefo_FitPlanes")

        left, cell_size_x, _, top, _, cell_size_y = transform
        
       
        right = left + cell_size_x * data.shape[1]
        bottom = top + cell_size_y * data.shape[0]

        fig1 = plt.Figure(figsize=(6, 6), dpi=150)
        ax_2d = fig1.add_subplot(111)
        ax_2d.imshow(hs, cmap='gray', extent=[left, right, bottom, top])
        ax_2d.set_title('Hillshade', fontweight="bold", fontsize=14)
        ax_2d.set_xlabel("Longitude", fontsize=12)
        ax_2d.set_ylabel("Latitude", fontsize=12)
        ax_2d.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax_2d.set_aspect('equal')
        
        
        
        if path_to_shapefile is not None:
            gdf = gpd.read_file(path_to_shapefile)
            if gdf.crs.to_string() != projection:
                gdf = gdf.to_crs(projection)
            gdf.plot(ax=ax_2d, color='red', label="Linear Features")
            ax_2d.legend(loc="upper left")
        
        fig2 = plt.Figure(figsize=(6, 6), dpi=150)
        # Create a figure and axis
        ax_3d = fig2.add_subplot(111, projection='3d')
        ax_3d.set_title('3D DEM with Fitted Planes', fontweight="bold", fontsize=14)
        ax_3d.set_xlabel("Longitude", fontsize=12)
        ax_3d.set_ylabel("Latitude", fontsize=12)
        ax_3d.set_zlabel("Elevation", fontsize=12)
        x_coords = np.linspace(left, right, data.shape[1])
        y_coords = np.linspace(top, bottom, data.shape[0])
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        
        

        

        # ax_3d.set_xlim([left, right])
        # ax_3d.set_ylim([bottom, top])
        # ax_3d.set_zlim([zmin, zmax])

        frame_2d = tk.Frame(root)
        frame_2d.grid(row=0, column=0, sticky='nsew')
        # canvas_2d = FigureCanvasTkAgg(fig1, master=frame_2d)
        # canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=False)
        # toolbar_2d = NavigationToolbar2Tk(canvas_2d, frame_2d)
        # toolbar_2d.update()
        # Create the canvas for the plot
        canvas_2d = FigureCanvasTkAgg(fig1, master=frame_2d)
        canvas_2d.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Fill the frame
        # Create the toolbar and place it at the top of the canvas
        toolbar_2d = NavigationToolbar2Tk(canvas_2d, frame_2d)
        toolbar_2d.update()
        toolbar_2d.pack(side=tk.TOP, fill=tk.X)  # Pack the toolbar at the top
        
        #######add colorbar figure to Canvas
        



        frame_3d = tk.Frame(root)
        frame_3d.grid(row=0, column=1, sticky='nsew')
        # canvas_3d = FigureCanvasTkAgg(fig2, master=frame_3d)
        # canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=False)
        # toolbar_3d = NavigationToolbar2Tk(canvas_3d, frame_3d)
        # toolbar_3d.update()
        canvas_3d = FigureCanvasTkAgg(fig2, master=frame_3d)
        canvas_3d.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Fill the frame
        # Create the toolbar and place it at the top of the canvas
        toolbar_3d = NavigationToolbar2Tk(canvas_3d, frame_3d)
        toolbar_3d.update()
        toolbar_3d.pack(side=tk.TOP, fill=tk.X)  # Pack the toolbar at the top

        # Create a frame on the right to hold controls
        controls_frame = tk.Frame(root)
        controls_frame.grid(row=0, column=3, rowspan=1, sticky='nsew')
        

        # Now put all buttons and Listbox inside this controls_frame
        draw_button = ttk.Button(controls_frame, text="Draw Plane", command=lambda: draw_plane(xx,yy))
        draw_button.pack(pady=2, padx=2)

        line_plane_button = ttk.Button(controls_frame, text="Create Planes from Lines", command=lambda: draw_planes_from_lines(xx,yy))
        line_plane_button.pack(pady=2, padx=2)

        save_plane_button = ttk.Button(controls_frame, text="Save Planes to OBJ", command=save_planes_to_obj) #save_fitted_planes_as_dxf
        save_plane_button.pack(pady=2, padx=2)
        # save_plane_button_dxf = ttk.Button(controls_frame, text="Save Planes to DXF", command=save_fitted_planes_as_dxf)
        # save_plane_button_dxf.pack(pady=10, padx=10)

        points_listbox = tk.Listbox(controls_frame, height=5, width=40, exportselection=False)
        points_listbox.pack(pady=2, padx=2, fill='both', expand=True)
        
        scrollbar = ttk.Scrollbar(controls_frame, orient="vertical", command=points_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        points_listbox.config(yscrollcommand=scrollbar.set)
        points_listbox.bind("<Double-Button-1>", on_point_double_click)
        ####
    
        
        ####

        fig1.canvas.mpl_connect('button_press_event', lambda event: onclick(event, points, data, transform))
        
        
        # Adding the size grip for window resizing
        sizegrip = ttk.Sizegrip(root)
        sizegrip.grid(row=3, column=1, sticky='nsew')

        # Making the design responsive
        root.grid_rowconfigure(0, weight=1)  # Canvas row
        root.grid_rowconfigure(1, weight=0)  # Toolbar row
        root.grid_rowconfigure(2, weight=0)  # Button row
        root.grid_columnconfigure(0, weight=1)  # 2D plot column
        root.grid_columnconfigure(1, weight=1)  # 3D plot column
        
        
                
        root.mainloop()


    
    plot_dem(data, transform, projection, line_shapefile)
   
    
#akhdefo_fitPlane(dem_data='currie/dem_1m.tif', line_shapefile='currie/lines.shp', out_planeFolder='Planes_out/new', limit_extend=False)

import os
import shutil

def move_files_with_string(source_dir: str ="", dest_dir: str ="", search_string: str =".tif"):
    """
    Move files from a source directory to a destination directory based on a search string present in their paths.

    Parameters:
    - source_dir (str): The directory from which files are to be moved.
    - dest_dir (str): The destination directory where files will be moved.
    - search_string (str): The string to search for in the file paths.

    This function traverses the source directory, including its subdirectories. 
    Files whose paths contain the search string are moved to the destination directory. 
    If a file with the same name exists in the destination, it's renamed to avoid overwriting.

    Errors during file movement (e.g., permission issues, non-existent directories) are logged but do not stop the process.
    """

    if not os.path.exists(source_dir):
        raise ValueError(f"Source directory '{source_dir}' does not exist.")

    # Create the destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if search_string in file_path:
                dest_file_path = os.path.join(dest_dir, file)

                # Check if the file already exists in destination
                counter = 1
                while os.path.exists(dest_file_path):
                    # Split filename and extension
                    file_base, file_extension = os.path.splitext(file)
                    # Create a new filename with a counter to avoid overwriting
                    dest_file_path = os.path.join(dest_dir, f"{file_base}_{counter}{file_extension}")
                    counter += 1

                # Move the file to the destination directory
                try:
                    shutil.move(file_path, dest_file_path)
                except Exception as e:
                    print(f"Error moving file {file_path} to {dest_file_path}: {e}")

#############################################
#This part is still under construction

import numpy as np

def partial_derivative(f, dx, dy):
    """
    Calculate the central difference approximation for the partial derivatives
    of a 2D function with respect to x and y.

    :param f: 2D numpy array of function values, representing the raster
    :param dx: Spacing in the x-direction (assumed to be constant)
    :param dy: Spacing in the y-direction (assumed to be constant)
    :return: Two 2D numpy arrays, one for the partial derivative with respect to x (df_dx)
             and one for the partial derivative with respect to y (df_dy)
    """
     # Initialize arrays to store the partial derivatives
    df_dx = np.zeros_like(f)
    df_dy = np.zeros_like(f)

    # Compute the partial derivatives for the internal points
    for xi in range(1, f.shape[0] - 1):
        for yj in range(1, f.shape[1] - 1):
            df_dx[xi, yj] = (f[xi + 1, yj] - f[xi - 1, yj]) / (2 * dx)
            df_dy[xi, yj] = (f[xi, yj + 1] - f[xi, yj - 1]) / (2 * dy)

    # Handle the boundaries by setting the derivative to zero or using one-sided differences
    # Here we choose to set the boundary derivative to zero
    # Alternatively, use forward/backward difference at the boundaries if needed

    return df_dx, df_dy


import numpy as np
import rasterio

def calculate_slope(dem, aspect, dx , dy):
    """
    Calculate the slope at each pixel using the aspect to determine direction.

    Parameters:
    dem (numpy.ndarray): Digital Elevation Model (DEM) array.
    aspect (numpy.ndarray): Aspect array.
    dx (float):  x Spatial resolution of the raster (distance between pixels).
    dy (float):  y Spatial resolution of the raster (distance between pixels).

    Returns:
    numpy.ndarray: Array of slope values in degrees.
    """
    
    # grad_y, grad_x = np.gradient(dem, dx, dy)
    
    
    # Assuming 'dem' and 'aspect' are your existing numpy arrays
    # Check if the shapes are different
    # if dem.shape != aspect.shape:
    #     # Resize 'dem' to match the shape of 'aspect'
    #     aspect = np.resize(aspect, dem.shape)
    # else:
    #     aspect = aspect



    grad_y, grad_x = np.gradient(dem, dx, dy)
    #grad_y, grad_x=partial_derivative(dem, dx, dy)

    if grad_x.shape != grad_y.shape:
        raise ValueError("Gradient arrays have mismatched shapes.")
    
    aspect_radians = np.deg2rad(aspect)
    directional_grad = np.cos(aspect_radians) * grad_x + np.sin(aspect_radians) * grad_y
    #slope_degrees = np.rad2deg(np.arctan(directional_grad))
   
    # Calculate slope in radians
    slope_radians = np.arctan(directional_grad)

    # Ensure slope is within 0 to π/2 radians (0 to 90 degrees) range
    #slope_radians = slope_radians % (np.pi / 2)

    # Convert slope to degrees
    slope_degrees = (np.rad2deg(slope_radians)) % 90 
    #slope_degrees= (135-slope_degrees)% 90 
    
   
    
    return slope_degrees, directional_grad

def calculate_height_change(slope, distance, dem):
    """
    Calculate the height change using the slope and distance for each pixel.

    Parameters:
    slope (numpy.ndarray): Array of slope values in degrees.
    distance (numpy.ndarray): Array of distance values.

    Returns:
    numpy.ndarray: Array of height changes.
    """
    slope_radians = np.deg2rad(slope)
    height_change = np.tan(slope_radians) * distance
    
    
    
    return height_change 

def calculate_volume_change(height_change, pixel_area):
    """
    Calculate the volume change for each pixel.

    Parameters:
    height_change (numpy.ndarray): Array of height changes.
    pixel_area (float): Area of a single pixel.
    
    Returns:
    numpy.ndarray: Array of volume changes.
    """
    volume_change = height_change * pixel_area
    print (f'Total Volume: {np.nansum(volume_change)} cubic meter')
    return volume_change

from skimage.filters import gaussian

def displacement_to_volume(dem_path="", aspect_path="", displacement_path="", slope_output_path="", height_output_path="", volume_output_path="", dx=None , dy=None , pixel_area=None):
    """
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
    """
    with rasterio.open(dem_path) as dem_raster, \
         rasterio.open(aspect_path) as aspect_raster, \
         rasterio.open(displacement_path) as displacement_raster:

        dem = dem_raster.read(1,  masked=True)
        aspect = aspect_raster.read(1,  masked=True)
        displacement = displacement_raster.read(1,  masked=True)
        
        x_resolution, y_resolution = displacement_raster.res
        
        if dx is None:
            dx=x_resolution
        if dy is None:
            dy=y_resolution
        if pixel_area is None:
            pixel_area=dx * dy
        from scipy.ndimage import zoom

        # Example dimensions - replace these with actual dimensions
        dem_shape = dem.shape
        aspect_shape = aspect.shape
        displacement_shape = displacement.shape

        # Calculating zoom factors
        zoom_factor_aspect = [dem_dim / aspect_dim for dem_dim, aspect_dim in zip(dem_shape, aspect_shape)]
        zoom_factor_displacement = [dem_dim / displacement_dim for dem_dim, displacement_dim in zip(dem_shape, displacement_shape)]

        # Resizing aspect and displacement arrays
        aspect = zoom(aspect, zoom_factor_aspect, order=1)  # cubic interpolation
        displacement = zoom(displacement, zoom_factor_displacement, order=1)  # cubic interpolation
        
        # aspect = np.resize(aspect, dem.shape)
        # displacement = np.resize(displacement, dem.shape)
        
        slope, directional_grad = calculate_slope(dem, aspect, dx , dy )
        height_change  = calculate_height_change(slope, directional_grad, dem)
        #height_change=gaussian(height_change, sigma=3)
        volume_change = calculate_volume_change(height_change, pixel_area)
       
       

        # Define a function to export data to a GeoTIFF
        def export_to_geotiff(data, output_path, reference_raster):
            with rasterio.open(
                output_path, 'w',
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=np.float32,
                crs=reference_raster.crs,
                transform=reference_raster.transform, nodata=np.nan
            ) as output_raster:
                output_raster.write(data, 1)

        # Exporting slope, height change, and volume change to GeoTIFFs
        export_to_geotiff(slope, slope_output_path, dem_raster)
        export_to_geotiff(height_change, height_output_path, dem_raster)
        export_to_geotiff(volume_change, volume_output_path, dem_raster)
       



import numpy as np
from osgeo import gdal

def calculate_and_save_aspect_raster(ew_raster_path: str ="", ns_raster_path: str ="", output_raster_path: str =""):
    """
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
    
    """
    def read_raster(file_path):
        """Read a raster file and return the data array and geotransform."""

        # Open the dataset
        dataset = gdal.Open(file_path)
        if dataset is None:
            raise IOError("Could not open file at {}".format(file_path))

        # Get the first raster band
        band = dataset.GetRasterBand(1)

        # Read the data as a numpy array
        data = band.ReadAsArray()

        # Get no-data value from the band
        nodata_value = band.GetNoDataValue()

        # Check if there is a no-data value defined
        if nodata_value is not None:
            # Create a mask that is True for valid pixels
            mask = data != nodata_value

            # Apply the mask to filter out no-data pixels
            valid_data = np.where(mask, data, np.nan)  # Replace no-data with NaN
        else:
            valid_data = data  # All data is valid if there is no no-data value

        # Retrieve geotransformation
        geotransform = dataset.GetGeoTransform()

        # Close the dataset
        #dataset = None

        return valid_data, geotransform, dataset

    def calculate_aspect(ew_data, ns_data):
        """Calculate the aspect from EW and NS displacement data."""
        with np.errstate(divide='ignore', invalid='ignore'):
            aspect = np.arctan2(ew_data, ns_data)
            aspect_deg=np.degrees(aspect)
            aspect_deg = (450 -aspect_deg ) % 360
        return aspect_deg

    def save_raster(output_path, data, geo_transform, reference_dataset):
        """Save the data as a raster file."""
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = data.shape
        out_raster = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
        out_raster.SetGeoTransform(geo_transform)
        out_band = out_raster.GetRasterBand(1)
        out_band.WriteArray(data)
        out_raster.SetProjection(reference_dataset.GetProjection())
        out_band.FlushCache()

    # Read the EW and NS raster data
    ew_data, geo_transform, dataset = read_raster(ew_raster_path)
    ns_data, _, _ = read_raster(ns_raster_path)

    # Calculate the aspect and save the result
    aspect_data = calculate_aspect(ew_data, ns_data)
    save_raster(output_raster_path, aspect_data, geo_transform, dataset)

# Example usage:
# calculate_and_save_aspect_raster('path_to_ew_raster.tif', 'path_to_ns_raster.tif', 'path_to_output_aspect_raster.tif')



##############################################

