import os
from osgeo import gdal
import tempfile
import numpy as np

import gc

def mask_raster_with_template(input_raster_path, mask_raster_path, noData_value=np.nan):
    """
    Masks a georeferenced raster file using a binary raster mask template.

    Parameters:
    - input_raster_path (str): Path to the input georeferenced raster file.
    - mask_raster_path (str): Path to the binary raster mask template.

    Returns:
    None. The input raster file will be replaced by the masked raster.
    """
    
    # Open the input raster and mask raster
    input_ds = gdal.Open(input_raster_path, gdal.GA_ReadOnly)
    mask_ds = gdal.Open(mask_raster_path, gdal.GA_ReadOnly)

    # Create memory target raster with same dimensions as input raster
    mem_drv = gdal.GetDriverByName('MEM')
    target_ds = mem_drv.Create('', input_ds.RasterXSize, input_ds.RasterYSize, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform(input_ds.GetGeoTransform())
    target_ds.SetProjection(input_ds.GetProjection())

    # Reproject mask raster to match input raster
    gdal.ReprojectImage(mask_ds, target_ds, mask_ds.GetProjection(), input_ds.GetProjection(), gdal.GRA_NearestNeighbour)
    mask_band = target_ds.GetRasterBand(1).ReadAsArray()

    # Create a temporary file to store masked raster
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tif").name

    # Loop through bands in input raster and apply mask
    out_ds = gdal.GetDriverByName('GTiff').Create(temp_file, input_ds.RasterXSize, input_ds.RasterYSize, input_ds.RasterCount, input_ds.GetRasterBand(1).DataType)
    out_ds.SetGeoTransform(input_ds.GetGeoTransform())
    out_ds.SetProjection(input_ds.GetProjection())

    for band in range(1, input_ds.RasterCount + 1):
        input_band_data = input_ds.GetRasterBand(band).ReadAsArray()
        input_band_data[mask_band == 0] = noData_value  # Set pixels to 0 where mask is 0

        out_band = out_ds.GetRasterBand(band)
        out_band.WriteArray(input_band_data)
        out_band.FlushCache()

    input_ds = None
    mask_ds = None
    out_ds = None
    
    # Replace original raster with the masked raster
    # os.remove(input_raster_path)
    # os.rename(temp_file, input_raster_path)
    import shutil
    shutil.copy(temp_file, input_raster_path)
    os.remove(temp_file)



def mask_all_rasters_in_directory(directory, mask_raster_path):
    
    """
    Masks all georeferenced raster files in a specified directory using a binary raster mask template.

    Parameters:
    - directory (str): Path to the directory containing the georeferenced raster files.
    - mask_raster_path (str): Path to the binary raster mask template.

    Returns:
    None. Each raster file in the specified directory will be replaced by its corresponding masked raster.
    
    """
    
    for file in os.listdir(directory):  # This will only list files/directories in the given directory
        if file.lower().endswith(('.tif', '.tiff')):
            input_raster_path = os.path.join(directory, file)
            mask_raster_with_template(input_raster_path, mask_raster_path)


import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from pykrige.ok import OrdinaryKriging
from scipy.stats import zscore


#Calculate Linear Velocity for each data point
def linear_VEL(df, dnames):
    
    dd_list=[x.replace("D", "") for x in dnames]
    dates_list=([datetime.strptime(x, '%Y%m%d') for x in dd_list])
    days_num=[( ((x) - (pd.Timestamp(year=x.year, month=1, day=1))).days + 1) for x in dates_list]
    days_num=list(range(0, len(dnames)))
    dslope=[]
    std_slope=[]
    for index, dr in df.iterrows():
        #if index==0:
        rows=df.loc[index, :].values.flatten().tolist()
        row_values=rows
        # dfr = pd.DataFrame(dr).transpose()
        # dfr = dfr.loc[:, ~dfr.columns.str.contains('^Unnamed')]
    
        #slopeVEL=best_fit_slope_and_intercept(days_num, row_values)
        #print("slope", slopeVEL[0])
        slope, intercept, r_value, p_value, std_err = stats.linregress(days_num, row_values)
        dslope.append(slope)
        std_slope.append(std_err)
    return dslope, std_slope

# def replace_outliers_with_nan_zscore(gdf, column_name, threshold):
#     # Create a copy of the GeoDataFrame to avoid modifying the original
#     #modified_geodataframe = geodataframe.copy()

#     # Calculate Z-scores for the specified column
#     z_scores = np.abs(zscore(gdf[column_name]))

#     # Replace outliers with NaN values based on the threshold
#     gdf.loc[z_scores > threshold, column_name] = np.nan

#     return gdf

# import numpy as np
# import geopandas as gpd
# import gstools as gs
# import matplotlib.pyplot as plt

# # Function to fit variogram models and visualize them
# def fit_variogram_models(x, y, z, latlon=False , show_plot=False):
#     bin_center, gamma = gs.vario_estimate((x, y), z, latlon=latlon)
    
#     # Models to test
#     models = {
#         "Gaussian": gs.Gaussian,
#         "Exponential": gs.Exponential,
#         "Matern": gs.Matern,
#         "Integral": gs.Integral,
#         "Cubic": gs.Cubic,
#         "Stable": gs.Stable,
#         "Rational": gs.Rational,
#         "Spherical": gs.Spherical,
#         "SuperSpherical": gs.SuperSpherical,
#         "JBessel": gs.JBessel,
#         "HyperSpherical": gs.HyperSpherical,
#         "TPLSimple": gs.TPLSimple
#     }

#     scores = {}
#     fig, ax = plt.subplots(figsize=(10, 5))
#     ax.scatter(bin_center, gamma, color="k", label="Empirical")

#     best_model, best_score, best_fit = None, -10, None
    
#     for model_name, Model in models.items():
#         try:
#             fit_model = Model(dim=2)
#             _, _, r2 = fit_model.fit_variogram(bin_center, gamma, return_r2=True)
#             scores[model_name] = r2
            
#             if r2 > best_score:
#                 best_score = r2
#                 best_model = model_name
#                 best_fit = fit_model
            
#             # Plot the model
#             fit_model.plot(x_max=max(bin_center), ax=ax, label=f"{model_name} (R2: {r2:.5f})")
#         except Exception as e:
#             print(f"Error with model {model_name}: {e}")
    
#     ax.legend()
#     ax.set_title(f'Variogram Models with Fitted Data, Best Model= {best_model}, with score{best_score}')
#     if show_plot== True:
#         plt.show()
#     else:
#         plt.close()
#     return best_model, best_score, scores, best_fit
# def interpolate_kriging_nans_geodataframe(data, threshold=None, variogram_model=None, 
# out_fileName=None, plot=False, Total_days=None, VEL_scale=None, VEL_Mode=None):
    
#     if isinstance(data, str):
#         if data[-4:] == '.shp':
#             gdf = gpd.read_file(data)

#             geom=gdf['geometry']
#             crs_ini=gdf.crs
           
#             out_fileName = data
#             Total_days=Total_days
#         else:
#             raise ValueError("Unsupported file format.")
#     elif isinstance(data, gpd.GeoDataFrame):
#         gdf = data
#         out_fileName = None
#     else:
#         raise ValueError("Unsupported data type.")

#     unwanted_cols = ['CODE','geometry', 'x', 'y', 'VEL', 'VEL_STD']
#     columns_to_interpolate = [col for col in gdf.columns if col not in unwanted_cols]


#     for col in columns_to_interpolate:
#         if threshold is not None:
#             gdf = replace_outliers_with_nan_zscore(gdf, col, threshold)
#         try:
#             known_data = gdf[~gdf[col].isna()]
#             unknown_data = gdf[gdf[col].isna()]

#             known_coords = [(geom.x, geom.y) for geom in known_data.geometry]
#             unknown_coords = [(geom.x, geom.y) for geom in unknown_data.geometry]


#             #known_coords = list(known_data.geometry.apply(lambda geom: (geom.x, geom.y)))
#             known_values =  [x for x in known_data[col]]

#             #unknown_coords = list(unknown_data.geometry.apply(lambda geom: (geom.x, geom.y)))
#             if variogram_model is None:
#                 best_model, best_score, scores, best_fit=fit_variogram_models(gdf.x, gdf.y, gdf[col], latlon=False, show_plot=plot)
#             else:
#                 best_fit=variogram_model
#             ok = OrdinaryKriging(
#                 [coord[0] for coord in known_coords],
#                 [coord[1] for coord in known_coords],
#                 known_values,
#                 variogram_model=best_fit,
#                 verbose=False
#             )

#             interpolated_values, _ = ok.execute(
#                 'points',
#                 [coord[0] for coord in unknown_coords],
#                 [coord[1] for coord in unknown_coords]
#             )

#             gdf.loc[unknown_data.index, col] = interpolated_values
#         except Exception as e:
#             print(f"Error with kriging interpolation: {e}, interpolation performed using pandas interpolation")
        
#         gdf[col]=gdf[col].interpolate()
#     # Interpolating NaN values in the subset DataFrame
#     gdf[columns_to_interpolate] = gdf[columns_to_interpolate].interpolate(axis=1, limit_direction='both')
        
#     zcol=columns_to_interpolate[0]
#     #zcol=[gdf[z] for z in columns_to_interpolate]
#     ######################
#     if isinstance(data, str):
#         if data[-4:] == '.shp':
#             # Reset the index and convert it to a column
#             gdf = gdf.reset_index()

#             # Rename the index column to "CODE"
#             gdf.rename(columns={'index': 'CODE'}, inplace=True)

            

#             if VEL_Mode=='linear' and VEL_scale=='year':
#                 VEL, VEL_STD=linear_VEL(gdf[columns_to_interpolate], columns_to_interpolate)
#                 gdf['VEL']=VEL 
#                 gdf['VEL']= gdf['VEL']/ Total_days * 365
            
#                 gdf['VEL_STD']=VEL_STD 
#                 gdf['VEL_STD']=gdf['VEL_STD']/ Total_days * 365

#             if VEL_Mode=='mean' and VEL_scale=='year':
#                 VEL=gdf[columns_to_interpolate].mean(axis=1)
#                 VEL_STD=gdf[columns_to_interpolate].std(axis=1)
#                 gdf['VEL']=VEL 
#                 gdf['VEL']=gdf['VEL']/ Total_days * 365
                
#                 gdf['VEL_STD']=VEL_STD
#                 gdf['VEL_STD']=gdf['VEL_STD'] / Total_days * 365

#             if VEL_Mode=='linear' and VEL_scale=='month':
#                 VEL, VEL_STD=linear_VEL(gdf[columns_to_interpolate], columns_to_interpolate)
#                 gdf['VEL']=VEL 
#                 gdf['VEL']=gdf['VEL']/ Total_days * 30

#                 gdf['VEL_STD']=VEL_STD 
#                 gdf['VEL_STD']=gdf['VEL_STD']/ Total_days * 30

#             if VEL_Mode=='mean' and VEL_scale=='month':
#                 VEL=gdf[columns_to_interpolate].mean(axis=1)
#                 VEL_STD=gdf[columns_to_interpolate].std(axis=1)
#                 gdf['VEL']=VEL
#                 gdf['VEL']=gdf['VEL'] / Total_days * 30

#                 gdf['VEL_STD']=VEL_STD 
#                 gdf['VEL_STD']=gdf['VEL_STD']/ Total_days * 30

#             if VEL_Mode=='linear' and VEL_scale==None:
#                 VEL, VEL_STD=linear_VEL(gdf[columns_to_interpolate], columns_to_interpolate)
#                 gdf['VEL']=VEL 
#                 gdf['VEL_STD']=VEL_STD 

#             if VEL_Mode=='mean' and VEL_scale==None:
#                 VEL=gdf[columns_to_interpolate].mean(axis=1)
#                 VEL_STD=gdf[columns_to_interpolate].std(axis=1)
#                 gdf['VEL']=VEL
#                 gdf['VEL_STD']=VEL_STD

#             column_order = columns_to_interpolate  # New columns added at the beginning
#             # # Insert new columns at the beginning of the list
#             columns_to_insert = ['CODE', 'x', 'y', 'VEL', 'VEL_STD']  # Inserted in this order
#             col_geo=['geometry']
#             column_order= columns_to_insert + columns_to_interpolate+col_geo
#             gdf=gdf[column_order]
            
#             # Get the CRS of the GeoDataFrame
#             gdf.crs=crs_ini

    

#     if out_fileName is not None:

#         gdf.to_file(out_fileName)

#     if plot is not False:

#         for col in columns_to_interpolate:
#             fig, axes = plt.subplots(1, 2, figsize=(12, 6))

#             gdf.plot(ax=axes[0], column=col, cmap='rainbow', legend=True, markersize=5)
#             axes[0].set_title(f'Before Interpolation - {col}')

#             gdf.plot(ax=axes[1], column=col, cmap='rainbow', legend=True, markersize=5)
#             axes[1].set_title(f'After Interpolation - {col}')

#             plt.tight_layout()
#             plt.show()

    
#     return np.array(gdf.x) , np.array(gdf.y) , np.array(gdf[zcol]), gdf
# # Usage example
# shapefile_path = 'flowx1.shp'
# x, y, z=interpolate_kriging_nans_geodataframe(shapefile_path='flowx.shp', threshold=None, variogram_model='gaussian', 
# out_fileName='flow11', plot=True)
###############################################################################################




######################################

def process_shapefile_with_rasters(shapefile_path, rasterfile_paths):
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Process rasterfile paths
    if isinstance(rasterfile_paths, str):
        rasterfile_paths = [rasterfile_paths]
    raster_paths = sorted(Path(rasterfile_paths[0]).glob('*.tif'), key=lambda x: x.stem)
    rasterfile_paths = [str(path) for path in raster_paths]

    # Load rasters
    rasters = [rasterio.open(path) for path in rasterfile_paths]

    # Extract 'D' columns
    d_columns = [col for col in gdf.columns if col.startswith('D')]

    # Map rasters to 'D' columns
    if len(rasters) == 1 and len(d_columns) <= rasters[0].count:
        raster_band_mapping = {col: (rasters[0], band) for band, col in enumerate(d_columns, start=1)}
    elif len(rasters) == len(d_columns):
        raster_band_mapping = {col: (raster, 1) for col, raster in zip(d_columns, rasters)}
    else:
        raise ValueError("The number of provided rasters or bands does not match the number of 'D' columns in the shapefile.")

    # Sample values for each 'D' column
    for col in d_columns:
        raster, band = raster_band_mapping[col]
        x_values = gdf.geometry.apply(lambda geom: geom.x)
        y_values = gdf.geometry.apply(lambda geom: geom.y)
        sampled_values = raster.sample(list(zip(x_values, y_values)), indexes=band)
        gdf[col] = pd.Series(sampled_values).apply(lambda val: val[0])

    # # # Check for NaN values in 'D' columns
    # all_nan = gdf[d_columns].isna().all(axis=1)
    # # Drop all rows with any NaN values
    # gdf.dropna(inplace=True)
    
    # gdf = gdf.dropna()
    gdf=gdf.fillna(0)
    
    gdf[d_columns]=gdf[d_columns]
    gdf['ssim_mean']=gdf[d_columns].mean(axis=1)
    gdf['ssim_std']=gdf[d_columns].std(axis=1)
    # Drop the columns
    #gdf = gdf.drop(columns=d_columns)
    

    # # Correct displacement to a stable point
    # try:
    #     #VEL, VEL_STD_series = linear_VEL(gdf[d_columns], d_columns)
    #     # Calculate mean, ignoring NaN values
    #     # Calculate mean, ignoring NaN values
    #     VEL = gdf[d_columns].mean(axis=1)

    #     # Calculate standard deviation, ignoring NaN values
    #     VEL_STD_series = gdf[d_columns].std(axis=1)
        
    #     gdf['ssim_mean']=VEL
    #     gdf['ssim_std']=VEL_STD_series
    #     # Drop the specified columns
    #     #gdf = gdf.drop(columns=d_columns, errors='ignore')
    #     # VEL_STD_series = pd.Series(VEL_STD_series)
    #     # min_std_dev = VEL_STD_series.min()
    #     # filtered_indices = VEL_STD_series[VEL_STD_series == min_std_dev].index
    #     # avg_velocities = gdf[d_columns].loc[filtered_indices].mean(axis=1)
    #     # reference_index = avg_velocities.idxmin()
    # except Exception as e:
    #     print("Error in displacement correction:", e)
        #reference_index = None

    return gdf

def find_best_match(gdf):
    # # Check if the dataframe is empty
    if gdf.empty:
        print("Warning: The dataframe is empty. No data to process.")
        return None
     # Diagnostic: Print DataFrame summary
    #print("DataFrame Summary:\n", gdf.describe())
    # Check for exact match
    min_vel_indices = gdf[gdf['VEL'] == gdf['VEL'].min()].index
    min_vel_std_indices = gdf[gdf['VEL_STD'] == gdf['VEL_STD'].min()].index
    # max_ssim_mean_indices = gdf[gdf['ssim_mean'] == gdf['ssim_mean'].max()].index
    # min_ssim_std_indices = gdf[gdf['ssim_std'] == gdf['ssim_std'].min()].index

    #intersection_indices = set(min_vel_indices) & set(min_vel_std_indices) & set(max_ssim_mean_indices) & set(min_ssim_std_indices)
    intersection_indices = set(min_vel_indices) & set(min_vel_std_indices) 

    if intersection_indices:
        best_common_index = list(intersection_indices)
        return best_common_index[0]

    # Calculate score if no exact match
    # Adjust weights as necessary
    gdf['score'] = (
        (gdf['VEL'] - gdf['VEL'].min()).abs() +
        (gdf['VEL_STD'] - gdf['VEL_STD'].min()).abs() )
    
    #(gdf['ssim_mean'].max() - gdf['ssim_mean']).abs() + (gdf['ssim_std'] - gdf['ssim_std'].min()).abs()

     # Check if 'score' column is empty
    if gdf['score'].empty:
        print("Error: The score calculation resulted in an empty series.")
        return None

    # Find the index with minimum score
    best_match_index = gdf['score'].idxmin()
    print('best match', best_match_index)
    
    return best_match_index


######################################3



#############################
import geopandas as gpd
import rasterio
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
import xml.etree.ElementTree as ET

def calculate_slopes_std_and_residuals(geodataframe):
    L_slopes = []
    M_slopes = []
    L_std_devs = []
    M_std_devs = []

    # Extract the date columns (columns starting with 'D')
    date_columns = [col for col in geodataframe.columns if col.startswith('D')]
    dates = pd.to_datetime([col for col in date_columns], format='D%Y%m%d')
    days_since_start = (dates - dates.min()).days


    # Interpolate date_columns row-wise
    for index, row in geodataframe.iterrows():
        row[date_columns] = row[date_columns].interpolate(method='linear')
        row[date_columns].fillna(method='ffill', inplace=True)
        row[date_columns].fillna(method='bfill', inplace=True)
        geodataframe.loc[index, date_columns] = row[date_columns]
        
    # Iterate through each row in the GeoDataFrame
    for index, row in geodataframe.iterrows():
        
        values = [row[col] for col in date_columns]
        

        M_slopes.append(np.mean(values))
        M_std_devs.append(np.std(values))

        # Calculate the slope and intercept using polyfit
        L_slope, intercept = np.polyfit(days_since_start, values, 1)
        L_slopes.append(L_slope)

        # Predict values using the regression line and calculate residuals
        predicted_values = L_slope * days_since_start + intercept
        residuals = values - predicted_values

        # Calculate the standard deviation of the residuals
        L_std_dev = np.std(residuals)
        L_std_devs.append(L_std_dev)

    # Add the slopes, intercepts, and standard deviations as new columns
    geodataframe['VEL'] = L_slopes
    geodataframe['VEL_STD'] = L_std_devs
    # geodataframe['M_VEL'] = M_slopes
    # geodataframe['MSTD_VEL'] = M_std_devs

    # Ensure the returned object is a GeoDataFrame
    return gpd.GeoDataFrame(geodataframe)



best_match_index=None
def update_nodata_values(shapefile_path='', 
                         rasterfile_paths='',
                         interpolate=True, VEL_Mode=None , VEL_scale=None , master_reference=True, Total_days=None , spatial_ref=False):
    
    global best_match_index
    # Load the shapefile
    gdf = gpd.read_file(shapefile_path)
    #gdf = gdf.drop('CODE')
    crs_ini=gdf.crs

    raster_folder = Path(rasterfile_paths)
    raster_paths = sorted(raster_folder.glob('*.tif'), key=lambda x: x.stem)
    if isinstance(rasterfile_paths, str):
        rasterfile_paths = [rasterfile_paths]
        rasterfile_paths=[str(path) for path in raster_paths]

    # Load rasters
    rasters = [rasterio.open(path) for path in rasterfile_paths]

    d_columns = [col for col in gdf.columns if col.startswith('D')]
    if len(rasters) == 1 and len(d_columns) <= rasters[0].count:
        raster_band_mapping = {col: (rasters[0], band) for band, col in enumerate(d_columns, start=1)}
    elif len(rasters) == len(d_columns):
        raster_band_mapping = {col: (raster, 1) for col, raster in zip(d_columns, rasters)}
    else:
        raise ValueError("The number of provided rasters or bands does not match the number of 'D' columns in the shapefile.")

    for col in d_columns:
        raster, band = raster_band_mapping[col]
        
        # Extract x, y coordinates for all rows at once
        x_values = gdf.geometry.apply(lambda geom: geom.x)
        y_values = gdf.geometry.apply(lambda geom: geom.y)
        # Sample all values for the current column at once
        sampled_values = raster.sample(list(zip(x_values, y_values)), indexes=band)
        # Update the entire column in one go
        #gdf[col] = [val[0] for val in sampled_values]
        gdf[col] = pd.Series(sampled_values).apply(lambda val: val[0])
    

    # Check if all cells in d_columns are NaN
    all_nan = gdf[d_columns].isna().all(axis=1)

    if interpolate:
        # Interpolate only for rows where all cells in d_columns are not NaN
        gdf.loc[~all_nan, d_columns] = gdf.loc[~all_nan, d_columns].interpolate(
            method='linear', limit_direction='both', axis=1
        )
        ######
         # Check if there are still NaN values after interpolation
        still_nan_after_interpolate = gdf[d_columns].isna().any(axis=1)

        # Apply ffill (forward fill) and then bfill (backward fill) as fallbacks
        gdf.loc[still_nan_after_interpolate, d_columns] = gdf.loc[still_nan_after_interpolate, d_columns].ffill(axis=1).bfill(axis=1)
        
        #####

    # Drop rows where all cells in d_columns are NaN
    gdf = gdf[~all_nan]
   
    gdf = gdf.dropna()
    # Check if master_reference is not None before proceeding with the code
    if master_reference is not None:
    
        #Subtract columns based on master_reference flag
        if master_reference=='single': 
            master_col = d_columns[0]
            #cols_without_mastercol=d_columns[1:]
            for col in d_columns:
                gdf[col] = gdf[col] - gdf[master_col]
            gdf[d_columns] = gdf[d_columns].cumsum(axis=1)
            
        elif master_reference=='multiple':
            for i in range(1, len(d_columns)):
                
                gdf[d_columns[i]] = gdf[d_columns[i]] - gdf[d_columns[i-1]]
            gdf[d_columns[0]] = gdf[d_columns[0]] - gdf[d_columns[0]]
            gdf[d_columns] = gdf[d_columns].cumsum(axis=1)
    else:
        # Code to handle the case when master_reference is None
          # You can add appropriate handling here, or leave it empty if you want to skip the code entirely
        #gdf[d_columns] = gdf[d_columns].cumsum(axis=1)
        pass
       
    # Set data types: first d_column to int, the rest to float32
    #gdf[d_columns[0]] = gdf[d_columns[0]].astype(int)
    #for col in d_columns[1:]:
    for col in d_columns:
        gdf[col] = gdf[col].astype(np.float32)
    for raster in rasters:
        raster.close()
    ########################
    
    #  # Reset the index and convert it to a column
    # gdf = gdf.reset_index()

    # # # Rename the index column to "CODE"
    # gdf=gdf.rename(columns={'index': 'CODE'})
    gdf=calculate_slopes_std_and_residuals(gdf)
    def calculate_VEL(gdf, VEL_scale=VEL_scale, Total_days=Total_days):
        if VEL_Mode=='linear' and VEL_scale=='year':
            # VEL, VEL_STD=linear_VEL(gdf[d_columns], d_columns)
            # gdf['VEL']=VEL 
            #gdf['VEL']= gdf['VEL']/ Total_days * 365
            gdf['VEL']= gdf['VEL'] *  365.25
        
            #gdf['VEL_STD']=VEL_STD 
            #gdf['VEL_STD']=gdf['VEL_STD']/ Total_days * 365
            gdf['VEL_STD']=gdf['VEL_STD']/Total_days * 365.25
            
            
        
        if VEL_Mode=='mean' and VEL_scale=='year':
            VEL=gdf[d_columns].mean(axis=1)
            VEL_STD=gdf[d_columns].std(axis=1)
            gdf['VEL']=VEL 
            gdf['VEL']=gdf['VEL']/ Total_days * 365.25
            #gdf['VEL']= gdf['VEL'] * 365.25
        
            gdf['VEL_STD']=VEL_STD
            gdf['VEL_STD']=gdf['VEL_STD'] / Total_days * 365.25
            #gdf['VEL_STD']=gdf['VEL_STD'] * 365

        if VEL_Mode=='linear' and VEL_scale=='month':
            # VEL, VEL_STD=linear_VEL(gdf[d_columns], d_columns)
            # gdf['VEL']=VEL 
            #gdf['VEL']=gdf['VEL']/ Total_days * 30
            gdf['VEL']=gdf['VEL'] * 30

            #gdf['VEL_STD']=VEL_STD 
            #gdf['VEL_STD']=gdf['VEL_STD']/ Total_days * 30
            gdf['VEL_STD']=gdf['VEL_STD']/Total_days * 30

        if VEL_Mode=='mean' and VEL_scale=='month':
            VEL=gdf[d_columns].mean(axis=1)
            VEL_STD=gdf[d_columns].std(axis=1)
            gdf['VEL']=VEL
            gdf['VEL']=gdf['VEL'] / Total_days * 30

            gdf['VEL_STD']=VEL_STD 
            gdf['VEL_STD']=gdf['VEL_STD']/ Total_days * 30

        if VEL_Mode=='linear' and VEL_scale==None:
            # VEL, VEL_STD=linear_VEL(gdf[d_columns], d_columns)
            # gdf['VEL']=VEL 
            # gdf['VEL_STD']=VEL_STD 
            print('')

        if VEL_Mode=='mean' and VEL_scale==None:
            VEL=gdf[d_columns].mean(axis=1)
            VEL_STD=gdf[d_columns].std(axis=1)
            gdf['VEL']=VEL
            gdf['VEL_STD']=VEL_STD
            
        
        return gdf

    gdf=calculate_VEL(gdf, VEL_scale=VEL_scale, Total_days=Total_days)
    # column_order = d_columns  # New columns added at the beginning
    # # # Insert new columns at the beginning of the list
    # columns_to_insert = ['CODE', 'x', 'y', 'VEL', 'VEL_STD']  # Inserted in this order
    # col_geo=['geometry']
    # column_order= columns_to_insert + d_columns+col_geo
    # gdf=gdf[column_order]
    
    ########
    # Extracting x and y coordinates from the 'geometry' column
    # gdf['x'] = gdf.apply(lambda row: row.geometry.x, axis=1)
    # gdf['y'] = gdf.apply(lambda row: row.geometry.y, axis=1)
    col_titles=['geometry','x', 'y', 'VEL', 'VEL_STD' ]+d_columns
    # Reset the index and convert it to a column
    gdf = gdf.reset_index(drop=True)

    # Rename the index column to "CODE"
    #gdf = gdf.rename(columns={'index': 'CODE'})
    gdf = gdf.reindex(columns=col_titles)
    
    # Get the CRS of the GeoDataFrame
    gdf.crs=crs_ini
    #gdf=gdf.dropna()
    import xml.etree.ElementTree as ET
    
    #####Correct displacement to a stable point############
    # try:
    #     # _, VEL_STD_series = linear_VEL(gdf[d_columns], d_columns)
        
    #     # # Filter points with the lowest standard deviation
    #     # VEL_STD_series = pd.Series(VEL_STD_series)
    #     # min_std_dev = VEL_STD_series.min()
    #     # filtered_indices = VEL_STD_series[VEL_STD_series == min_std_dev].index
        
    #     # # From the filtered points, select the point with the lowest average velocity
    #     # avg_velocities = gdf[d_columns].loc[filtered_indices].mean(axis=1)
    #     # reference_index = avg_velocities.idxmin()
    try: 
        
        
        if best_match_index is None:
            best_match_index=find_best_match(gdf)
        
        # # Compare each geometry in the GeoDataFrame with the target geometry
        # for best_match_index, row in gdf.iterrows():
        #     if row['geometry'] == geometry_point:
        #         return best_match_index
        
        print(best_match_index)
        
        
        import shutil
        # # Create the subdirectory again
        # os.makedirs(sub_dir_path, exist_ok=True)
        folder_directory = os.path.dirname(shapefile_path)
        base_filename = os.path.basename(shapefile_path)
        updated_dir=folder_directory+ "/"+ "updated_shapefiles"
        os.makedirs(updated_dir, exist_ok=True) 
        
        #   # Check if the subdirectory exists
        # if os.path.exists(folder_directory):
        #     # Remove the existing subdirectory
        #     shutil.rmtree(folder_directory)
        


        #####################
        # Save the updated and interpolated GeodataFrame to a new shapefile or return it
        updated_shapefile_path = updated_dir+ "/" + base_filename
        gdf.to_file(updated_shapefile_path, driver='ESRI Shapefile')
        
        # Correct the velocity of each month based on the reference point
        if best_match_index is not None:
            
            if spatial_ref==True:
                for col in gdf[d_columns]:
                    gdf[col] = gdf[col] - gdf[col].iloc[best_match_index]

            # Extract x, y values of the reference point
            # ref_point = gdf.geometry.iloc[best_match_index]
            # ref_point=ref_point
            # x, y = ref_point.x , ref_point.y
            x=gdf.x.iloc[best_match_index]
            y=gdf.y.iloc[best_match_index]
            VEL_ref, VEL_ref_STD=(gdf.VEL.iloc[best_match_index], gdf.VEL_STD.iloc[best_match_index])
            
            dic_data={
                        "ReferencePoint": {"Coordinates": (x, y)}, "VELDATA":{ "VEL": VEL_ref, "VEL_STD": VEL_ref_STD}}
            
            def dict_to_xml(data):
                from xml.etree.ElementTree import Element, SubElement, tostring
                from xml.dom import minidom
                root = Element('Data')

                for key, value in data.items():
                    point = SubElement(root, key)
                    for inner_key, inner_value in value.items():
                        if isinstance(inner_value, dict):
                            coord = SubElement(point, inner_key)
                            for coord_key, coord_value in inner_value.items():
                                coord_element = SubElement(coord, coord_key)
                                coord_element.text = str(coord_value)
                        else:
                            element = SubElement(point, inner_key)
                            element.text = str(inner_value)

                return minidom.parseString(tostring(root)).toprettyxml(indent="   ", encoding='UTF-8')
            xml_name=updated_shapefile_path + ".xml"
            # Using the function to write the dictionary to an XML file
            xml_data =dict_to_xml(dic_data)
            xml_data_str = xml_data.decode('UTF-8')
            
            # Saving the XML to a file
            with open(xml_name, 'w' ) as file:
                file.write(xml_data_str)
            
            # Create an XML with x, y values
            # metadata = ET.Element("Metadata")
            # ref_point_elem = ET.SubElement(metadata, "ReferencePoint")
            # x_elem = ET.SubElement(ref_point_elem, "x")
            # x_elem.text = str(x)
            # y_elem = ET.SubElement(ref_point_elem, "y")
            # y_elem.text = str(y)
            # Creating the XML structure
            # metadata = ET.Element("Metadata")
            # ref_point_elem = ET.SubElement(metadata, "ReferencePoint")
            # x_elem = ET.SubElement(ref_point_elem, "x")
            # #x_elem.text = str(x)
            # x_elem.set('value', str(x))  # Setting the 'value' attribute
            # y_elem = ET.SubElement(ref_point_elem, "y")
            # #y_elem.text = str(y)
            # y_elem.set('value', str(y))  # Setting the 'value' attribute

            
            # # Write the XML to a file
            # tree = ET.ElementTree(metadata)
            # tree.write(updated_shapefile_path + ".xml")
            
            # with open(shapefile_path + ".xml", "wb") as file:
            #     tree.write(file, encoding='utf-8', xml_declaration=True)
            
            print(f'ReferencePoint x, y: { x, y}, VEL, VEL_STD: { VEL_ref, VEL_ref_STD}')

    except Exception as e:
        print(f"An error occurred: {e}")

    
    update_dir_shapefile=updated_shapefile_path
   
    
    
    # with open(updated_shapefile_path[:-4] + ".cpg", "w") as f:
    #     f.write("UTF-8")
    return gdf , update_dir_shapefile

##################################################################################################
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from tqdm import tqdm

def merge_geodataframes_generator(gdfs):
    total_gdfs = len(gdfs)

    for i, gdf in enumerate(gdfs):
        # Extracting x, y coordinates from the geometry on-the-fly without adding to the dataframe
        # xy = gdf.geometry.apply(lambda geom: (geom.x, geom.y))
        # gdf['x'], gdf['y'] = zip(*xy)
        gdf['x'] = gdf.geometry.x
        gdf['y'] = gdf.geometry.y

        # Drop the 'geometry' column
        if 'geometry' in gdf:
            gdf = gdf.drop(columns='geometry')

        description = f"Merging GeoDataFrame {i + 1}/{total_gdfs}"
        yield gdf, description



    # Correcting the date format in column names and filtering columns
def correct_date_format(col):
    if col.startswith('D'):
        date_part = col[1:]
        try:
            # Check if the date part is in valid date format
            datetime.strptime(date_part, '%Y%m%d')
            return col
        except ValueError:
            # If not, return None to indicate an invalid date format
            return None
    else:
        # Return non-date columns as is
        return col


def merge_geodataframes(gdfs):
    ############3
    
    ################
    gdf_gen = merge_geodataframes_generator(gdfs)
    
    # Instead of merging iteratively, collect all dataframes and concatenate at once
    dfs_to_merge = [gdf for gdf, _ in tqdm(gdf_gen, total=len(gdfs), desc="Preparing GeoDataFrames")]

    # Concatenate all dataframes
    merged_df = pd.concat(dfs_to_merge, axis=0, ignore_index=True)

    # Drop duplicates based on x, y
    merged_df = merged_df.drop_duplicates(subset=['x', 'y'])

    # Create the final GeoDataFrame with one geometry column
    geometry = [Point(xy) for xy in zip(merged_df.x, merged_df.y)]
    final_gdf = gpd.GeoDataFrame(merged_df, geometry=geometry)
    
    
    return final_gdf


###################################################################

import os
import re
from datetime import datetime
from os import listdir
from os.path import isfile, join
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity as ssim


def mask_raster(dem_array=None, mask_path=None, no_data_value=np.nan, scatter_x=None, scatter_y=None):
    """
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
    
    
    
    
    """
    from scipy.ndimage import zoom

    # Read the binary mask
    with rasterio.open(mask_path, 'r') as mask_src:
        mask_array = mask_src.read(1)

    # If the shapes don't match and dem_array is provided, resize mask_array
    if dem_array is not None and dem_array.shape[:2] != mask_array.shape:
        y_scale = dem_array.shape[0] / mask_array.shape[0]
        x_scale = dem_array.shape[1] / mask_array.shape[1]
        mask_array = zoom(mask_array, (y_scale, x_scale))

    #################3Threshold mask_array
    mask_array = (mask_array >= 0.5).astype(np.int32)

    # Check if scatter_x and scatter_y are provided
    scatter_x_masked, scatter_y_masked = None, None
    if scatter_x is not None and scatter_y is not None:
        scatter_x = np.asarray(scatter_x)
        scatter_y = np.asarray(scatter_y)
        # Ensure scatter coordinates are within the bounds of the mask_array
        scatter_x_clipped = np.clip(scatter_x, 0, mask_array.shape[1]-1).astype(int)
        scatter_y_clipped = np.clip(scatter_y, 0, mask_array.shape[0]-1).astype(int)
        
        valid_indices = mask_array[scatter_y_clipped, scatter_x_clipped] == 1
        scatter_x_masked = scatter_x[valid_indices]
        scatter_y_masked = scatter_y[valid_indices]

    # If dem_array is provided, mask it
    masked_array = None
    if dem_array is not None:
        if dem_array.ndim == 2:
            masked_array = np.where(mask_array == 1, dem_array, no_data_value)
        else:
            masked_array = np.where(mask_array[:, :, np.newaxis] == 1, dem_array, no_data_value)

        # Handle casting to original data type
        original_dtype = dem_array.dtype
        if np.issubdtype(original_dtype, np.integer):
            int_no_data = -9999
            masked_array = np.where(np.isnan(masked_array), int_no_data, masked_array).astype(original_dtype)
        else:
            masked_array = masked_array.astype(original_dtype)

    if dem_array is not None and scatter_x is not None and scatter_y is not None:
        return masked_array, scatter_x_masked, scatter_y_masked
    elif dem_array is not None:
        return masked_array
    else:
        return scatter_x_masked, scatter_y_masked


# Correcting the date format in column names and filtering columns
def correct_date_format(col):
    if col.startswith('D'):
        date_part = col[1:]
        try:
            # Check if the date part is in valid date format
            datetime.strptime(date_part, '%Y%m%d')
            return col
        except ValueError:
            # If not, return None to indicate an invalid date format
            return None
    else:
        # Return non-date columns as is
        return col

def Optical_flow_akhdefo(input_dir="", output_dir="", AOI=None, zscore_threshold=2 , ssim_thresh=0.75, image_resolution='3125mm', interpolate=None, show_figure=False, point_size=2,
                          dem_path="", smoothing_kernel_size=11, Vegetation_mask=None, VEL_scale='year', VEL_Mode='linear',
                            good_match_option=0.75, hillshade_option=True, shapefile_output=False, max_triplet_interval=24,
                            pixel_size=20,num_chunks=10,overlap_percentage=0 , pyr_scale=0.5, levels=15, winsize=32,iterations= 7, poly_n=7,poly_sigma= 1.5, flags=1, 
                            master_reference='single', selection_Mode='triplet' , start_date=None , end_date=None , krig_method='ordinary', spatial_ref=False):
   
    """
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

    """


    def detect_keypoints(image):
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)
        return keypoints, descriptors


    def compare_images(image1, image2):
        # Rasterio reads data as (bands, height, width)
        # OpenCV expects data as (height, width, channels)
        # So we need to transpose the data
        # image1 = np.transpose(image1, [1, 2, 0])
        # image2 = np.transpose(image2, [1, 2, 0])
        # Convert the images to grayscale
        if image1.shape[2] < 3:
            gray1 = image1[:,:,0]  # Take only the first channel
        else:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        if image2.shape[2] < 3:
            gray2 = image2[:,:,0]  # Take only the first channel
        else:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        # image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        # Calculate SSIM
       
        mask1=gray1[gray1==0] 
        mask2=gray2[gray2==0]
        
        ssim_index, ssim_map = ssim(gray1, gray2, full=True)
        ssim_map[mask1]=np.nan
        ssim_map[mask2]=np.nan
        ssim_map[ssim_map>0.95]=np.nan
        # Compute the structural similarity index (SSIM) between the two images
        return ssim_map


    def match_features(image1, image2, descriptor1, descriptor2, zscore_threshold=zscore_threshold, good_match_option=good_match_option):
        good_matches = [] # Initialize an empty list for good_matches
    
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptor1, descriptor2, k=2)

        # Calculate distances for all matches
        distances = [m.distance for m, n in matches]
        
        # Calculate mean and standard deviation of distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Define a threshold based on the Z-score
        z_score_threshold = zscore_threshold

        #mean_distance + z_score_threshold * std_distance
        
        # Filter matches based on the Z-score
        if good_match_option is not None:
            good_matches = [m for m, n in matches if m.distance < good_match_option * n.distance]


        else:

            good_matches = [m for m, n in matches if m.distance < mean_distance + z_score_threshold * std_distance]

    
        return good_matches

    import cv2
    import numpy as np
    from scipy.stats import zscore
    from skimage.filters import gaussian

    def calculate_optical_flow(image1, image2, zscore_threshold=2.0, ssim_thresh=ssim_thresh, pyr_scale=0.5, levels=15, winsize=32,iterations= 3, poly_n=5,poly_sigma= 1.5, flags=1):
        # Rasterio reads data as (bands, height, width)
        # OpenCV expects data as (height, width, channels)
        # So we need to transpose the data
        # image1 = np.transpose(image1, [1, 2, 0])
        # image2 = np.transpose(image2, [1, 2, 0])
        
        # Convert the images to grayscale

        if image1.shape[2] < 3:
            gray1 = image1[:,:,0]  # Take only the first channel
        else:
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        if image2.shape[2] < 3:
            gray2 = image2[:,:,0]  # Take only the first channel
        else:
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

        # Confirm that gray1 and gray2 are both 2D (grayscale) images of the same size
        assert gray1.ndim == 2, "gray1 is not a grayscale image"
        assert gray2.ndim == 2, "gray2 is not a grayscale image"
        assert gray1.shape == gray2.shape, "gray1 and gray2 are not the same size"
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None,pyr_scale=pyr_scale, levels=levels, winsize=winsize,iterations= iterations, poly_n=poly_n,poly_sigma= poly_sigma, flags=flags)

        ssim=compare_images(image1, image2)
    

        flowx=flow[..., 0]
        flowy=flow[..., 1]
        
        flowx[ssim <ssim_thresh] = 0
        flowy[ssim <ssim_thresh] = 0
        
        # Compute the magnitude and angle of the 2D vectors
        magnitude, angle = cv2.cartToPolar(flowx, flowy)
        #magnitude = np.sqrt(flowx**2 + flowy**2)
        
        

        # Compute z-scores for the x_flow
        z_scores_x = zscore(flow[..., 0], axis=None)

        # Compute z-scores for the y_flow
        z_scores_y = zscore(flow[..., 1], axis=None)
        
        # Create a mask for vectors with a z-score less than the threshold
        mask_y = np.abs(z_scores_y) < zscore_threshold
        mask_x = np.abs(z_scores_x) < zscore_threshold
        
        # Zero out the vectors where the mask is False
        flowx[~mask_x] = 0
        flowy[~mask_x] = 0

        flowx[~mask_y] = 0
        flowy[~mask_y] = 0

        
        
        magnitude[~mask_x] = 0
        magnitude[~mask_y] = 0
        
        # Post-process magnitude to make it negative when flowx is negative
        #magnitude = np.where(flowx < 0, -magnitude, magnitude)
       
        return magnitude, flowx, flowy, ssim

    def filter_velocity(flow, good_matches, keypoints1, keypoints2):
        points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        velocity = []
        for i in range(len(points1)):
            velocity.append(flow[int(points1[i][1]), int(points1[i][0])])
        return np.array(velocity), points1, points2

    def calculate_velocity_displacement(velocity, flowx, flowy, time_interval, conversion_factor):
        if time_interval == 0:
            raise ValueError("Time interval must not be zero.")
        
        
        if selection_Mode=='pair':
            nframes=2
        elif selection_Mode=='triplet':
            nframes=3
        velocity= velocity/nframes * conversion_factor/time_interval
        
        flowx = flowx/nframes * conversion_factor/time_interval
        
        flowy = flowy/nframes * conversion_factor/time_interval
        
        
        

        return velocity, flowx, flowy


    import re

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def separate_floats_letters(input_string):
        floats = re.findall(r'\d+\.\d+|\d+', input_string)
        letters = re.findall(r'[a-zA-Z]+', input_string)
        return letters, floats

    input_string = image_resolution
    unit, img_res = separate_floats_letters(input_string)

    import earthpy.plot as ep
    import earthpy.spatial as es
    import matplotlib.pyplot as plt
    import numpy as np
    # Import necessary packages
    import rasterio as rio
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    def calculate_hillshade(dem_file_path, hillshade_option=True):
        # Open the raster data

        with rio.open(dem_file_path) as dem_src:
            dem = dem_src.read(1)

        # Calculate hillshade from the DEM
        if hillshade_option==True:
            hillshade = es.hillshade(dem)
        else:
            hillshade=dem
        return hillshade

    
    
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
    
    
    def plot_reference_point(xml_file_path, list_figures=[], figure_paths=""):
        """
        Extracts x, y coordinates from the given XML file and plots them on the provided axis.

        Args:
            - xml_file_path (str): Path to the XML file containing the reference point data.
            - ax (matplotlib.axes.Axes, optional): Axes on which to plot. If None, a new figure and axes will be created.

        Returns:
            - matplotlib.axes.Axes: The axes on which the data was plotted.
        
        """
        if xml_file_path is not None:
            x, y, vel, vel_std = get_values_from_xml(xml_file_path)
            for idx, (fig, axs) in enumerate(list_figures):
                #print("x, y", x, y)
                for ax in axs.ravel():
                    # We will annotate the point (x=5, y=sin(5))
                    ax.scatter(x, y, label="REF", color='b', marker='s')
                    ax.legend()
            # 4. Save each figure to a PNG file
            for idx, (fig, _) in enumerate(list_figures):
                fig.savefig(figure_paths[idx]+".png")
            
        else:
            raise ValueError("ReferencePoint x or y not found in the provided XML file.")

        
       


    

    def plot_velocity_displacement(image1, image2, velocity, flowx, flowy, points1, points2, date1, date2, pdf_filename=None,
                                    time_interval=1, show_figure=False, unit='unit',
                                     s=10, bounds=[10,10,10,10], dem_file=dem_path, hillshade_option=hillshade_option):
        
        
        if hillshade_option==True:
            
            hillshade=calculate_hillshade(dem_file , hillshade_option=hillshade_option)
        
        elif hillshade_option==False:
            # if len(image1.shape) == 2:  # It's already grayscale
            #     gray=image1
                
            # elif len(image1.shape) == 3 and image1.shape[2] == 3:  # It's RGB
            #     # Convert RGB to grayscale using the weighted method
            #     gray = 0.299 * image1[:, :, 0] + 0.587 * image1[:, :, 1] + 0.114 * image1[:, :, 2]
            #     gray=gray.astype(np.uint8)
            
            # hillshade=calculate_hillshade(gray , hillshade_option=hillshade_option)
            hillshade=image1
                
           
            

        #image1=image1.transpose([1, 2, 0])  
        # image size in pixels
        image_width = image1.shape[1]
        image_height = image1.shape[0]

        # image bounds in geographic coordinates
        geo_bounds = {
            'left': bounds[0],
            'right': bounds[1],
            'bottom': bounds[2],
            'top': bounds[3],
        }


        pixels = points1
        
            
        # convert pixel coordinates to geographic coordinates
        geo_coords = [(geo_bounds['left'] + (x / image_width) * (geo_bounds['right'] - geo_bounds['left']),
                    geo_bounds['top'] - (y / image_height) * (geo_bounds['top'] - geo_bounds['bottom'])) for x, y in pixels]

        # separate the coordinates for plotting
        lons, lats = zip(*geo_coords)

      
        def normalize(data, vmin=None, vmax=None ,cmap=None):
            import matplotlib.colors as mcolors
            import numpy as np
            
            if vmin is not None and vmax is not None:
                vmin, vmax = np.nanmin(data), np.nanmax(data)
            if vmin < 0 and vmax > 0:
                # Data has negative values, use TwoSlopeNorm
                norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                if cmap is None:
                    cmap = 'rainbow'
                return cmap, norm
            else:
                # Data has no negative values, use standard normalization
                norm= mcolors.Normalize(vmin=vmin, vmax=vmax)
                if cmap is None:
                    cmap = 'viridis'
                
                return cmap, norm

        
        import cmocean

        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10), sharey=False)

        # Plot flowx
        #ep.plot_rgb(image1, ax=axes[0, 0], title=f'Disp-X({unit}) - {date1} to {date2}', extent=bounds)
        axes[0, 0].imshow(hillshade, cmap='gray', extent=bounds)
        minmax = np.max(np.abs(flowx))
        d=normalize(flowx, cmap='rainbow', vmin=-minmax, vmax=minmax)
        cmap, norm=d

        flowx_scatter = axes[0, 0].scatter(lons, lats, c=flowx, s=s, cmap=cmap, norm=norm)
       
        # Create colorbar for flowx
        flowx_colorbar_axes = make_axes_locatable(axes[0, 0]).append_axes("right", size="5%", pad=0.01)
        fig.colorbar(flowx_scatter, cax=flowx_colorbar_axes, orientation="vertical").set_label(unit, labelpad=0.5)
        axes[0,0].set_title(f'Disp-X({unit}) - {date1} to {date2}')
        # Plot flowy
        #ep.plot_rgb(image1, ax=axes[0, 1], extent=bounds, title=f'Disp-Y({unit}) - {date1} to {date2}')
        axes[0, 1].imshow(hillshade, cmap='gray', extent=bounds)
        minmax = np.max(np.abs(flowy))
        d=normalize(flowy, cmap='rainbow', vmin=-minmax, vmax=minmax)
        cmap, norm=d
        flowy_scatter = axes[0, 1].scatter(lons, lats, c=flowy, s=s, cmap= cmap, norm=norm)
        axes[0, 1].set_title(f'Disp-Y({unit}) - {date1} to {date2}')

        # Create colorbar for flowy
        flowy_colorbar_axes = make_axes_locatable(axes[0, 1]).append_axes("right", size="5%", pad=0.01)
        fig.colorbar(flowy_scatter, cax=flowy_colorbar_axes, orientation="vertical").set_label(unit, labelpad=0.5)

        # Plot Velocity Magnitude
        #ep.plot_rgb(image1, ax=axes[1, 0], extent=bounds, title=f'Velocity - {date1} to {date2}')
        
        axes[1, 0].imshow(hillshade, cmap='gray', extent=bounds)
        minmax = np.max(np.abs(velocity))
        min_v= np.nanmin(velocity)

        d=normalize(velocity, cmap='rainbow', vmin=min_v, vmax=minmax)
        cmap, norm=d
        velocity_scatter = axes[1, 0].scatter(lons, lats, c=velocity, s=s,  cmap=cmap, norm=norm)
        axes[1, 0].set_title(f'Velocity - {date1} to {date2}')
        # Create colorbar for velocity
        velocity_colorbar_axes = make_axes_locatable(axes[1, 0]).append_axes("right", size="5%", pad=0.01)
        fig.colorbar(velocity_scatter, cax=velocity_colorbar_axes, orientation="vertical").set_label(f'{(unit)}/{str(time_interval)}days', labelpad=0.5)

        # Plot Velocity Direction
        #ep.plot_rgb(image1, ax=axes[1, 1], extent=bounds, title=f'Velocity Direction - {date1} to {date2}')
        axes[1, 1].imshow(hillshade, cmap='gray', extent=bounds)
        velocity_direction = (360 - np.arctan2(flowy, flowx) * 180 / np.pi + 90) % 360
        
        d=normalize(velocity_direction, cmap=cmocean.cm.phase, vmin=0, vmax=360)
        cmap, norm=d
        velocity_direction_scatter = axes[1, 1].scatter(lons, lats, c=velocity_direction, s=s, cmap= cmap, norm=norm)
        axes[1, 1].set_title(f'Velocity Direction - {date1} to {date2}')

        # Create colorbar for velocity direction
        velocity_direction_colorbar_axes = make_axes_locatable(axes[1, 1]).append_axes("right", size="5%", pad=0.01)
        fig.colorbar(velocity_direction_scatter, cax=velocity_direction_colorbar_axes, orientation="vertical").set_label('degrees')
        

        # Set the extent of the axes
        axes[0, 0].set_xlim([bounds[0], bounds[1]])
        axes[0, 0].set_ylim([bounds[2], bounds[3]])
        axes[0, 1].set_xlim([bounds[0], bounds[1]])
        axes[0, 1].set_ylim([bounds[2], bounds[3]])
        axes[1, 0].set_xlim([bounds[0], bounds[1]])
        axes[1, 0].set_ylim([bounds[2], bounds[3]])
        axes[1, 1].set_xlim([bounds[0], bounds[1]])
        axes[1, 1].set_ylim([bounds[2], bounds[3]])
        
        
        # Automatically adjust subplot parameters for a tight layout
        plt.tight_layout()

        if pdf_filename:
            plt.savefig(pdf_filename)
        
        if show_figure==False:
            plt.close()


    
        flowx_scatter=flowx_scatter.get_offsets()
        x_data = flowx_scatter[:, 0]
        y_data = flowx_scatter[:, 1]

        

        return  lons, lats , pixels[:, 0], pixels[:, 1], (fig,axes)

    def extract_date_from_filename(filename):
        try:
            # Searching for a date in the format 'YYYY-MM-DD' or 'YYYYMMDD'
            match = re.search(r'(\d{4}-\d{2}-\d{2})|(\d{8})', filename)
            if match is not None:
                date_str = match.group()
                # Determine the date format
                date_format = '%Y%m%d' if '-' not in date_str else '%Y-%m-%d'
                # Parse the date string
                date_obj = datetime.strptime(date_str, date_format).date()
                return date_obj.strftime('%Y-%m-%d')
            else:
                print("No date string found in filename.")
                return None
        except ValueError:
            print(f"Date string '{date_str}' in filename is not in expected format.")
            return None



    def mean_of_arrays(array1, array2):
        # Determine the size of the larger array
        max_size = max(array1.shape, array2.shape)

        # Use np.pad to extend the smaller array with zeros
        array1 = np.pad(array1, (0, max_size[0] - array1.shape[0]))
        array2 = np.pad(array2, (0, max_size[0] - array2.shape[0]))
       # Compute the mean of the two arrays element-wise
        mean_array = np.nanmean([array1, array2], axis=0)
        

        return mean_array

    

    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    def replace_nan_with_nearest(x, y, z, width, height):
        try:
            # Create a 2D grid of coordinates based on the x, y values
            xi = np.linspace(np.nanmin(x), np.nanmax(x), width)
            yi = np.linspace(np.nanmin(y), np.nanmax(y), height)
            xi, yi = np.meshgrid(xi, yi)

            # Flatten the coordinates grid and build a KDTree
            flattened_coordinates = np.column_stack((xi.ravel(), yi.ravel()))
            tree = cKDTree(flattened_coordinates)

            # Query the tree for nearest neighbors to each point in x, y
            _, indices = tree.query(np.column_stack((x, y)))

            # Replace NaNs with z values at these indices
            #zi = np.full_like(xi, np.nan)
            
            zi = np.zeros_like(xi)
            np.put(zi, indices, z)
        
            return zi

        except Exception as e:
            print("An error occurred:", str(e))
            return None
        
        return zi


    

    

    def save_xyz_as_geotiff(x, y, z, filename, reference_raster, shapefile=None, interpolate=None, 
    smoothing_kernel_size=smoothing_kernel_size, Vegetation_mask=Vegetation_mask):
        try:
            # Get the CRS, width, height, and transform from the reference raster
            with rasterio.open(reference_raster) as src:
                crs = src.crs
                width = src.width
                height = src.height
                transform = src.transform
                bounds = src.bounds
                x_min = bounds.left
                x_max = bounds.right
                y_min = bounds.bottom
                y_max = bounds.top
                pixel_size_x = src.res[0]
                pixel_size_y = src.res[1]
                

            # Create a 2D grid of coordinates based on the x, y values
            xi = np.linspace(np.min(x), np.max(x), width)
            yi = np.linspace(np.min(y), np.max(y), height)
            xi, yi = np.meshgrid(xi, yi)

            # Create an array of the same size as the x, y grid filled with NaN
            #zi = np.full_like(xi, yi, np.nan)
            #zi = np.zeros_like(xi)

            
                

            if interpolate is not None:
                # Interpolate z values onto the new grid
                zi = griddata((x, y), z, (xi, yi), method=interpolate, rescale=True)

                # Replace interpolated values outside the range with mean of initial z values
                z_min = np.min(z)
                z_max = np.max(z)
                zi[zi < z_min] = np.mean(z)
                zi[zi > z_max] = np.mean(z)

                # Find the indices of interpolated points exceeding the data range
                out_of_range_indices = np.logical_or(xi < np.min(x), xi > np.max(x)) | np.logical_or(yi < np.min(y), yi > np.max(y))

                # Replace out-of-range interpolated points with the mean of valid data points
                zi_valid = zi[~out_of_range_indices]
                mean_valid = np.nanmean(zi_valid)
                zi = np.where(out_of_range_indices, mean_valid, zi)
            if interpolate is None:
                zi=replace_nan_with_nearest(x, y, z, width, height)
                # # Flatten the coordinates grid and build a KDTree
                # flattened_coordinates = np.column_stack((xi.ravel(), yi.ravel()))
                # tree = cKDTree(flattened_coordinates)

                # # Query the tree for nearest neighbors to each point in x, y
                # _, indices = tree.query(np.column_stack((x, y)))

                # # Replace NaNs with z values at these indices
                # np.put(zi, indices, z)
                

                
            
             # Apply low-pass filter
            if smoothing_kernel_size is not None:
                #zi_initial=zi
                zi = gaussian(zi, sigma=smoothing_kernel_size )  # Adjust sigma according to your desired smoothing strength
                #if interpolate is None:
                    #zi[zi_initial == 0] = np.nan

            if shapefile is not None:
                # Load shapefile, convert it to the correct CRS and get its geometry
                gdf = gpd.read_file(shapefile).to_crs(crs)
                shapes = gdf.geometry.values

                # Generate a mask from the shapes
                mask = geometry_mask(shapes, transform=transform, out_shape=zi.shape, invert=False, all_touched=True)

                # Apply the mask to the interpolated data
                zi = np.where(mask, np.nan, zi)

            if Vegetation_mask is not None:
                zi=mask_raster(zi, Vegetation_mask )


            # Define the profile
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': zi.dtype,
                'crs': crs,
                'transform': transform,
                'nodata': np.nan,  # specify the nodata value
            }

            # Write to a new .tif file
            with rasterio.open(filename + ".tif", 'w', **profile) as dst:
                dst.write(zi, 1)

        except Exception as e:
            print("An error occurred while creating the GeoTIFF:")
            print(e)

        return zi
    
    import os
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds
    from shapely.geometry import box
    from rasterio.mask import mask
    import geopandas as gpd
    def crop_to_overlap(folder_path, start_date=start_date, end_date=end_date):
        
        if start_date is not None and end_date is not None:
            from datetime import datetime, timedelta

            # User provided start and end date strings
            start_date = start_date
            end_date = end_date

            # Converting the strings to datetime objects
            start = datetime.strptime(start_date, '%Y%m%d')
            end = datetime.strptime(end_date, '%Y%m%d')

            # Generating the list of dates
            date_list = []
            current_date = start
            while current_date <= end:
                date_list.append(current_date.strftime('%Y-%m-%d'))
                current_date += timedelta(days=1)
              
        ##############
        
        image_files = sorted(os.listdir(folder_path))
        
        ##Filter images based on start and end date
        if start_date is not None and end_date is not None:
            image_files = [item for item in image_files if item.split('.')[0] in date_list]
        else:
            image_files
        #########################
        
        valid_extensions = ['.tif', '.jpg', '.png', '.bmp', '.tiff']
        image_path_list=[]
        bound_list=[]

        # Calculate mutual overlap
        overlap_box = None
        for file in image_files:
            if os.path.splitext(file)[1] in valid_extensions:
                image_path = os.path.join(folder_path, file)
                image_path_list.append(image_path)
                with rasterio.open(image_path) as src:
                    meta=src.meta
                    bounds = src.bounds
                    bound_list.append(bounds)
                    image_box = box(*bounds)
                    if overlap_box is None:
                        overlap_box = image_box
                    else:
                        overlap_box = overlap_box.intersection(image_box)

        # Read images and crop to mutual overlap
        cropped_images = []
        keypoints=[]
        descriptors=[]
        
        for image_path in image_path_list:
            with rasterio.open(image_path) as src:
                overlap_window = from_bounds(*overlap_box.bounds, transform=src.transform)
                cropped_image = src.read(window=overlap_window)
                maskNan=cropped_image[cropped_image==0]
                meta=src.meta
                 # Rasterio reads data as (bands, height, width)
                #OpenCV expects data as (height, width, channels)
                #So we need to transpose the data
                cropped_image = np.transpose(cropped_image, [1, 2, 0])
                cropped_images.append(cropped_image)
                kp, des = detect_keypoints(cropped_image)
                keypoints.append(kp)
                descriptors.append(des)
        
        if start_date is not None and end_date is not None:     
            filtered_dates=date_list
        else:
            filtered_dates=image_files
        #print("Cropped {} images.".format(len(cropped_images)))
        return cropped_images, bound_list, keypoints, descriptors, image_path_list, filtered_dates, meta, maskNan


    ##########################################################temp
   
    





###################################

    import os
    from datetime import datetime

    import numpy as np
    import rasterio
    from tqdm import tqdm
    from akhdefo_functions import Auto_Variogram
    

    def feature_matching(folder_path=input_dir, output_dir=output_dir, zscore_threshold=zscore_threshold, 
    AOI=AOI, conversion_factor=float(img_res[0]), ssim_thresh=ssim_thresh, Vegetation_mask=Vegetation_mask, 
    VEL_scale=VEL_scale, VEL_Mode=VEL_Mode, shapefile_output=shapefile_output, 
    smoothing_kernel_size=smoothing_kernel_size, pixel_size=pixel_size,num_chunks=num_chunks,overlap_percentage=overlap_percentage, 
    pyr_scale=pyr_scale, levels=levels, winsize=winsize,iterations= iterations, poly_n=poly_n,poly_sigma= poly_sigma, flags=flags,
    master_reference=master_reference, selection_Mode=selection_Mode , start_date=start_date , end_date=end_date):
        
        folder_path = folder_path
        
        images, bound_list, keypoints, descriptors, image_path_list, filtered_dates, meta, maskNan = crop_to_overlap(folder_path)
        image_files = [filename for filename in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, filename))]

        image_files = sorted(image_files)
        if start_date is not None and end_date is not None:
            image_files = [item for item in image_files if item.split('.')[0] in filtered_dates]

        #image_files = sorted(os.listdir(folder_path))
        # images = []
        # keypoints = []
        # descriptors = []
        # bound_list=[]
        # # List of valid extensions
        # valid_extensions = ['.tif', '.jpg', '.png', '.bmp']
        # image_path_list=[]
        # for file in image_files :
        #     if os.path.splitext(file)[1] in valid_extensions:
        #         image_path = os.path.join(folder_path,file)
        #         image_path_list.append(image_path)
        #         with rasterio.open(image_path) as src:
        #             image = np.dstack([src.read(i) for i in src.indexes])  # This line stacks the bands of the image
        #             bounds=src.bounds
        #             bound_list.append(bounds)
        #             #image=src.read(1)
        #         images.append(image)
        #         kp, des = detect_keypoints(image)
        #         keypoints.append(kp)
        #         descriptors.append(des)


        
        ######################
        mean_vel_list=[]
        mean_flowx_list=[]
        mean_flowy_list=[]
        pointx_list=[]
        pointsy_list=[]
        dates_names_list=[]

        lf=0

        if Vegetation_mask is not None:
            from scipy.ndimage import zoom
            mask_file=rasterio.open(Vegetation_mask)
            mask_data=mask_file.read(1)
            # Ensure mask_data is boolean (0 or 1)
            
        dem_src= rio.open(dem_path)  
        dem_crs=dem_src.crs

        geodfs_x = []
        geodfs_y = []
        geodfs_v = []
        list_figs=[]
        figure_paths=[]
        
        ############################
        if selection_Mode=='pair':
            loop_flag=1
        elif selection_Mode=='triplet':
            loop_flag=2
        
        ############################
        for i in tqdm(range(0, len(images)-loop_flag), desc="Processing"):
            bound=bound_list[i]
            image1 = images[i]
            image2 = images[i + 1]
            image3=images[i + loop_flag]
            
           
                
            
            
            ############################
            if selection_Mode=='pair':
                image3=image2
            elif selection_Mode=='triplet':
                image3=image3
            
            ############################
            keypoints1 = keypoints[i]
            keypoints2 = keypoints[i + 1]
            keypoints3 = keypoints[i + loop_flag]
            descriptors1 = descriptors[i]
            descriptors2 = descriptors[i + 1]
            descriptors3 = descriptors[i + loop_flag]

            if Vegetation_mask is not None:
                # If the shapes don't match and dem_array is provided, resize mask_array
                if image1 is not None and image1.shape[:2] != mask_data.shape:
                    y_scale = image1.shape[0] / mask_data.shape[0]
                    x_scale = image1.shape[1] / mask_data.shape[1]
                    mask_data = zoom(mask_data, (y_scale, x_scale))

                    # Threshold mask_array
                    mask_data = (mask_data >= 0.5).astype(np.int32)
                    mask_data = mask_data.astype(bool)
                
                # Apply the mask for each band using broadcasting
                image1[~mask_data, :] = 0
                image2[~mask_data, :] = 0
                image3[~mask_data, :] = 0

                # plt.imshow(image1)
                # plt.show()



            # descriptors1 and descriptors2 are assumed to be numpy arrays
            descriptors12 = np.concatenate((descriptors1, descriptors2), axis=0)
            descriptors13 = np.concatenate((descriptors1, descriptors3), axis=0)

            keypoints12 = np.concatenate((keypoints1, keypoints2), axis=0)
            keypoints13 = np.concatenate((keypoints1, keypoints3), axis=0)

            good_matches12 = match_features(image1, image2, descriptors12, descriptors13, good_match_option=good_match_option)
            good_matches13 = match_features(image1, image3, descriptors12, descriptors13, good_match_option=good_match_option)

            flow12, flowx12, flowy12 , ssim1= calculate_optical_flow(image1, image2, zscore_threshold=zscore_threshold, ssim_thresh=ssim_thresh ,
                                                              pyr_scale=pyr_scale, levels=levels, winsize=winsize,iterations= iterations, poly_n=poly_n,poly_sigma= poly_sigma, flags=flags)
            flow13, flowx13, flowy13 , ssim2= calculate_optical_flow(image1, image3, zscore_threshold=zscore_threshold, ssim_thresh=ssim_thresh,
                                                              pyr_scale=pyr_scale, levels=levels, winsize=winsize,iterations= iterations, poly_n=poly_n,poly_sigma= poly_sigma, flags=flags)

            flow=mean_of_arrays(flow12, flow13)
            flowx=mean_of_arrays(flowx12, flowx13)
            flowy=mean_of_arrays(flowy12,flowy13)
            
            ssim=(ssim1+ssim2)/2.0

            vel, points1_i, points2 = filter_velocity(flow, good_matches12, keypoints12, keypoints13)
            flowx, points1_i, points2 = filter_velocity(flowx, good_matches12, keypoints12, keypoints13)
            flowy, points1_i, points2 = filter_velocity(flowy, good_matches12, keypoints12, keypoints13)
            
            # vel13, points1, points3 = filter_velocity(flow13, good_matches13, keypoints12, keypoints13)
            # flowx13, points1, points3 = filter_velocity(flowx13, good_matches13, keypoints12, keypoints13)
            # flowy13, points1, points3 = filter_velocity(flowy13, good_matches13, keypoints12, keypoints13)

            # points12 = np.concatenate((points1_i[:,0], points2[:,1]), axis=0)
            # points13 = np.concatenate((points1[:,0], points3[:,1]), axis=0)

            # print(points12.shape)
            # print(points13.shape)

            #Extract All dates to List for Later use
            # if master_reference:
            #     date1 = (extract_date_from_filename(image_files[0])).replace("-", "")
            # else:
            #     date1 = (extract_date_from_filename(image_files[lf])).replace("-", "")
            date1 = (extract_date_from_filename(image_files[lf])).replace("-", "")
            start_date_init=(extract_date_from_filename(image_files[0])).replace("-", "")
            date2 = (extract_date_from_filename(image_files[lf + loop_flag])).replace("-", "")
            date3= (extract_date_from_filename(image_files[lf + loop_flag])).replace("-", "")
            
             ############################
            if selection_Mode=='pair':
                date3=date2
            elif selection_Mode=='triplet':
                date3=date3
            
            ############################
            
            lf=lf+1

            time_interval_1_2 = (datetime.strptime(date2, '%Y%m%d') - datetime.strptime(date1, '%Y%m%d')).days
            time_interval_1_3 = (datetime.strptime(date3, '%Y%m%d') - datetime.strptime(date1, '%Y%m%d')).days
            if time_interval_1_2 == 0:
                print(f"Skipping computation for {date1} to {date2} as the time interval is zero.")
                continue  # Skip the rest of this loop iteration
            
            if time_interval_1_2 > max_triplet_interval:
                print(f"Skipping computation for {date1} to {date2} as the time interval is larger than {max_triplet_interval} days.")
                continue  # Skip the rest of this loop iteration
            
            if time_interval_1_3 > max_triplet_interval:
                print(f"Skipping computation for {date1} to {date3} as the time interval is larger than {max_triplet_interval} days.")
                continue  # Skip the rest of this loop iteration
        
            
            conversion_factor = float(img_res[0])  # 1 pixel = 0.1 centimeter, meter, or mm etc..

            ############################
            if selection_Mode=='pair':
                time_interval_1_3=time_interval_1_2
            elif selection_Mode=='triplet':
                time_interval_1_3=time_interval_1_3
            
            ############################
        
            vel, flowx, flowy = calculate_velocity_displacement(vel, flowx, flowy , time_interval_1_3, conversion_factor)
            

            mean_vel_list.append(vel)
            mean_flowx_list.append(flowx)
            mean_flowy_list.append(flowy)
            pointx_list.append(points1_i)
            pointsy_list.append(points2)

            X_folder=output_dir+"/flowx/"
            Y_folder=output_dir+"/flowy/"
            VEL_folder=output_dir+"/vel/"
            plot_folder=output_dir+"/plots/"

            os.makedirs(X_folder) if not os.path.exists(X_folder) else None
            os.makedirs(Y_folder) if not os.path.exists(Y_folder) else None
            os.makedirs(VEL_folder) if not os.path.exists(VEL_folder) else None
            os.makedirs(plot_folder) if not os.path.exists(plot_folder) else None
            

            file_name_x=X_folder+ str(date1)+"_" + str(date2)+ "_" + str(date3)
            file_name_y=Y_folder+ str(date1) + "_" + str(date2)+ "_" +str(date3)
            file_name_vel=VEL_folder+ str(date1)+ "_" + str(date2)+ "_" +str(date3)
            plot_name=plot_folder+ str(date1)+"_" + str(date2)+ "_" + str(date3)

            dates_names_list.append(str(date1) + "_" + str(date2)+ "_" + str(date3))
            
           #####export ssim raster for later use########
                
            # Step 1: Read metadata from the reference raster
            # with rasterio.open(dem_path) as ref_raster:
            #     ref_meta = ref_raster.meta
            #     ref_meta.update({
            #     'height': image1.shape[0],
            #     'width': image1.shape[1]
            # })
              
            meta.update({'dtype': 'float32' , 'nodata': np.nan})
            ssim[ssim<=0]=np.nan
            ssim[ssim >= 1]=np.nan
            ssim[maskNan]=np.nan
            ssim_outdir= output_dir +"/ssim"
            os.makedirs(ssim_outdir) if not os.path.exists(ssim_outdir) else None
            ssim_outdir_fname=ssim_outdir + "/" + str(date1)+ "_" + str(date2)+ "_" +str(date3)+ ".tif"
            with rasterio.open(ssim_outdir_fname, 'w', **meta) as dst_ssim:
                    dst_ssim.write(ssim, 1)
            
            
            ##########################################
            
            x, y, xi, yi, fig_and_axes= plot_velocity_displacement(image1, image3, vel, flowx, flowy, points1_i, points2, date1, date3, pdf_filename=plot_name, time_interval=time_interval_1_3 , 
                                             show_figure=show_figure, unit=unit[0], s=point_size,
                                               bounds=[bound.left, bound.right, bound.bottom, bound.top])
            
            list_figs.append(fig_and_axes)
            figure_paths.append(plot_name)
            ############### flowx To Point Shapefile ####################
            
            dfx=pd.DataFrame()
            dfx['x']=x
            dfx['y']=y
            z_data="D"+ str(date3)
            dfx[z_data]=flowx
            # Change the dtype of a specific column to float32
            dfx[z_data] = dfx[z_data].astype('float32')
            gdfx = gpd.GeoDataFrame(dfx, geometry=gpd.points_from_xy(dfx.x, dfx.y))

            #geodfs_x.append(gdfx)

            ############### flowy To Point Shapefile ####################
            dfy=pd.DataFrame()
            dfy['x']=x
            dfy['y']=y
            z_data="D"+ str(date3)
            dfy[z_data]=flowy
            # Change the dtype of a specific column to float32
            dfy[z_data] = dfy[z_data].astype('float32')
            gdfy = gpd.GeoDataFrame(dfy, geometry=gpd.points_from_xy(dfy.x, dfy.y))
            

            ############### 2D_Vel To Point Shapefile ####################
            dfv=pd.DataFrame()
            dfv['x']=x
            dfv['y']=y
            z_data="D"+ str(date3)
            dfv[z_data]= vel
            # Change the dtype of a specific column to float32
            dfv[z_data] = dfv[z_data].astype('float32')
            gdfv = gpd.GeoDataFrame(dfv, geometry=gpd.points_from_xy(dfv.x, dfv.y))
            
           
            #############################

            # east_x, east_y, east_z, gdfx=interpolate_kriging_nans_geodataframe(data=gdfx, 
            #  threshold=None, variogram_model=None, out_fileName=None, plot=False)

            # north_x, north_y, north_z, gdfy=interpolate_kriging_nans_geodataframe(data=gdfy, 
            #  threshold=None, variogram_model=None, out_fileName=None, plot=False)

            # vel2D_x, vel2D_y, vel2D_z, gdfv=interpolate_kriging_nans_geodataframe(data=gdfv, 
            #  threshold=None, variogram_model=None, out_fileName=None, plot=False)
            
            
            
            east_z=flowx
            north_z=flowy
            vel2D_z=vel
            
            geodfs_x.append(gdfx)
            geodfs_y.append(gdfy)
            geodfs_v.append(gdfv)

            gdfx.crs=dem_crs
            gdfy.crs=dem_crs
            gdfv.crs=dem_crs
            
            if interpolate=='kriging':
                

                plot_folder_x=output_dir+'/kriging_plots_x/'
                plot_folder_Y=output_dir+'/kriging_plots_y/'
                plot_folder_VEL=output_dir+'/kriging_plots_2dvel/'
                os.makedirs(plot_folder_x) if not os.path.exists(plot_folder_x) else None
                os.makedirs(plot_folder_Y) if not os.path.exists(plot_folder_Y) else None
                os.makedirs(plot_folder_VEL) if not os.path.exists(plot_folder_VEL) else None
                
                fname_rasters=str(date1)+"_" + str(date2)+ "_" + str(date3)
                
                
                
                try:
                    Auto_Variogram(data=gdfx, column_attribute=z_data, latlon=False, aoi_shapefile=AOI, 
                                pixel_size=pixel_size,num_chunks=num_chunks,overlap_percentage=overlap_percentage, out_fileName=fname_rasters, 
                                plot_folder=plot_folder_x,  smoothing_kernel=smoothing_kernel_size, geo_folder=X_folder, krig_method=krig_method)
                except Exception as e:
                    print(f"Auto_Variogram failed with error: {e}")
                    save_xyz_as_geotiff(xi, yi, east_z, file_name_x, dem_path, AOI, interpolate='nearest')
                
                try:
                    Auto_Variogram(data=gdfy, column_attribute=z_data, latlon=False, aoi_shapefile=AOI,
                                pixel_size=pixel_size,num_chunks=num_chunks,overlap_percentage=overlap_percentage, out_fileName=fname_rasters, 
                                plot_folder=plot_folder_Y, smoothing_kernel=smoothing_kernel_size, geo_folder=Y_folder, krig_method=krig_method)
                except Exception as e:
                    print(f"Auto_Variogram failed with error: {e}")
                    save_xyz_as_geotiff(xi, yi, north_z, file_name_y, dem_path, AOI, interpolate='nearest' )
                
                try:
                    Auto_Variogram(data=gdfv, column_attribute=z_data, latlon=False, aoi_shapefile=AOI, 
                                pixel_size=pixel_size,num_chunks=num_chunks,overlap_percentage=overlap_percentage, out_fileName=fname_rasters, 
                                plot_folder=plot_folder_VEL, smoothing_kernel=smoothing_kernel_size, geo_folder=VEL_folder, krig_method=krig_method)
                except Exception as e:
                    print(f"Auto_Variogram failed with error: {e}")
                    save_xyz_as_geotiff(xi, yi, vel2D_z, file_name_vel, dem_path, AOI , interpolate='nearest')
                
            else:
                   
                save_xyz_as_geotiff(xi, yi, east_z, file_name_x, dem_path, AOI, interpolate=interpolate )
                save_xyz_as_geotiff(xi, yi, north_z, file_name_y, dem_path, AOI, interpolate=interpolate )
                save_xyz_as_geotiff(xi, yi, vel2D_z, file_name_vel, dem_path, AOI , interpolate=interpolate)


        if Vegetation_mask is not None:
            mask_all_rasters_in_directory(X_folder, Vegetation_mask)
            mask_all_rasters_in_directory(Y_folder, Vegetation_mask)
            mask_all_rasters_in_directory(VEL_folder, Vegetation_mask)
            
            
        dates_list=[extract_date_from_filename(filename) for filename in image_files]
        # Filter image_files based on extensions and extract dates
        # image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
        # dates_list = [extract_date_from_filename(filename) for filename in image_files if os.path.splitext(filename)[1].lower() in image_extensions]
            
        Total_days = (datetime.strptime(extract_date_from_filename(image_files[len(image_files)-1]), '%Y-%m-%d') - datetime.strptime(extract_date_from_filename(image_files[0]), '%Y-%m-%d')).days
        
        
        # # Concatenate GeoDataFrames
        # geodfs_x = pd.concat(geodfs_x, axis=0).reset_index(drop=True)
        # geodfs_y = pd.concat(geodfs_y, axis=0).reset_index(drop=True)
        # geodfs_v = pd.concat(geodfs_v, axis=0).reset_index(drop=True)
        

        
 
        if shapefile_output==True:
            
            shapefile_temp_dir=output_dir+"/"+"temp_shapefile_dir"
            
            os.makedirs(shapefile_temp_dir, exist_ok=True)
            
            shapefileName=shapefile_temp_dir +"/" + str(start_date_init)+ "_" + str(date2)+ "_" + str(date3)
            print('Wait for processing to complete writing data into shapefile for timeseries...')
            
           
            geodfs_x=merge_geodataframes(geodfs_x)
            geodfs_x.crs=dem_crs
            geodfs_x.to_file(shapefileName +'_E.shp')
            
            del geodfs_x
            
            geodfs_y=merge_geodataframes(geodfs_y)
            #######################3######
        
            geodfs_y.crs=dem_crs
            geodfs_y.to_file(shapefileName +'_N.shp')
            del geodfs_y
            ###########################
           
            geodfs_v=merge_geodataframes(geodfs_v)
            #print("DataFrame 2DVEL Summary:\n", geodfs_v.describe())
            geodfs_v.crs=dem_crs
            geodfs_v.to_file(shapefileName + '_2DVEL.shp')
            
            del geodfs_v
           
            import akhdefo_functions
            from akhdefo_functions import Crop_to_AOI
            # ####3Look for stable pixel###########
            # os.makedirs(ssim_outdir + '/cropped', exist_ok=True)

            # Crop_to_AOI(Path_to_WorkingDir=ssim_outdir, 
            #                   Path_to_AOI_shapefile=AOI, 
            #                   output_CroppedDir=ssim_outdir+ '/cropped', file_ex='.tif')
            # gdf_ssim=process_shapefile_with_rasters(shapefileName +'_E.shp', ssim_outdir + '/cropped')
            # #print(gdf_ssim.ssim_V.iloc[reference_index])
            # gdf_ssim.crs=dem_crs
            # gdf_ssim.to_file(output_dir+"/ssim_E.shp")
            
            # ###########3
            # gdf_ssim=process_shapefile_with_rasters(shapefileName +'_N.shp', ssim_outdir + '/cropped')
            # #print(gdf_ssim.ssim_V.iloc[reference_index])
            # gdf_ssim.crs=dem_crs
            # gdf_ssim.to_file(output_dir+"/ssim_N.shp")
            
        
            # ########
            
            # gdf_ssim=process_shapefile_with_rasters(shapefileName +'_E.shp', ssim_outdir + '/cropped')
            # #print(gdf_ssim.ssim_V.iloc[reference_index])
            # gdf_ssim.crs=dem_crs
            # gdf_ssim.to_file(output_dir+"/ssim_2DVEL.shp")
            #################
            

           
            raster_folder=[VEL_folder, Y_folder, X_folder]
            #SSIM_shape_list=[shapefileName +'_E.shp', shapefileName +'_N.shp', shapefileName +'_2DVEL.shp']
            
            ####Loop to update raster to crop to AOI#####3
            import shutil 
            for c, k in enumerate(tqdm(raster_folder, desc="Processing:Updating and Cropping Rasters")):
                cropped_dir=raster_folder[c] + '/cropped'
               
                # Check if the subdirectory exists
                if os.path.exists(cropped_dir):
                    # Remove the existing subdirectory
                    shutil.rmtree(cropped_dir)

                # # Create the subdirectory again
                # os.makedirs(sub_dir_path, exist_ok=True)
                
                os.makedirs(cropped_dir, exist_ok=True)
                Crop_to_AOI(Path_to_WorkingDir=raster_folder[c], Path_to_AOI_shapefile=AOI, output_CroppedDir=cropped_dir, file_ex='.tif')
            
            import time
            # Wait for 10 seconds
            time.sleep(10)
            
            list_shp_paths=[]
            data_list=[shapefileName +'_2DVEL.shp', shapefileName +'_N.shp', shapefileName + '_E.shp' ]
            ######################################
            for i, k in enumerate(tqdm(data_list, desc="Processing: Update Shapefiles " )):
                cropped_dir=raster_folder[i] + '/cropped'
               
                print(f'processing {data_list[i]} started... ', "\n")
                os.makedirs(cropped_dir, exist_ok=True)
                
                updated_geodf, update_shapefile_dir=update_nodata_values(shapefile_path=data_list[i], rasterfile_paths=cropped_dir,interpolate=True, VEL_Mode=VEL_Mode , VEL_scale=VEL_scale ,
                                     master_reference=master_reference, Total_days=Total_days, spatial_ref=spatial_ref)
                list_shp_paths.append(update_shapefile_dir)
                
                
                # update_nodata_values(shapefile_path=data_list[i], rasterfile_paths=cropped_dir,interpolate=False, VEL_Mode=VEL_Mode , VEL_scale=VEL_scale ,
                #                      master_reference=master_reference, Total_days=Total_days, reference_index=best_match_index)
                
                #update_nodata_values(shapefile_path=data_list[k], rasterfile_paths=raster_folder[k],interpolate=False, VEL_Mode=VEL_Mode , VEL_scale=VEL_scale , master_reference=master_reference, Total_days=Total_days)
                #update_nodata_values(shapefile_path=data_list[k], rasterfile_paths=raster_folder[k],interpolate=False, VEL_Mode=VEL_Mode , VEL_scale=VEL_scale , master_reference=master_reference, Total_days=Total_days)

                # interpolate_kriging_nans_geodataframe(data=data_list[k], 
                #     threshold=None, variogram_model=None, out_fileName=None, plot=False, 
                #     Total_days=Total_days,VEL_scale=VEL_scale, VEL_Mode=VEL_Mode)

                print(f'processing {data_list[i]} completed... ', "\n")
            
           
        
        print(f'Total Days: {Total_days}')
        with open(output_dir+"/Names.txt", "w") as file:
            for item in dates_names_list:
                # write each item on a new line
                file.write("%s\n" % item)

        
        def find_file(shapefile_temp_dir, suffix):
            for root, dirs, files in os.walk(shapefile_temp_dir):
                for file in files:
                    if file.endswith(suffix):
                        return os.path.join(root, file)
            return None
        
        xml_file_path=find_file(shapefile_temp_dir, "_2DVEL.shp.xml")
        
       
        print("file used", xml_file_path)
       # Check if the file exists
        if xml_file_path is not None:
            plot_reference_point(xml_file_path, list_figs, figure_paths)
            
            
        print ("start calculating aspect...")
        
       
        gdfe=gpd.read_file(list_shp_paths[2])
        gdfn=gpd.read_file(list_shp_paths[1])
        gdfv=gpd.read_file(list_shp_paths[0])
        
        
        gdf_crs=gdfe.crs
        
        # Calculate aspect for gdf1
        gdfe['aspect'] = np.degrees(np.arctan2(gdfn['VEL'], gdfe['VEL']))
        
        gdfe['aspect']=(450-gdfe['aspect']) % 360

        # Calculate aspect for gdf2
        gdfn['aspect'] = np.degrees(np.arctan2(gdfn['VEL'], gdfe['VEL']))
        
        gdfn['aspect']=(450-gdfn['aspect']) % 360        
        # Calculate aspect for gdf2
        gdfv['aspect'] = np.degrees(np.arctan2(gdfn['VEL'], gdfe['VEL']))
        gdfv['aspect']=(450-gdfv['aspect']) % 360                                    
        
        gdfe.to_file(list_shp_paths[2], driver='ESRI Shapefile')
        gdfn.to_file(list_shp_paths[1], driver='ESRI Shapefile')
        gdfv.to_file(list_shp_paths[0], driver='ESRI Shapefile')
        
        
        print ("calculating aspect completed")
        
        #print(f'Dates: {dates_list}')

    #     data=[dates_list,pointx_list, pointsy_list, mean_flowx_list, mean_flowy_list,mean_vel_list ]
    #    # Create DataFrame
    #     df = pd.DataFrame(data, columns=column_names)

        # Free up memory by manually invoking garbage collection
        gc.collect()
        #image1, image3, mean_vel_list, mean_flowx_list, mean_flowy_list, points1_i, points2, dates_list[0], dates_list[len(dates_list)-1]
        
        #return 

    feature_matching(folder_path=input_dir, output_dir=output_dir, zscore_threshold=zscore_threshold, AOI=AOI, conversion_factor=float(img_res[0]), ssim_thresh=ssim_thresh)

   

    
#######################################



def interpolate_xyz(x, y, z, filename, reference_raster, shapefile=None, interpolate=None, 
                        smoothing_kernel_size=None, mask=None):
    
    """
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

    
    """
    try:
        # Get the CRS, width, height, and transform from the reference raster
        with rasterio.open(reference_raster) as src:
            crs = src.crs
            width = src.width
            height = src.height
            transform = src.transform
            bounds = src.bounds
            x_min = bounds.left
            x_max = bounds.right
            y_min = bounds.bottom
            y_max = bounds.top
            pixel_size_x = src.res[0]
            pixel_size_y = src.res[1]
            

        # Create a 2D grid of coordinates based on the x, y values
        xi = np.linspace(np.min(x), np.max(x), width)
        yi = np.linspace(np.min(y), np.max(y), height)
        xi, yi = np.meshgrid(xi, yi)

        # Create an array of the same size as the x, y grid filled with NaN
        #zi = np.full_like(xi, yi, np.nan)
        #zi = np.zeros_like(xi)

        
            

        if interpolate is not None:
            # Interpolate z values onto the new grid
            zi = griddata((x, y), z, (xi, yi), method=interpolate, rescale=True)

            # Replace interpolated values outside the range with mean of initial z values
            z_min = np.min(z)
            z_max = np.max(z)
            zi[zi < z_min] = np.mean(z)
            zi[zi > z_max] = np.mean(z)

            # Find the indices of interpolated points exceeding the data range
            out_of_range_indices = np.logical_or(xi < np.min(x), xi > np.max(x)) | np.logical_or(yi < np.min(y), yi > np.max(y))

            # Replace out-of-range interpolated points with the mean of valid data points
            zi_valid = zi[~out_of_range_indices]
            mean_valid = np.nanmean(zi_valid)
            zi = np.where(out_of_range_indices, mean_valid, zi)
        else:
            zi=replace_nan_with_nearest(x, y, z, width, height)
            # # Flatten the coordinates grid and build a KDTree
            # flattened_coordinates = np.column_stack((xi.ravel(), yi.ravel()))
            # tree = cKDTree(flattened_coordinates)

            # # Query the tree for nearest neighbors to each point in x, y
            # _, indices = tree.query(np.column_stack((x, y)))

            # # Replace NaNs with z values at these indices
            # np.put(zi, indices, z)
            

            
        
            # Apply low-pass filter
        if smoothing_kernel_size is not None:
            #zi_initial=zi
            zi = gaussian(zi, sigma=smoothing_kernel_size )  # Adjust sigma according to your desired smoothing strength
            #if interpolate is None:
                #zi[zi_initial == 0] = np.nan

        if shapefile is not None:
            # Load shapefile, convert it to the correct CRS and get its geometry
            gdf = gpd.read_file(shapefile).to_crs(crs)
            shapes = gdf.geometry.values

            # Generate a mask from the shapes
            mask1 = geometry_mask(shapes, transform=transform, out_shape=zi.shape, invert=False, all_touched=True)

            # Apply the mask to the interpolated data
            zi = np.where(mask1, np.nan, zi)

        if mask is not None:
            zi=mask_raster(zi, mask )


        # Define the profile
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': zi.dtype,
            'crs': crs,
            'transform': transform,
            'nodata': np.nan,  # specify the nodata value
        }

        # Write to a new .tif file
        with rasterio.open(filename + ".tif", 'w', **profile) as dst:
            dst.write(zi, 1)

    except Exception as e:
        print("An error occurred while creating the GeoTIFF:")
        print(e)
    
    return zi