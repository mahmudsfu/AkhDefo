
def Time_Series(stacked_raster_EW=r"", stacked_raster_NS=r"", velocity_points=r"", dates_name=r"", output_folder="", outputFilename="",
                 std=1, VEL_Scale='year' , velocity_mode="mean", master_reference=False):
    
    '''
    This program uses candiate velocity points from stackprep function and performs linear interpolation in time-domain to calibrate
    stacked velocity. Additionally produces corrected timeseries velocity(daily) in a shapefile.
    
    Parameters
    ----------
    b   
    stacked_raster_EW: str
    
    stacked_raster_NS: str
    
    velocity_points: str 
        Velcity Candidate points
    
    dates_name: str
        text file include name of each date in format YYYYMMDD
    
    output_folder: str
    
    outputFilename: str
    
    VEL_Scale: str
        'year' , "month" or empty  to calculate velocity within provided dataset date range
    
    velocity_mode: str
        "mean" or "linear"
        
    master_reference: bool
        True if calculate TS to a single reference date, False if calculate TS to subsequent Reference dates
    
    Returns
    -------
    
    Time-series shape file of velocity and direction EW, NS, and 2D(resultant Velocity and direction)
    
    '''
    import glob
    import os
    from datetime import datetime
    from os.path import join

    import dask.dataframe as dd
    import geopandas as gpd
    import geowombat as gw
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    from dateutil import parser 
    
    
    def Helper_Time_Series(stacked_raster=r"", velocity_points=r"", dates_name=r"", output_folder="", outputFilename="", std=1 , VEL_Scale=VEL_Scale):
        
        '''
        stacked_raster: Path to raster stack .tif file
        
        velocity_points: Path to velocity points in arcmap shapfile format .shp
        
        dates_name: path to text file contains date names of each time series triplets .txt
        
        output_folder: Path to output Folder
        
        outputFilename: name for out time-series velocity shapefile
        '''
        
        
        
        if not os.path.exists(output_folder):
                os.makedirs(output_folder)
    
        
        #Open Raster stack, extract pixel info into shape file

        with gw.open(stacked_raster, stack_dim='time') as src:
            print(src)
            #df = src.gw.extract(velocity_points)
            df = src.gw.extract(velocity_points, use_client=True , dtype='float32')
            df[df.select_dtypes(np.float64).columns] = df.select_dtypes(np.float64).astype(np.float32)

        

        #Import names to label timeseries data    
        names = []
        dnames=[]
        with open(dates_name, 'r') as fp:
            for line in fp:
                # remove linebreak from a current name
                # linebreak is the last character of each line
                x = 'D'+ line[:-1]

                # add current item to the list
                names.append(x)
                dnames.append(x[:-18])

        print (len(dnames))
        print(len(df.columns))

        cci=(len(df.columns)- len(dnames))
        df2=df.iloc[:, cci:]
        cc=np.arange(1,cci)
        #Add Timesereises column names
        
        # #find outliers using z-score iter 1
        # lim = np.abs((df2[cc] - df2[cc].mean(axis=1)) / df2[cc].std(ddof=0, axis=1)) < std
        
        # # # # replace outliers with nan
        # df2[cc]= df2[cc].where(lim, np.nan)
        
        
        
        # df2[cc] = df2[cc].astype(float).apply(lambda x: x.interpolate(method='linear', limit_direction='both'), axis=1).ffill().bfill()
       
        
        # df2=df2.T
        
        # #find outliers using z-score iter 2
        # lim = np.abs((df2 - df2.mean(axis=0)) / df2.std(ddof=0,axis=0)) < std
        # #lim=df2.apply(stats.zscore, axis=1) <1
        # # # # replace outliers with nan
        # df2= df2.where(lim, np.nan)
        
        # df2= df2.astype(float).apply(lambda x: x.interpolate(method='linear', limit_direction='both'), axis=0).ffill().bfill()
        
        # for col in df2.columns:
        #     #print (col)
        #     #df2[col]=pd.to_numeric(df2[col])
        #     df2[col]= df2[col].interpolate(method='index', axis=0).ffill().bfill()
        
        # df2=df2.T
            
           
        #Add Timesereises column names
        df2.columns = dnames
        
        df2 = dd.from_pandas(df2, npartitions=10)
        
        
        # define a function to replace outliers with NaN using z-scores along each row
        def replace_outliers(row, stdd=std):
            zscores = (row - row.mean()) / row.std()
            row[abs(zscores) > stdd] = np.nan
            return row

        # apply the function to each row using apply
        df2 = df2.apply(replace_outliers, axis=1)
        
        #df2=df2.compute()
        
        # Select columns with 'float64' dtype  
        #float64_cols = list(df2.select_dtypes(include='float64'))

        # The same code again calling the columns
        df2[dnames] = df2[dnames].astype('float32')
        
        
        
        df2[dnames] = df2[dnames].apply(lambda x: x.interpolate(method='linear', limit_direction='both'), axis=1).ffill().bfill()
        
        df2=df2.compute()
        
        df2=df2.T
        for col in df2.columns:
            #print (col)
            #df2[col]=pd.to_numeric(df2[col])
            df2[col]= df2[col].interpolate(method='index', axis=0).ffill().bfill()
        
        df2=df2.T
        
        df2.columns = dnames
        
        
        

        # # interpolate missing values along each row
        # df2.interpolate(axis=1, limit_direction='both', limit_area='inside', method='linear', inplace=True)
                
        #  # Forward fill the DataFrame
        # df2.ffill(inplace=True)

        # # Backward fill the DataFrame
        # df2.bfill(inplace=True)
        
        #Calculate Linear Velocity for each data point
        def linear_VEL(df, dnames):
            #########
            # Extract the date columns (columns starting with 'D')
            date_columns = [col for col in df.columns if col.startswith('D')]
            dates = pd.to_datetime([col for col in date_columns], format='D%Y%m%d')
            days_since_start = (dates - dates.min()).days
            ######
            # def best_fit_slope_and_intercept(xs,ys):
            #     from statistics import mean
            #     xs = np.array(xs, dtype=np.float64)
            #     ys = np.array(ys, dtype=np.float64)
            #     m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
            #         ((mean(xs)*mean(xs)) - mean(xs*xs)))
                
            #     b = mean(ys) - m*mean(xs)
                
            #     return m, b
            dd_list=[x.replace("D", "") for x in dnames]
            dates_list=([datetime.strptime(x, '%Y%m%d') for x in dd_list])
            days_num=[( ((x) - (pd.Timestamp(year=x.year, month=1, day=1))).days + 1) for x in dates_list]
            days_num=list(range(0, len(dnames)))
            dslope=[]
            std_slope=[]
            for index, dr in df.iterrows():
                #if index==0:
                rows=df.loc[index, :].values.flatten().tolist()
                values = [dr[col] for col in dnames]
                row_values=rows
                # dfr = pd.DataFrame(dr).transpose()
                # dfr = dfr.loc[:, ~dfr.columns.str.contains('^Unnamed')]
            
                #slopeVEL=best_fit_slope_and_intercept(days_num, row_values)
                #print("slope", slopeVEL[0])
                #slope, intercept, r_value, p_value, std_err = stats.linregress(days_num, row_values)
                # Calculate the slope and intercept using polyfit
                slope, intercept = np.polyfit(days_since_start, values, 1)
                dslope.append(slope)

                # Predict values using the regression line and calculate residuals
                predicted_values = slope * days_since_start + intercept
                residuals = values - predicted_values
                
                #dslope.append(slope)
                std_slope.append(residuals)
            return dslope, std_slope
        
        
        
        
            
        
        ###########################################################################
  
        
        dnames_new=[x.replace("D", "") for x in dnames]
        def input_dates(start_date="YYYYMMDD", end_date="YYYYMMDD"):
            start_date1=parser.parse(start_date)
            end_date2=parser.parse(end_date)
            date_list_start=[]
            date_list_end=[]
            for idx, item in enumerate(dnames_new):
                #filepath1, img_name = os.path.split(item) 
                str_date1=item
                str_date2=dnames_new[len(dnames_new)-1]
                #input start date
                date_time1 = parser.parse(str_date1)
                date_list_start.append(date_time1)
                #input end date
                date_time2 = parser.parse(str_date2)
                date_list_end.append(date_time2)
            st_date=min(date_list_start, key=lambda d: abs(d - start_date1))
            text_date1=st_date.strftime("%Y%m%d")
            End_date=min(date_list_end, key=lambda d: abs(d - end_date2))
            No_ofDays=(End_date-st_date).days
            
            text_date2=End_date.strftime("%Y%m%d")
            return [text_date1, text_date2, No_ofDays]

        velocity_scale=(input_dates(start_date=dnames_new[0], end_date=dnames_new[len(dnames_new)-1]))
        
        #################################
        # for idx, row in df2[dnames].iterrows():
        #     lim = np.abs((row[dnames] - df2[dnames]()) / row[dnames].std(ddof=0)) < 1
        #     row[dnames]= row[dnames].where(lim, np.nan)
        #     row[dnames] = row[dnames].astype(float).apply(lambda x: x.interpolate(method='linear', limit_direction='both'), axis=1).ffill().bfill()
            
        
        print (df2.describe())
        temp_df=pd.DataFrame()
        temp_df[dnames[0]]=df2[dnames[0]]
        #Choosing first date as reference for Time Series
        
        if master_reference==True:
            
            df2 = df2.sub(df2[dnames[0]], axis=0)
        else:
            
            df2=df2.diff(axis = 1, periods = 1)
        # count=0
        # for idx, col in enumerate(df2.columns):
        #     df2[col] = df2[col].sub(df2[dnames[count]], axis=0)
        #     count=count+1
            
       
        df2[dnames[0]]=0
            
        linear_velocity=linear_VEL(df2[dnames], dnames)
        out=df2
        if velocity_mode=="mean":
            out['VEL']=out[dnames].mean(axis=1)
            out['VEL_STD']=out[dnames].std(axis=1)
        elif velocity_mode=="linear":
            out['VEL']=linear_velocity[0]
            out['VEL_STD']=linear_velocity[1]
        if VEL_Scale=="month": 
            out['VEL']=out['VEL']/velocity_scale[2] * 30  #velocity_scale[2] is number of days
            out['VEL_STD']=out['VEL_STD'] /velocity_scale[2] *30
        elif VEL_Scale=="year":
            out['VEL']=out['VEL']/velocity_scale[2] * 365
            out['VEL_STD']=out['VEL_STD']/velocity_scale[2] * 365
        else:
            out['VEL']=out['VEL']
            out['VEL_STD']=out['VEL_STD']
               
        
            
        out['geometry']=df['geometry']
        out['CODE']=df['SiteID']
        #out[dnames[0]]=temp_df[dnames[0]]
        # out['HEIGHT']=0
        # out['H_STDEV']=0
        #out['V_STDEV']=out[dnames].std(axis=1)
        #out['COHERENCE']=0
        #out['H_STDEF']=0
        out['x']=df['x']
        out['y']=df['y']

        col_titles=['CODE','geometry','x', 'y', 'VEL', 'VEL_STD' ]+dnames
        out = out.reindex(columns=col_titles)
        
        

        geo_out=gpd.GeoDataFrame(out, geometry='geometry', crs=df.crs)

        geo_out.to_file(output_folder +"/" + outputFilename)
        (geo_out)

        return geo_out, dnames, linear_VEL
    
    if output_folder=="":
            output_folder= "stack_data/TS"
            
    
    if not os.path.exists(output_folder):
                os.makedirs(output_folder)
    if outputFilename=="":
            outputFilename= "TS_2D_"+ os.path.basename(velocity_points)
            
            
            
    EW=Helper_Time_Series(stacked_raster=stacked_raster_EW, velocity_points=velocity_points ,
                             dates_name=dates_name, output_folder=output_folder, outputFilename="TS_EW_"+ os.path.basename(velocity_points), std=std, VEL_Scale=VEL_Scale)
                             
    NS=Helper_Time_Series(stacked_raster=stacked_raster_NS, velocity_points=velocity_points, 
                             dates_name=dates_name, output_folder=output_folder, outputFilename="TS_NS_"+ os.path.basename(velocity_points), std=std, VEL_Scale=VEL_Scale)
    
    if not os.path.exists(output_folder):
                os.makedirs(output_folder)
    if outputFilename=="":
            outputFilename= "TS_2D_"+ os.path.basename(velocity_points)
            
            
    gdf_ew=EW[0]
    gdf_ns=NS[0]
    dnames=NS[1]
    df_2D_VEL=pd.DataFrame()
    df_2D_VEL['CODE']=gdf_ew['CODE']
    df_2D_VEL['geometry']=gdf_ew['geometry']
    df_2D_VEL['x']=gdf_ew['x']
    df_2D_VEL['y']=gdf_ew['y']
    
   
   #Calculate resultant velocity magnitude
    for col in gdf_ew[dnames].columns:
       
        df_2D_VEL[col]=np.hypot(gdf_ns[col],gdf_ew[col])
       
    df_2D_VEL['VEL_MEAN']=df_2D_VEL[dnames].mean(axis=1)
    df_2D_VEL['V_STDEV']=df_2D_VEL[dnames].std(axis=1)
    #we call linear velocity function from above then reuse it below to replace VEL_2D Mean an STD below for lines
    # linear_2D_Velocity_function=EW[2]
    # linear_2D_Velocity=linear_2D_Velocity_function(df_2D_VEL[dnames], dnames)
    # df_2D_VEL['VEL']=linear_2D_Velocity[0]
    # df_2D_VEL['V_STDEV']=linear_2D_Velocity[1]
    #############################
    col_titles=['CODE','geometry','x', 'y', 'VEL_MEAN' , 'V_STDEV' ]+ dnames 
    df_2D_VEL = df_2D_VEL.reindex(columns=col_titles)
    gdf_2D_VEL=gpd.GeoDataFrame(df_2D_VEL, geometry='geometry', crs=gdf_ew.crs)
    
    
    
    gdf_2D_VEL.to_file(output_folder +"/" + outputFilename)
    
    
    #Calculate resultant velocity direction
    
    dir_df_2D_VEL=pd.DataFrame()
    dir_df_2D_VEL['CODE']=gdf_ew['CODE']
    dir_df_2D_VEL['geometry']=gdf_ew['geometry']
    dir_df_2D_VEL['x']=gdf_ew['x']
    dir_df_2D_VEL['y']=gdf_ew['y']
    
    newcol_dir_list=[]
    for col in gdf_ew[dnames].columns:
        newcol_dir= col
        newcol_dir_list.append(newcol_dir)
        dir_df_2D_VEL[newcol_dir]=np.arctan2(gdf_ns[col],gdf_ew[col])
        dir_df_2D_VEL[newcol_dir]=np.degrees(dir_df_2D_VEL[newcol_dir])
        dir_df_2D_VEL[newcol_dir]=(450-dir_df_2D_VEL[newcol_dir]) % 360
    dir_df_2D_VEL['VELDir_MEAN']=dir_df_2D_VEL[newcol_dir_list].mean(axis=1)
    col_titles=['CODE','geometry','x', 'y', 'VELDir_MEAN'  ]+ newcol_dir_list
    dir_df_2D_VEL = dir_df_2D_VEL.reindex(columns=col_titles)
    dir_gdf_2D_VEL=gpd.GeoDataFrame(dir_df_2D_VEL, geometry='geometry', crs=gdf_ew.crs)
    
    dir_gdf_2D_VEL.to_file(output_folder +"/" + outputFilename[:-4]+"_dir.shp")
    
    #Calcuate Mean Corrected velocity products MEAN X, Y, 2D and Dir
    corrected_mean_products=pd.DataFrame()
    corrected_mean_products['CODE']=gdf_ew['CODE']
    corrected_mean_products['geometry']=gdf_ew['geometry']
    corrected_mean_products['x']=gdf_ew['x']
    corrected_mean_products['y']=gdf_ew['y']
    corrected_mean_products['VEL_E']=gdf_ew['VEL']
    corrected_mean_products['VEL_N']=gdf_ns['VEL']
    #corrected_mean_products['VEL_2D']=df_2D_VEL['VEL_MEAN']
    corrected_mean_products['VEL_2D']=df_2D_VEL['VEL_MEAN']
    corrected_mean_products['2DV_STDEV']=df_2D_VEL['V_STDEV']
    corrected_mean_products['VEL_2DDir']=dir_df_2D_VEL['VELDir_MEAN']
    corrected_mean_products_geo=gpd.GeoDataFrame(corrected_mean_products, geometry='geometry', crs=gdf_ew.crs)
    
    corrected_mean_products_geo.to_file(output_folder +"/" + outputFilename[:-4]+"_mean.shp")
    
    

   

def akhdefo_dashApp(Path_to_Shapefile: str ="" , port: int =8051 , Column_Name: str =None, BaseMap: bool =False, basemap_type: str =None):
    """
    Initializes and runs a Dash web application for geospatial data analysis based on a provided shapefile.
    This application provides a user interface to visualize geospatial data on maps and time-series plots.
    It allows users to select a date range, a velocity range, and different plot types and options, such as
    whether to include a trendline or not. It also has options for customizing the plot appearance by choosing 
    from various color scales and editing the axis labels and title.

    Parameters:
        - Path_to_Shapefile (str): The file path to the shapefile that contains the geospatial data to be analyzed.
            If left empty, the application will not be able to start correctly.
        - port (int): The port number on which the Dash application will run. Defaults to 8051.

    Dependencies:
        - dash: The core library for running the Dash app.
        - dash_bootstrap_components: Provides Bootstrap components for a nicer layout and responsiveness.
        - geopandas: For reading and handling geospatial data in the form of shapefiles.
        - plotly and plotly.express: For creating interactive plots.
        - pandas: For data manipulation and analysis.
        - matplotlib and cmocean: For additional colormap options.
        - numpy: For numerical operations, especially related to creating custom trendlines.

    The application consists of two main interactive elements:
        - A map plot showing the spatial distribution of velocity data (`VEL`) where the user can select points.
        - A time-series plot that displays the average of selected data points over time, with the option to add a trendline.

            Additionally, the application includes a modal for editing plot titles and axis labels, inputs for
            minimum and maximum velocity values, date pickers to filter data based on time, dropdowns for choosing color scales 
            and plot types, and radio items for trendline options.

    Callbacks:
        - The application has callbacks to handle user interactions, updating plots based on user inputs, 
            and toggling the modal for editing plot attributes.

    Returns:
        - The Dash app object configured to be run with `app.run_server(port=port)`.

    Raises:
        - IOError: If the shapefile path is incorrect or the file cannot be read.
        - Exception: If there are any other issues reading the geospatial data or initializing the app.

    Usage:
        To use this function, ensure that you have a valid shapefile and simply call `akhdefo_dashApp(path_to_shapefile)`.
        Access the app in a web browser by navigating to `http://127.0.0.1:<port>` where `port` is the port number passed to the function.
    
    """

    import dash
    from dash import dcc, html, Input, Output, State
    import dash_bootstrap_components as dbc
    from dash_bootstrap_components import Modal
    import geopandas as gpd
    import plotly.express as px
    import pandas as pd
    import matplotlib.pyplot as plt
    import cmocean
    import matplotlib.cm as cm
    import numpy as np
    import plotly.graph_objs as go
    from numpy import polyfit
    import os
    import earthpy.spatial as es
    import rasterio
    import earthpy.plot as ep
    import rioxarray as rxr
    from scipy import signal
    
    
   ##############################
   
    import matplotlib.dates as mdates
    from datetime import datetime
    from sklearn.linear_model import LinearRegression
    # def find_zero_crossing_date(df_avg):
    #     # Function to safely calculate inverse velocity
    #     def inverse_velocity(velocity):
    #         if velocity == 0 or np.isnan(velocity):
    #             return np.nan
    #         else:
    #             return 1 / velocity

    #     # Applying inverse velocity calculation
    #     df_avg['Inverse_Average'] = df_avg['Average'].apply(inverse_velocity)

    #     # Convert string dates to datetime objects if necessary
    #     if isinstance(df_avg['Date'].iloc[0], str):
    #         df_avg['Date'] = pd.to_datetime(df_avg['Date'])

    #     # Converting dates to ordinal numbers for regression analysis
    #     date_ordinals = np.array([d.toordinal() for d in df_avg['Date']]).reshape(-1, 1)
        
    #     # Filtering out NaN values
    #     valid = ~df_avg['Inverse_Average'].isna()
    #     valid_date_ordinals = date_ordinals[valid]
    #     valid_inverse_velocity_values = df_avg['Inverse_Average'][valid].values.reshape(-1, 1)
        
    #     # Linear regression model
    #     model = LinearRegression()
    #     model.fit(valid_date_ordinals, valid_inverse_velocity_values)
        
    #     # Predicting over a larger range to extrapolate
    #     prediction_dates = np.arange(valid_date_ordinals.min(), valid_date_ordinals.max() + 365)  # Extrapolating for one more year
    #     predicted_inverse_velocities = model.predict(prediction_dates.reshape(-1, 1))
        
    #     # Finding the zero-crossing point
    #     zero_crossing = None
    #     for date, velocity in zip(prediction_dates, predicted_inverse_velocities.ravel()):
    #         if velocity <= 0:
    #             zero_crossing = date
    #             break
        
    #     zero_crossing_date = datetime.fromordinal(zero_crossing).strftime('%Y-%m-%d') if zero_crossing else None
        
    #     return zero_crossing, zero_crossing_date, valid_date_ordinals, valid_inverse_velocity_values, prediction_dates, predicted_inverse_velocities
    
   
   
   
   
   
   
   
   
   ##############################
    
        
    # Get all the Matplotlib colormaps
    mpl_colormaps = plt.colormaps()

    # Get all the cmocean colormaps
    cmocean_colormaps = [name for name in cmocean.cm.cmapnames]

    # Combine both lists, avoiding duplicates
    available_colormaps = list(set(mpl_colormaps + cmocean_colormaps))

    # Sort the list if you want an alphabetical order
    available_colormaps.sort()

    def matplotlib_to_plotly(cmap_name, pl_entries=256):
        try:
            # Attempt to get the colormap from Matplotlib's colormaps registry
            cmap = plt.colormaps[cmap_name]
        except KeyError:
            # If retrieval fails, return a default colormap
            print(f"Colormap '{cmap_name}' is not recognized by Matplotlib, using 'viridis'.")
            cmap = plt.colormaps['viridis']
        
        # Normalize the colormap scale to fit Plotly's expected format
        scale = np.linspace(0, 1, pl_entries)
        # Create the colorscale list comprehension as expected by Plotly
        plotly_colorscale = [
            [float(i)/(pl_entries - 1), f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})']
            for i, c in enumerate(cmap(np.arange(pl_entries)))
        ]
        return plotly_colorscale

    # Constants
    SHAPEFILE_PATH = Path_to_Shapefile

    # Read shapefile and handle potential errors
    try:
        gdf = gpd.read_file(SHAPEFILE_PATH)
        gdf = gdf.to_crs(epsg=4326)
    except Exception as e:
        print(f"Error reading shapefile: {e}")
        exit()

    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    # Extract the columns that start with 'D' to represent dates
    date_cols = [col for col in gdf.columns if col.startswith('D')]
    # Convert these columns to datetime objects, ignoring the 'D' prefix
    date_objs = pd.to_datetime([col[1:] for col in date_cols])

    app.layout = dbc.Container([
        html.H1("Akhdefo Geospatial Data Analysis", className="text-center mb-4"),
        dbc.Row([
            # Left side - Plots
            dbc.Col([dcc.Graph(id='time-series-plot'), dcc.Checklist(id='show-hide',options=[ {'label': 'Show/Hide Inverse Velocity', 'value': 'SH'} ],value=[]   ),  
                     dcc.Checklist(id='detrend-id',options=[ {'label': 'Detrend', 'value': 'DE'} ],value=[]   ), 
                dcc.Graph(id='map-plot', config={'staticPlot': False, 'displayModeBar': True, 'modeBarButtonsToAdd': ['select2d', 'lasso2d']}),
            #     dcc.Graph( id='map-plot',figure={'data': [go.Heatmap(z=hillshade)],
            # 'layout': go.Layout(title='Heatmap Example') }),
                
            ], width=9),  # Adjust width to 8 out of 12 units for the left column
            
            # Right side - Controls
            dbc.Col([
                # Date Picker
                dcc.DatePickerRange(
                    id='date-picker-range',
                    start_date=date_objs.min(),
                    end_date=date_objs.max(),
                    display_format='YYYY-MM-DD',
                    clearable=True,
                    with_portal=True
                ), html.Br(),
                dcc.Input(
                    id='input-vel-min',
                    type='number',
                    placeholder='Enter min VEL value',
                ),

                dcc.Input(
                    id='input-vel-max',
                    type='number',
                    placeholder='Enter max VEL value',
                ), html.Br(),

                # Color Scale Dropdown
                dcc.Dropdown(
                    id='colorscale-dropdown',
                    options=[{'label': c, 'value': c} for c in available_colormaps],
                    value='rainbow',  # default value
                    clearable=False
                ),
                # Plot Type Dropdown
                dcc.Dropdown(
                    id='plot-type-dropdown',
                    options=[
                        {'label': 'Scatter Plot', 'value': 'scatter'},
                        {'label': 'Line Plot', 'value': 'line'},
                        {'label': 'Both Scatter and Line', 'value': 'both'}
                    ],
                    value='scatter',  # default value
                    clearable=False
                ),
                # Trendline Option Radio Items
                dcc.RadioItems(
                    id='trendline-option',
                    options=[
                        {'label': 'No Trendline', 'value': 'none'},
                        {'label': 'With Trendline', 'value': 'ols'}
                    ],
                    value='none',
                    labelStyle={'display': 'block'}
                ),
                # Modal Trigger Button
                html.Button('Edit Axis Labels & Title', id='open-modal-button'),
            ], width=3),   # Adjust width to 4 out of 12 units for the right column
        ]),

        # Modal for Editing Axis Labels and Title
        dbc.Modal(
            id='edit-modal',
            is_open=False,
            children=[
                html.Div([
                    dcc.Input(id='xaxis-label-input', type='text', placeholder='Enter X-Axis Label'),
                    dcc.Input(id='yaxis-label-input', type='text', placeholder='Enter Y-Axis Label'),
                    dcc.Input(id='plot-title-input', type='text', placeholder='Enter Plot Title'),
                    html.Button('Update', id='update-plot-button')  #
                ])
            ]
        )
    ], fluid=True)


    @app.callback(
        Output('edit-modal', 'is_open'),
        [Input('open-modal-button', 'n_clicks')],
        [State('edit-modal', 'is_open')]
    )
    def toggle_modal(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open

    @app.callback(
        Output('map-plot', 'figure'),
        [Input('colorscale-dropdown', 'value'), Input('input-vel-min', 'value'),
        Input('input-vel-max', 'value')]
    )
    
    ###########
    

   
    
    ############
    def update_scatter_plot(colorscale, vel_min, vel_max):
        # Convert Matplotlib colorscale name to a Plotly colorscale
        plotly_colorscale = matplotlib_to_plotly(colorscale)
        # Set default values if None
        # if vel_min is None:
        #     vel_min = gdf['VEL'].min()

        # if vel_max is None:
        #     vel_max = gdf['VEL'].max()
        basename = os.path.basename(SHAPEFILE_PATH)
        
        
        # Check for the existence of columns
        columns_to_check = ['VEL', 'VEL_E', 'VEL_N', 'L_VEL', 'M_VEL', Column_Name]
        for column in columns_to_check:
            
            if column in gdf.columns:
                
                # Set default values if None
                if vel_min is None:
                    vel_min = gdf[column].min()
                    

                if vel_max is None:
                    vel_max = gdf[column].max()
                   
               
                if Column_Name is not None:
                    vel_color=Column_Name
                else:
                    vel_color=column
                
                
        #         gdf = gdf.to_crs(epsg=4326)
        #         center = gdf.unary_union.centroid
        #         scatter_fig=px.scatter_geo(gdf, lat=gdf.geometry.y,lon=gdf.geometry.x, color=vel_color, color_continuous_scale=plotly_colorscale, range_color=[vel_min, vel_max],
        #                                    projection='natural earth').update_geos(
        #     center=dict(lat=center.y, lon=center.x),
        #     lonaxis_range=[center.x - 5, center.x + 5],  # Adjust as needed
        #     lataxis_range=[center.y - 5, center.y + 5],  # Adjust as needed
        # )
                ########origional
                if BaseMap==True:
                    
                    # # Create a Plotly figure for hillshade
                    # scatter_fig = go.Figure(data=go.Heatmap(
                    #     z=hillshade,
                    #     x=lon_array[0],
                    #     y=lat_array[:, 0],
                    #     colorscale='gray',
                    #     showscale=False
                    # ))

                    # Add scatter plot from GeoDataFrame using Plotly Express
                    
                    
                    # scatter=px.scatter_geo(
                    #                 gdf,
                    #                 lat=gdf.geometry.y,  # Assuming these are latitude values
                    #                 lon=gdf.geometry.x,  # Assuming these are longitude values
                    #                 color=vel_color,
                    #                 color_continuous_scale=plotly_colorscale,
                    #                 title=f"Velocity Scatter Plot For {basename}",
                    #                 range_color=[vel_min, vel_max]
                    #             )
                    # scatter = px.scatter(
                    # gdf,
                    # y=gdf.geometry.y,
                    # x=gdf.geometry.x,
                    # color=vel_color,
                    # color_continuous_scale=plotly_colorscale,
                    # title=f"Velocity Scatter Plot For {basename}",  range_color=[vel_min, vel_max])

                    # Add the scatter data to the hillshade plot
                    #scatter_fig.add_trace(scatter.data[0])
                    
                    scatter_fig=px.scatter_mapbox(gdf, lat=gdf.geometry.y, lon=gdf.geometry.x,  color=vel_color,
                    color_continuous_scale=plotly_colorscale, zoom=10 , range_color=[vel_min, vel_max], width=1000, height=800 , title=f"Velocity Scatter Plot For {basename}")
                    
                    if basemap_type=='image':
                        scatter_fig.update_layout(mapbox_style="satellite", mapbox_accesstoken='pk.eyJ1IjoibWFobXVkbSIsImEiOiJjbHFlOW1tN2owbHZyMmtxZjRnZGdqYWx3In0.XPSKGgiiQ1IKkQ6IXOxfeg')
                        #scatter_fig.update_traces(marker_cauto=False, selector=dict(type='scattermapbox'))
                        
                        #scatter_fig.update_traces(cluster_color=np.array(gdf[vel_color]), selector=dict(type='scattermapbox'))
                        # scatter_fig.update_traces(marker_cmax=vel_max, selector=dict(type='scattermapbox'))
                        # scatter_fig.update_traces(marker_cmin=vel_min, selector=dict(type='scattermapbox'))
                    else:
                        
                        scatter_fig.update_layout(mapbox_style='open-street-map')
                        scatter_fig.update_layout(margin={'r':0 ,"t":50, "l":0,"b":10})
                        titlef="Velocity Scatter Plot For:" + basename
                        scatter_fig.update_layout(title=titlef)
                        scatter_fig.update_traces(cluster_color=np.array(gdf[vel_color]), selector=dict(type='scattermapbox'))

                        
                        
                    scatter_fig.update_traces(marker_colorbar_ticks='outside', selector=dict(type='scattermapbox'))
                    scatter_fig.update_traces(selected_marker_size=15, selector=dict(type='scattermapbox'))
                    #scatter_fig.update_traces(selected_marker_opacity=0.2, selector=dict(type='scattermapbox'))
                    scatter_fig.update_traces(unselected_marker_opacity=0.3, selector=dict(type='scattermapbox'))
                    # scatter_fig.update_traces(marker_cauto=False, selector=dict(type='scattermapbox'))
                    # scatter_fig.update_traces(marker_cmax=vel_max, selector=dict(type='scattermapbox'))
                    # scatter_fig.update_traces(marker_cmin=vel_min, selector=dict(type='scattermapbox'))
                else:
                    
                    scatter_fig = px.scatter(
                        gdf,
                        y=gdf.geometry.y,
                        x=gdf.geometry.x,
                        color=vel_color,
                        color_continuous_scale=plotly_colorscale,
                        title=f"Velocity Scatter Plot For {basename}",  range_color=[vel_min, vel_max])
                
                    
            
           
        return scatter_fig

    @app.callback(
        Output('time-series-plot', 'figure'),
        [Input('map-plot', 'selectedData'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date'),
        Input('trendline-option', 'value'),
        Input('plot-type-dropdown', 'value'),
        Input('update-plot-button', 'n_clicks')],
        [State('xaxis-label-input', 'value'),
        State('yaxis-label-input', 'value'),
        State('plot-title-input', 'value')] , Input('show-hide', 'value'),  Input('detrend-id', 'value')
    )
    def display_selected_data(selectedData, start_date, end_date, trendline_option, plot_type, n_clicks, xaxis_label, yaxis_label, plot_title, show_hide_value, detrend_val):
        if not selectedData:
            raise dash.exceptions.PreventUpdate

        selected_indices = [point['pointIndex'] for point in selectedData['points']]
        subset = gdf.iloc[selected_indices]

    
        avg_values = subset[date_cols].mean()
        df_avg = pd.DataFrame({'Date': date_objs, 'Average': avg_values.values})
        
        if "DE" in detrend_val:
            # Detrend the 'Data' column
            df_avg['Average'] = signal.detrend(df_avg['Average'])

        mask = (df_avg['Date'] >= start_date) & (df_avg['Date'] <= end_date)
        filtered_df = df_avg[mask]

        trendline = 'ols' if trendline_option == 'ols' else None
        
        # Fit a linear regression model manually
        x_values = np.array([(d - df_avg['Date'].min()).days for d in filtered_df['Date']])
        
        # x_values=start_date+pd.to_timedelta(x_values, unit='D')
        
        y_values = filtered_df['Average'].values
        
        slope, intercept= polyfit(x_values, y_values, 1)
        
        #############################################
       
        
        ################################################3
        
        
        # Calculate the standard deviation of the regression errors
        std_dev = np.std(y_values)
     ###################################################################################################################
        filtered_df['Inverse Average'] = 1 / filtered_df['Average'].replace(0, np.nan)  # Avoid division by zero

        # Linear regression for extrapolation
        valid_data = filtered_df.dropna(subset=['Inverse Average'])
        valid_date_ordinals = np.array([d.toordinal() for d in valid_data['Date']])
        valid_inverse_average_values = valid_data['Inverse Average'].values

        model = LinearRegression()
        model.fit(valid_date_ordinals.reshape(-1, 1), valid_inverse_average_values)

        # Predicting over a larger range to extrapolate
        prediction_dates = np.arange(valid_date_ordinals.min(), valid_date_ordinals.max() + 90)  # Extrapolating for 3 more year
        predicted_inverse_averages = model.predict(prediction_dates.reshape(-1, 1))

        # Finding the zero-crossing point
        zero_crossing = None
        for zero_date, avg in zip(prediction_dates, predicted_inverse_averages.ravel()):
            if avg <= 0:
                zero_crossing = zero_date
                break
        ###################################################################################################
        # Calculate annual change based on the slope
        # Convert slope from per day to per year (assuming 365.25 days per year to account for leap years)
        annual_change = slope * 365.25

        if plot_type == 'scatter':
            fig = px.scatter(
                filtered_df,
                x='Date',
                y='Average',
                title=plot_title,
                trendline=trendline , trendline_scope="overall", trendline_color_override="green")
            
            #####
            

                
            ####
            
        elif plot_type == 'line':
            fig = px.line(
                filtered_df,
                x='Date',
                y='Average',
                title=plot_title
            )
        elif plot_type == 'both':
            fig = px.scatter(
                filtered_df,
                x='Date',
                y='Average',
                title=plot_title,
                trendline=trendline, trendline_scope="overall", trendline_color_override="green"
            )
            fig.add_traces(px.line(
                filtered_df,
                x='Date',
                y='Average' 
            ).data)
            
        
        
        xaxis_label=xaxis_label if xaxis_label is not None else 'Dates'
        yaxis_label=yaxis_label if yaxis_label is not None else 'mm'
        # Set the default y-label to 'mm'
        fig.update_yaxes(title=yaxis_label)
        fig.update_xaxes(title=xaxis_label)
        # If trendline_option is set to 'ols', adjust the color
        # if trendline_option == 'ols':
        #     # Assume the last trace added is the trendline
        #     fig.data[-1].line.color = 'blue'

        if len(filtered_df) > 1:
            cumulative_change = filtered_df['Average'].iloc[-1] - filtered_df['Average'].iloc[0]
            mean_value = filtered_df['Average'].mean()
            mean_std=filtered_df['Average'].std()
            
            
             # Here, replace 'mm/year' with the user-provided y-axis title followed by '/year'
            yaxis_unit = yaxis_label if yaxis_label is not None else 'mm'
            unit_per_year = f"{yaxis_unit}/year"
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.1,
                text=f"Cumulative Change: {cumulative_change:.4f} {yaxis_unit}",
                showarrow=False,
                font=dict(size=14, color="blue"),
                xanchor="center",
                yanchor="top"
            )
            
            
            
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.2,
                text=f"Mean: {mean_value:.4f}, {unit_per_year}, std_dev: {mean_std:.4f}",
                showarrow=False,
                font=dict(size=14, color="red"),
                xanchor="center",
                yanchor="top" 
            )
            
            
                
             # Here, replace 'mm/year' with the user-provided y-axis title followed by '/year'
            yaxis_unit = yaxis_label if yaxis_label is not None else 'mm'
            unit_per_year = f"{yaxis_unit}/year"

            # Add or modify the annotation with the updated unit
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.15,  # Adjusted for the new annotation
                text=f"Annual Linear Change: {annual_change:.4f} {unit_per_year}, std_dev:{std_dev:.4f}",
                showarrow=False,
                font=dict(size=14, color="green"),
                xanchor="center",
                yanchor="top"
            )
            
          ##########3
            #     # Add y2 data if checkbox is checked
            if 'SH' in show_hide_value:
                # Exporting to a text file
                #filtered_df.to_csv('output.txt', sep='\t', index=False)
            
    
                # Plot extrapolated filtered_df
                extrapolated_dates = [pd.Timestamp.fromordinal(int(d)) for d in prediction_dates]
                fig.add_scatter(x=filtered_df['Date'], y=filtered_df['Inverse Average'], mode='lines+markers', name=' Inverse Velocity', yaxis='y2')
                fig.add_scatter(x=extrapolated_dates, y=predicted_inverse_averages, mode='lines', name='Extrapolated Inverse Velocity', yaxis='y2', line=dict(dash='dash'))

                if zero_crossing is not None:
                    zero_crossing_date = pd.Timestamp.fromordinal(int(zero_crossing))
                    fig.add_scatter(x=[zero_crossing_date], y=[0], mode='markers', marker=dict(color='red', size=12), yaxis='y2', name=f'Zero Crossing Point \n{str(zero_crossing_date)[:-8]}')
               
                # Update the layout for the secondary y-axis
            fig.update_layout(
             yaxis2=dict(
                #title="Inverse Velocity" if "Inverse Velocity" else 'Inverse Average',
                overlaying='y',
                side='right'
            )
        )
         
        ########

        return fig
    
    @app.callback(
    [
        Output('time-series-plot', 'layout'),
        Output('map-plot', 'layout')
    ],
    [
        Input('update-plot-button', 'n_clicks')
    ],
    [
        State('xaxis-label-input', 'value'),
        State('yaxis-label-input', 'value'),
        State('plot-title-input', 'value')
    ]
)
    def update_plot_labels(n_clicks, xaxis_label, yaxis_label, plot_title):
        if n_clicks is None:
           
            raise dash.exceptions.PreventUpdate
        
        # Update the layout for each plot with the new labels and title
        time_series_layout_update = {
            'xaxis': {'title': xaxis_label},
            'yaxis': {'title': yaxis_label},
            'title': plot_title
        }

        map_plot_layout_update = {
            'xaxis': {'title': xaxis_label},
            'yaxis': {'title': yaxis_label},
            'title': plot_title
        }

        return time_series_layout_update, map_plot_layout_update



    return app.run_server(port=port)




    


            

    



    

