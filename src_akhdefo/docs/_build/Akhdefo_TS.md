## Time_Series

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

## akhdefo_dashApp

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

## Helper_Time_Series

stacked_raster: Path to raster stack .tif file

velocity_points: Path to velocity points in arcmap shapfile format .shp

dates_name: path to text file contains date names of each time series triplets .txt

output_folder: Path to output Folder

outputFilename: name for out time-series velocity shapefile
