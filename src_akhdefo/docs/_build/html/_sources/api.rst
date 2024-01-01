Documentation!
==================================

This documentation covers the `akhdefo_functions` module, providing a suite of tools for various tasks including data processing, image analysis, geospatial operations, and more.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Data Acquisition and Processing
-------------------------------

refs/akhdefo_functions.akhdefo_download_planet.rst
refs/akhdefo_functions.download_RTC.rst
refs/akhdefo_functions.read_data_prep
refs/akhdefo_functions.unzip.rst
'download_RTC<refs/akhdefo_functions.move_files>'
'download_RTC<refs/akhdefo_functions.move_files_with_string>'

File and Image Management
-------------------------
.. currentmodule:: akhdefo_functions
.. autosummary::
   :toctree: generated/
   :template: mytemplate.rst

   akhdefo_functions.copyImage_Data
   akhdefo_functions.copyUDM2_Mask_Data
   akhdefo_functions.rasterClip
   akhdefo_functions.mask_raster
   akhdefo_functions.mask_all_rasters_in_directory

Image Processing and Analysis
-----------------------------
.. currentmodule:: akhdefo_functions
.. autosummary::
   :toctree: generated/
   :template: mytemplate.rst

   akhdefo_functions.Filter_PreProcess
   akhdefo_functions.Crop_to_AOI
   akhdefo_functions.Mosaic
   akhdefo_functions.Coregistration
   akhdefo_functions.DynamicChangeDetection
   akhdefo_functions.Optical_flow_akhdefo
   akhdefo_functions.assign_fake_projection
   akhdefo_functions.stackprep
   

Time Series and Geospatial Analysis
-----------------------------------
.. currentmodule:: akhdefo_functions
.. autosummary::
   :toctree: generated/
   :template: mytemplate.rst

   akhdefo_functions.Time_Series
   akhdefo_functions.Auto_Variogram
   akhdefo_functions.Akhdefo_resample
   akhdefo_functions.Akhdefo_inversion
   akhdefo_functions.utm_to_latlon
   akhdefo_functions.resample_raster
   akhdefo_functions.akhdefo_fitPlane
   akhdefo_functions.crop_to_overlap
   akhdefo_functions.calculate_new_geotransform

Visualization and Reporting
---------------------------
.. currentmodule:: akhdefo_functions
.. autosummary::
   :toctree: generated/
   :template: mytemplate.rst

   akhdefo_functions.akhdefo_dashApp
   akhdefo_functions.akhdefo_viewer
   akhdefo_functions.akhdefo_ts_plot
   akhdefo_functions.MeanProducts_plot_ts
   akhdefo_functions.plot_stackNetwork

Machine Learning and Data Science
---------------------------------
.. currentmodule:: akhdefo_functions
.. autosummary::
   :toctree: generated/
   :template: mytemplate.rst

   akhdefo_functions.Lenet_Model_training
   akhdefo_functions.classification

Application and Tool Development
--------------------------------
.. currentmodule:: akhdefo_functions
.. autosummary::
   :toctree: generated/
   :template: mytemplate.rst

   akhdefo_functions.measure_displacement_from_camera
   akhdefo_functions.measure_displacement_from_camera_toFile

Miscellaneous Functions
-----------------------
.. currentmodule:: akhdefo_functions
.. autosummary::
   :toctree: generated/
   :template: mytemplate.rst

   akhdefo_functions.binary_mask
   akhdefo_functions.crop_point_shapefile_with_aoi
   akhdefo_functions.scatter_area_mask
   akhdefo_functions.akhdefo_orthorectify
   akhdefo_functions.create_vegetation_mask
   akhdefo_functions.calculate_volume
   akhdefo_functions.Raster_Correction
   akhdefo_functions.displacement_to_volume
   akhdefo_functions.calculate_and_save_aspect_raster
   akhdefo_functions.interpolate_xyz

