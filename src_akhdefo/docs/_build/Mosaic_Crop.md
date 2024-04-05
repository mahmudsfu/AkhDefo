## Mosaic

This program mosiacs raster images in geotif format as well as grab dates of the satellite image taken for further processing. 
The current version only supports PlanetLabs ortho SurfaceReflectance products.

Parameters
----------

Path_to_WorkingDir: str

output_MosaicDir: str

img_mode: int
     if img_mode=0 the the programs mosaics only the udm maskraster images.
     
     if img_mode=1 the program mosiacs only  rasters data images

Returns
-------
Mosaiced raster images

## rasterClip

Clip a raster file using an Area Of Interest (AOI) defined by a shapefile.
Uses GDAL for processing.

Parameters
----------
rasterpath : str
    Path to the input raster file.
aoi : str
    Path to the Area Of Interest shapefile.
outfilename : str
    Path to the output clipped raster file.

## Crop_to_AOI

This program used to clip multiple  raster files

Parameters
----------

Path_to_WorkingDir: str
    path to raster working directory 

Path_to_AOI_shapefile: str
    path to Area of interest in shapefile format

output_CroppedDir: str 
    path to save cropped raster files

Returns
-------
cropped raster files
