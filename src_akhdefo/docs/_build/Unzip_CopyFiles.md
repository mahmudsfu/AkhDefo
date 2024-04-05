## unzip

This program unzips all the zip products into one folder

Parameters
----------


zipdir : str
    path to directory contains all the zipfiles

dst_dir : str
    path to destination folder to copy all unzipped products.

Returns
-------
unzip folder

## copyImage_Data

This program copy all the data images into one single folder from subdirectories.

Parameters
----------

path_to_unzipped_folders : str


Path_to_raster_tifs : str

file_ext: str
    ignore this option if you want to copy all the files or
    type extension of the files of interestest such as ".tif"

Returns
-------
rasters

## copyUDM2_Mask_Data

This program copy all  raster masks.

Parameters
----------

path_to_unzipped_folders : str
    file extension must end with udm2_clip.tif

Path_to_UDM2raster_tifs : str

Returns
-------
rasters

## read_data_prep

This program reads planetlabs orthoimagery in zipfolder format
create different directory for the raster images and corresponding unusable data masks(udm)
the udm mask include snow, haze, etc... see planetslabs udm mask types for further details

Parameters
----------


zip_dir: str
    path to directory contains all the zipfiles

image_dir: str
    path to folder contains only raster data images

udm_mask_dir: str
    path to folder contains only udm mask ratsers

Returns
-------
unzip folder
image_dir
udm_mask_dir

## safe_copy

Safely copy a file to the specified directory. If a file with the same name already 
exists, the copied file name is altered to preserve both.

:param str file_path: Path to the file to copy.
:param str out_dir: Directory to copy the file into.
:param str dst: New name for the copied file. If None, use the name of the original
    file.
