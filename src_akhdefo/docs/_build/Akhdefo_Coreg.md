## raster_alignment

Aligns raster images in a given source directory to a reference image.

Args:
src_directory (str): Path to the source directory containing images to align.
ref_filename (str): Path to the reference image.
delete_files (bool): If True, deletes the temporary directory created for alignment process. Defaults to False.

Returns:
str: Path to the directory containing all registered images.

## Coregistration

Coregister multiple rasters using both the structural similarity index and the feature matching technique.
This function is based on the AROSICS Python library.

Parameters
----------
input_Folder : str
    Path to input raster folders.

output_folder : str
    Directory to store coregistered and georeferenced raster in GeoTIFF format.

grid_res : int
    Grid resolution for coregistration.

min_reliability : int
    Structural similarity index threshold to differentiate deformation from raster shift (min=20, max=100).

window_size : tuple
    Window size for pixel search in coregistration.

path_figures : str
    Directory to store figures depicting displaced pixels in raster coordinate system units.

showFig : bool
    If True, display results. If False, do not display results.

no_data : list
    No data values to be ignored for both reference and target image.

single_ref_path : str
    Provide path to raster if interested in coregistering all rasters to a single reference. 
    If left empty, the function will use subsequent rasters as reference.

step_size : int, default=3
    Determines how many images each image serves as a reference for. A value of 3 means every image 
    acts as a reference for the next two images, and a value of 2 means every image acts as a reference for the next one.

Returns
-------
None
    The function saves the coregistered rasters and corresponding figures to specified directories.

## create_directory

Helper function to create a directory if it doesn't exist.

## generate_filename

Helper function to generate a clean filename for the saved figure.

## coregister_images

Helper function to perform coregistration and optionally save results.
