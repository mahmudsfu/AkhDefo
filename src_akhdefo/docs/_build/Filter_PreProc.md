## Filter_PreProcess

This program prepare and uses filters to balanace raster image brightness

Parameters
----------
unfiltered_folderPath:  str

UDM2_maskfolderPath:    str

outpath_dir:    str

Udm_Mask_Option:    bool
    False if True the program uses planetlabs imagery unusable pixel mask to ignore and mask bad image pixels 

plot_figure:    bool
    True if you want to display output figure directly inside python


Returns
-------
geotif rasters
    Filtered geotif rasters
Figures
    plotted filtered rasters and mask for bad pixels

## Raster_Correction

Performs a series of image correction and enhancement operations on raster images,
specifically tailored for Synthetic Aperture Radar (SAR) data, but can be used for other rasters too.

This function achieves denoising, normalization, and contrast enhancement for raster images
based on user-specified parameters. The function is designed to work effectively with both 
optical and SAR imagery.

Args:
    input_path (str): Path to the directory containing input raster images.
    
    output_path (str): Directory where corrected images and plots will be saved.
    
    limit (int, optional): Maximum number of images to process. If None, all images are processed.
    
    lowpass_kernel_size (int, optional): Size of the Gaussian low-pass filter kernel.
    
    bilateral_win_size (int): Size of the bilateral filter window.
    
    bilateral_sigma_color (int): Standard deviation for color space in bilateral filter.
    
    bilateral_sigma_spatial (int): Standard deviation for spatial space in bilateral filter.
    
    clip_percentiles (list): 2-element list containing lower and upper percentiles for clipping pixel values.
    
    optical (bool): Indicates if the raster is optical imagery. Activates certain corrections specific to optical data.
    
    scale (str): Mode of scaling ('power' for logarithmic normalization and 'amplitude' for linear).
    
    Vegetation_mask (str, optional): Path to a raster file that represents a vegetation mask. Pixels in the input image that correspond to non-vegetation in the mask will be set to one.

Returns:
    None. Outputs are saved to the specified directory.

Outputs:
    - Corrected raster images saved in GeoTIFF format.
    - Plots comparing original, filtered, and corrected images along with their histograms.

Usage:
    Suitable for both SAR and optical imagery correction. Adjust filter parameters as per the nature of input raster
    and desired output.

Notes:
    - Ensure that the GDAL and other necessary libraries are correctly installed.
    - The function uses tqdm to provide a progress bar, so ensure it's installed if you want the progress visualization.
    - The function has been optimized for a balance between performance and quality, but processing large raster datasets
      might still be computationally intensive.
