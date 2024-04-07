###Start###
import glob
from itertools import count
import os
import rasterio
from rasterio.plot import show
from skimage import exposure
import numpy as np
import cv2
from rasterio.plot import show_hist
import matplotlib.pyplot as plt
#import seaborn_image as sea_img
import os
from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import cv2
from osgeo import gdal
import os
from skimage import restoration, exposure
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.filters import gaussian
import rasterio

def Filter_PreProcess(unfiltered_folderPath=r"", UDM2_maskfolderPath=r"", outpath_dir=r"" , Udm_Mask_Option=False, plot_figure=False):


    """
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

    """
    
    
    
    
    mypath = unfiltered_folderPath
    print("strat working on folder", mypath)

    #Setup Folder directories
    
    if not os.path.exists(outpath_dir):
        os.makedirs(outpath_dir)

    figs_dir=r"Figs_analysis/filter_figs"
    fig_masks=r"Figs_analysis/mask_figs"

    if not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    if not os.path.exists(fig_masks):
        os.makedirs(fig_masks)

    img_list=sorted(glob.glob(unfiltered_folderPath+ "/" +"*.tif"))
    udm_list=sorted(glob.glob(UDM2_maskfolderPath + "/" + "*.tif"))
    

    ###############

    for idx, item in enumerate(img_list):

        print(" start Processing Image Number: ", idx, ":", item)
        #img_item=img_list[idx]
        #msk_item=udm_list[idx]
        filepath1, filename = os.path.split(img_list[idx])
        #filename=filename.replace("-", "")
        #rgb=rgb_normalize(img_list[idx])
        img_src=rasterio.open(img_list[idx])
        
        meta=img_src.meta
        # meta.update({'nodata':0})
        # meta.update({'count': 3})
        # meta.update({'dtype': np.uint8})

        img_data = img_src.read([3, 2, 1], masked=True)/10000
        img_data_copy=img_data.copy()

        if UDM2_maskfolderPath != "":
            udm_src=rasterio.open(udm_list[idx])


            clean_mask = udm_src.read(1).astype(bool)
            snow_mask = udm_src.read(2).astype(bool)
            shadow_mask = udm_src.read(3).astype(bool)
            light_haze_mask = udm_src.read(4).astype(bool)
            heavy_haze_mask = udm_src.read(5).astype(bool)
            cloud_mask = udm_src.read(6).astype(bool)
            bad_mask=snow_mask+shadow_mask+cloud_mask+light_haze_mask+heavy_haze_mask
            minimum_mask=shadow_mask+cloud_mask+snow_mask
            bad_mask=~clean_mask + snow_mask
            img_data.mask = bad_mask 
            img_data = img_data.filled(fill_value=0)
        
        def normalize(array):
            array_min, array_max = array.min(), array.max()
            (array - array_min) / (array_max - array_min)
            array=exposure.rescale_intensity(array, out_range=(0, 255)).astype(np.uint8)
            return array
        def pct_clip(array,pct=[2,98]):
            array_min, array_max = np.nanpercentile(array,pct[0]), np.nanpercentile(array,pct[1])
            clip = (array - array_min) / (array_max - array_min)
            clip[clip>1]=1
            clip[clip<0]=0
            return clip

        # Convert to numpy arrays
        r_o = img_src.read(3)
        b_o = img_src.read(2)
        g_o = img_src.read(1)

        r_o=exposure.rescale_intensity(r_o, out_range=(0, 255)).astype(np.uint8)

        b_o=exposure.rescale_intensity(b_o, out_range=(0, 255)).astype(np.uint8)

        g_o=exposure.rescale_intensity(g_o, out_range=(0, 255)).astype(np.uint8)

        rgb_img_data = np.dstack((r_o, b_o, g_o))
        rgb_img_data=np.transpose(rgb_img_data, (2,0,1))

        if UDM2_maskfolderPath!="":
            r = exposure.equalize_hist(r_o, mask=clean_mask+ ~snow_mask)
            b = exposure.equalize_hist(b_o, mask=clean_mask+ ~snow_mask)
            g = exposure.equalize_hist(g_o, mask=clean_mask+ ~snow_mask)
        else:
            r = exposure.equalize_hist(r_o)
            b = exposure.equalize_hist(b_o)
            g = exposure.equalize_hist(g_o)
            
        r = exposure.equalize_adapthist(r, kernel_size=128, clip_limit=0.01, nbins=256)
        b = exposure.equalize_adapthist(b, kernel_size=128, clip_limit=0.01, nbins=256)
        g = exposure.equalize_adapthist(g, kernel_size=128, clip_limit=0.01, nbins=256)

        r=r.astype("float64")
        b=b.astype("float64")
        g=g.astype("float64")

        r[r_o==0]=0
        b[b_o==0]=0
        g[g_o==0]=0

        r=pct_clip(r)
        b=pct_clip(b)
        g=pct_clip(b)

        rgb = np.dstack((r, b, g))
        rgb=np.transpose(rgb, (2,0,1))
        #rgb[rgb==0]=255
        if Udm_Mask_Option==True:
            rgb[img_data==0]=0
            
        else:
            rgb[img_data_copy==0]=0
            


        print (rgb.shape)
        
        rgb=exposure.rescale_intensity(rgb, out_range=(0, 255)).astype(np.uint8)
        
        with rasterio.open( outpath_dir + "/" + str(filename), "w", driver='GTiff', width=img_src.shape[1],
            height=img_src.shape[0], count=3, dtype='uint8', crs=img_src.crs, transform=img_src.transform, nodata=0) as dst:
            dst.write(rgb)
        
        ##Set plot extent
        # import rioxarray as rxr
        # from rasterio.plot import plotting_extent
        # rs = rxr.open_rasterio(udm_list[0], masked=True)
        #rs_plotting_extent = plotting_extent(rs[0], rs.rio.transform())
        fig1, ((ax12, ax22), (ax32, ax42) ) = plt.subplots(2, 2, figsize=(30,20))
        show(rgb_img_data, ax=ax12)
        show_hist(rgb_img_data, ax=ax22, bins=10, lw=0.0, 
        stacked=False, alpha=0.3, histtype='stepfilled', density=True, title="Initial: "+str(filename[:10]))
        ax22.get_legend().remove()
        show(rgb, ax=ax32 )
        show_hist(rgb, ax=ax42, bins=10, lw=0.0, 
        stacked=False, alpha=0.3, histtype='stepfilled', density=True, title="CLAHE: "+str(filename[:10]))
        ax42.get_legend().remove()
        bn=img_data.shape
        xs=bn[2]
        ys=bn[1]
        ax12.set_xlim(0, xs)
        ax12.set_ylim(ys,0)
        ax32.set_xlim(0, xs)
        ax32.set_ylim(ys,0)
        
        fig1.savefig(figs_dir+ "/" +  str(filename[:10])+ ".jpg", dpi=150)
        
        if plot_figure==True:
            plt.show()
        else:
            plt.close(fig1)   

        if UDM2_maskfolderPath!="":
            fig2, (ax12 , ax22) = plt.subplots(2,1, figsize=(15,10))
            show(rgb, ax=ax12)
            show(~clean_mask, cmap="binary", ax=ax22)
            #plt.show()
            bn=img_data.shape
            xs=bn[2]
            ys=bn[1]
            ax12.set_xlim(0, xs)
            ax12.set_ylim(ys,0)
            ax22.set_xlim(0, xs)
            ax22.set_ylim(ys,0)
            fig2.savefig(fig_masks + "/" +  str(filename[:10])+ ".jpg", dpi=150)
        
            if plot_figure==True:
                plt.show()
            else:
                plt.close(fig2)
                
        if idx == len(img_list):
            break
        print("process is completed")
       
    print("All process is completed")
        

        
        # #show(bad_mask, title="bad_mask: "+str(udm_list[idx]), cmap="binary")

        # if Color==True:
        #         RGB=[3,2,1]
        #         img_data = img_src.read([RGB[0], RGB[1], RGB[2]])/1000  # apply RGB ordering and scale down
        #         img_copy=img_data.copy()
        #         meta.update({'driver':'GTiff',
        #          'width':img_src.shape[1],
        #          'height':img_src.shape[0],
        #          'count':3,
        #          'dtype':'float32',
        #          'crs':img_src.crs, 
        #          'transform':img_src.transform,
        #          'nodata':0})
        #         if Udm_Mask_Option==True:
        #             img_data = img_src.read([RGB[0], RGB[1], RGB[2]], masked=True)/1000
        #             img_data.mask = bad_mask
        #             img_data = img_data.filled(fill_value=0)
        #         else:
        #             img_data=img_data

        # elif Color==False:
        #     clean_mask = udm_src.read(1).astype(bool)
        #     gray=2
        #     img_data = img_src.read(gray)  # apply RGB ordering and scale down
        #     img_data=img_data.astype("float")
        #     img_data[clean_mask==False]=np.nan
        #     img_copy=img_data.copy()
        #     meta.update({'driver':'GTiff',
        #          'width':img_src.shape[1],
        #          'height':img_src.shape[0],
        #          'count':1,
        #          'dtype':'float32',
        #          'crs':img_src.crs, 
        #          'transform':img_src.transform,
        #          'nodata':0})


        # print ("img_data shape: ", img_data.shape)

        # img_data=exposure.rescale_intensity(img_data, out_range=(0, 255)).astype(np.uint8)
        
        # #Global Histogram Equalization
        # img_histo_equ = exposure.equalize_hist(img_data, mask=clean_mask)

        
        # img_histo_equ[img_copy==0]=0

        # img_histo_equ=exposure.rescale_intensity(img_histo_equ, out_range=(0, 255)).astype(np.uint8)

        # #Apply CLAHE Filter 1 and 2
        # filteredimage = exposure.equalize_adapthist(img_histo_equ, clip_limit=0.01, nbins=256)

        # filteredimage[img_copy==0]=0

        # if Udm_Mask_Option==True:
        #     filteredimage=filteredimage.astype("float")
        #     filteredimage[clean_mask==False]=np.nan

        # filteredimage=exposure.rescale_intensity(filteredimage, out_range=(0, 255)).astype(np.uint8)

        # filteredimage = exposure.equalize_adapthist(filteredimage, clip_limit=0.01, nbins=256)

        # filteredimage[img_copy==0]=0

        # if Udm_Mask_Option==True:
        #     filteredimage=filteredimage.astype("float")
        #     filteredimage[clean_mask==False]=np.nan

        
        # filteredimage=exposure.rescale_intensity(filteredimage, out_range=(0, 255)).astype(np.uint8)
       
       
        
    
        # filteredimage[img_copy==0]=0
        # if Udm_Mask_Option==True:
        #     filteredimage=filteredimage.astype("float")
        #     filteredimage[clean_mask==False]=np.nan

        # #filteredimage=np.transpose(filteredimage, (1,0))
        # filteredimage=np.transpose(filteredimage, (0,1, 2))
        # print (filteredimage.shape)
        # with rasterio.open(outpath_dir+ "/" + str(filename),  'w+', **meta) as dst:
        #     dst.write(filteredimage)



# def pct_clip(array, pct=[2, 98]):
#     """
#     Clip an array based on percentiles.

#     Args:
#         array (ndarray): Input array.
#         pct (list): List of percentiles [lower, upper] for clipping.
    
#     Returns:
#         ndarray: Clipped array.
#     """
#     array_min, array_max = np.nanpercentile(array, pct[0]), np.nanpercentile(array, pct[1])
#     clip = (array - array_min) / (array_max - array_min)
#     clip[clip > 1] = 1
#     clip[clip < 0] = 0
#     return clip



def Raster_Correction(input_path, output_path, limit=None, lowpass_kernel_size=5,
                      bilateral_win_size=9, bilateral_sigma_color=75, bilateral_sigma_spatial=75,
                      clip_percentiles=[2, 98], optical=False, scale='power', Vegetation_mask=None, CLAHE_filter=False):
    """
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
    """
    def pct_clip(array, pct=clip_percentiles):
        array = array.copy()  # Create a writable copy
        array_min, array_max = np.nanpercentile(array, pct[0]), np.nanpercentile(array, pct[1])
        clip = (array - array_min) / (array_max - array_min)
        clip[clip > 1] = 1
        clip[clip < 0] = 0
        return clip

    image_extensions = ('.tif', '.tiff', '.jpg', '.jpeg', '.png')  # Add more extensions if needed

    if limit is not None:
        # Get a list of image files in the directory
        image_files = [file for file in os.listdir(input_path) if any(file.endswith(ext) for ext in image_extensions)][:limit]
    else:
        image_files = [file for file in os.listdir(input_path) if any(file.endswith(ext) for ext in image_extensions)]


    # Create the plot directory if it doesn't exist
    plot_directory = os.path.join(output_path, "plots")
    os.makedirs(plot_directory, exist_ok=True)

    # Initialize progress bar
    progress_bar = tqdm(total=len(image_files), desc="Processing images", unit="image")

    if Vegetation_mask is not None:
        from scipy.ndimage import zoom
        mask_file=rasterio.open(Vegetation_mask)
        mask_data=mask_file.read(1)
        # Ensure mask_data is boolean (0 or 1)

    for image_file in image_files:
        try:
            # Open the input raster
            dataset = gdal.Open(os.path.join(input_path, image_file))

            if dataset is None:
                raise Exception(f"Failed to open raster: {image_file}")

            # Read the raster data
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()
            nodata_value = band.GetNoDataValue()
            

            data = np.ma.masked_invalid(data)
            
            if Vegetation_mask is not None:
                # If the shapes don't match and dem_array is provided, resize mask_array
                if data is not None and data.shape != mask_data.shape:
                    y_scale = data.shape[0] / mask_data.shape[0]
                    x_scale = data.shape[1] / mask_data.shape[1]
                    mask_data = zoom(mask_data, (y_scale, x_scale))

                # Threshold mask_array
                mask_data = (mask_data >= 0.5).astype(np.int32)
                mask_data = mask_data.astype(bool)

                data = np.where(mask_data == 1, data, 0)
                
            if scale=='power':
                    # if np.any(data < -1):
                    #     data_min = data.min()
                    #     data_max = data.max()
                    #     data = (data - data_min) / (data_max - data_min) * data_max
                    # Log normalize the image
                    data = np.log1p(data)  # Apply log transformation to the image
                    
                    # Step 2: Apply bilateral filter for denoising
                    if bilateral_win_size is not None:
                        denoise_bilateral = cv2.bilateralFilter(data, d=bilateral_win_size, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_spatial)
                    else:
                        denoise_bilateral=data

                    # Step 3: Normalize to 0-255 scale
                    #normalized_data = cv2.normalize(denoise_bilateral, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                    # Use rescale_intensity to stretch the intensity range to (0, 255)
                    normalized_data = (exposure.rescale_intensity(denoise_bilateral, in_range=(np.nanmin(denoise_bilateral), np.nanmax(denoise_bilateral)), out_range=(0, 255))).astype(np.uint8)

                    # If denoise_bilateral is a float image, ensure the output is in uint8 format for proper visualization
                    if normalized_data.dtype == np.float64:
                        normalized_data = (normalized_data * 255).astype(np.uint8)
                    
                       # Apply low-pass filter
                    if lowpass_kernel_size is not None:
                        normalized_data = gaussian(normalized_data, sigma=lowpass_kernel_size, preserve_range =True )  # Adjust sigma according to your desired smoothing strength
                        filtered_label='Low Pass Filter'
                        filtered=normalized_data
                    elif lowpass_kernel_size is None and optical==False:
                        #filtered=cv2.medianBlur(denoise_bilateral, 5)
                        filtered=normalized_data
                        filtered_label='Bilateral Filter'
                    else:

                        filtered_label='No-Filter Applied'

                        normalized_data=denoise_bilateral
                    
                    corrected_data = normalized_data
                        
            elif scale=='amplitude':
                    data=data
                    # Step 3: Normalize to 0-255 scale
                    #clipped_data = pct_clip(filtered)
                    if bilateral_win_size is not None:
                        
                        denoise_bilateral = cv2.bilateralFilter(data, d=bilateral_win_size, sigmaColor=bilateral_sigma_color, sigmaSpace=bilateral_sigma_spatial)  
                    else:
                        denoise_bilateral=data
                    #normalized_data = cv2.normalize(denoise_bilateral, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) 
                    normalized_data = exposure.rescale_intensity(denoise_bilateral, in_range=(np.nanmin(denoise_bilateral), np.nanmax(denoise_bilateral)), out_range=(0, 255))

                    # If denoise_bilateral is a float image, ensure the output is in uint8 format for proper visualization
                    if normalized_data.dtype == np.float64:
                        normalized_data = (normalized_data * 255).astype(np.uint8)
                    if lowpass_kernel_size is not None:
                        filtered = gaussian(normalized_data, sigma=lowpass_kernel_size, preserve_range =True )  # Adjust sigma according to your desired smoothing strength
                        filtered_label='Low Pass Filter'
                        corrected_data=filtered
                    elif lowpass_kernel_size is None and optical==False:
                        #filtered=cv2.medianBlur(denoise_bilateral, 5)
                        filtered=normalized_data
                        filtered_label='Bilateral Filter'
                        
                
                        corrected_data = normalized_data
                    else:

                        filtered_label='No-Filter Applied'

                        #normalized_data=denoise_bilateral
                    
                        corrected_data = normalized_data
                        
                   
            elif optical==True and scale is not None:
                
                print('set scale to None because you chose to process optical imagery')
            
           

            if optical==True:
                if clip_percentiles is not None:
                    
                    clipped_data = pct_clip(data)
                else:
                    clipped_data=data
                denoise_bilateral = clipped_data
                if CLAHE_filter:
                    corrected_data = exposure.equalize_adapthist(denoise_bilateral, kernel_size=128, clip_limit=0.01, nbins=256)
                else:
                    corrected_data=denoise_bilateral
                    #normalized_data = cv2.normalize(corrected_data, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
                    normalized_data = exposure.rescale_intensity(corrected_data, in_range=(np.nanmin(corrected_data), np.nanmax(corrected_data)), out_range=(0, 255))

                    # If denoise_bilateral is a float image, ensure the output is in uint8 format for proper visualization
                    if normalized_data.dtype == np.float64:
                        normalized_data = (normalized_data * 255).astype(np.uint8)
                corrected_data=normalized_data
                filtered=normalized_data
                filtered_label='CLAHE-Filter'
                

            
            
            
            
            # Plot and display the images and histograms
            fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))

            # Plot the original image
            axes[0, 0].imshow(data, cmap='gray')
            axes[0, 0].set_title('Original Image')

            # Plot the median_filter image
            axes[0, 2].imshow(filtered, cmap='gray')
            axes[0, 2].set_title(filtered_label)

            # Plot the denoise_bilateral image
            axes[0, 1].imshow(denoise_bilateral, cmap='gray')
            axes[0, 1].set_title('denoise_bilateral')

            # Plot the denoise_bilateral image
            axes[0, 3].imshow(corrected_data, cmap='gray')
            axes[0, 3].set_title('corrected_data')

            # Plot the histogram of the original data
            axes[1, 0].hist(data.flatten(), bins=50, color='b', alpha=0.7)
            axes[1, 0].set_title('Original Histogram')
            axes[1, 0].set_xlabel('Pixel Value')
            axes[1, 0].set_ylabel('Frequency')

            # Plot the histogram of the median_filter data
            axes[1, 2].hist(normalized_data.flatten(), bins=50, color='g', alpha=0.7)
            axes[1, 2].set_title(filtered_label)
            axes[1, 2].set_xlabel('Pixel Value')
            axes[1, 2].set_ylabel('Frequency')

            # Plot the histogram of the denoise_bilateral data
            axes[1, 1].hist(denoise_bilateral.flatten(), bins=50, color='r', alpha=0.7)
            axes[1, 1].set_title('denoise_bilateral Histogram')
            axes[1, 1].set_xlabel('Pixel Value')
            axes[1, 1].set_ylabel('Frequency')

            # Plot the histogram of the denoise_bilateral data
            axes[1, 3].hist(corrected_data.flatten(), bins=50, color='r', alpha=0.7)
            axes[1, 3].set_title('corrected_data Histogram')
            axes[1, 3].set_xlabel('Pixel Value')
            axes[1, 3].set_ylabel('Frequency')

            # Save the plot as a single figure
            output_filename = os.path.splitext(image_file)[0] + '_images_histograms.jpg'
            output_filepath = os.path.join(plot_directory, output_filename)
            plt.savefig(output_filepath, format='jpg')

            # Call tight_layout() to adjust the subplot spacing
            plt.tight_layout()

            # Close the plot
            plt.close()

            # Create an output raster with the corrected data
            output_filename = os.path.splitext(image_file)[0] + '.tif'
            output_filepath = os.path.join(output_path, output_filename)
            driver = gdal.GetDriverByName("GTiff")
            output_dataset = driver.Create(output_filepath, dataset.RasterXSize, dataset.RasterYSize, 1,
                                           gdal.GDT_Byte)
            
            if nodata_value is not None:
                corrected_data = np.where(corrected_data == nodata_value, np.nan, corrected_data)
            
            output_dataset.GetRasterBand(1).WriteArray(corrected_data)

            # Set the same geotransform and projection as the input raster
            output_dataset.SetGeoTransform(dataset.GetGeoTransform())
            output_dataset.SetProjection(dataset.GetProjection())

            # Close the datasets
            dataset = None
            output_dataset = None

        except Exception as e:
            print(f"Error processing raster: {image_file}")
            print(str(e))

        # Update progress bar
        progress_bar.update(1)

    # Close progress bar
    progress_bar.close()


# # Example usage
# input_path = "cropped_dir"
# output_path = "output_dir"
# limit = 5  # Maximum number of images to process, set to None for all images
# raster_correction(input_path, output_path, limit)

        