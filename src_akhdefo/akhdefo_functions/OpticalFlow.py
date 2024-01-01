def binary_mask(raster_path , shape_path, output_path, file_name ):
    
    """
    
    
    Function that generates a binary mask from a vector file (shp)
    
    Parameters
    ----------
    
    raster_path: str
        path to the .tif;

    shape_path: str
        path to the shapefile.

    output_path: str
        Path to save the binary mask.

    file_name: str
        Name of the file.
    
    Returns
    -------
    Raster: ndarray
        Binary Mask in tif format
    
    
    

        
    """
    import rasterio
    from rasterio.features import rasterize
    import geopandas as gpd
    import numpy as np
    import os
    
    if not os.path.exists( output_path):
        os.makedirs( output_path)

    # Load the shapefile into a GeoDataFrame
    shapefile = gpd.read_file(shape_path)

    # Load the raster that you want to use as the base for the mask
    with rasterio.open(raster_path) as src:
        transform = src.transform
        mask = src.read(1)
        mask = mask.astype('float32')
        mask[mask == src.nodata] = 0

    # Create a binary mask using the shapefile
    mask = mask.copy()
    mask[:] = 1
    for shape in shapefile.geometry:
        mask = rasterize(
            [(shape, 0)],
            out_shape=mask.shape,
            transform=transform,
            fill=1,
            all_touched=True,
            default_value=0,
            dtype=np.uint8
        )

    # Save the binary mask as a TIFF file
    with rasterio.open(output_path+"/" + file_name+".tif", "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
                    count=1, dtype=str(mask.dtype), crs=src.crs, transform=transform, nodata=0) as dst:
        dst.write(mask.astype(mask.dtype), 1)


def DynamicChangeDetection(Path_working_Directory=r"" , Path_UDM2_folder=r"", AOI_shapefile='',
 Path_to_DEMFile=r"", out_dir="", Coh_Thresh=0.75 , vel_thresh=0.063 , image_sensor_resolution=3125.0, udm_mask_option=False , cmap='jet', 
 Median_Filter=False, Set_fig_MinMax=False, show_figure=False, plot_option="origional", xres=10, yres=10):
    

    """
    This program calculates optical flow velocity from triplets of daily optical satellite images.
    Final Timeseris products will be a shapefile format using Time_Series function after stackprep step.
    
    Parameters
    ----------

    Path_working_Directory: str
        path to filtered raster images

    Path_UDM2_folder: str
        path to planetlabs udm2 mask files

    Path_to_DEMFile: str
        path to digital elevation model
    
    AOI_shapefile: str
        path to area of interest file in esri shapefile format

    Coh_Thresh: float
        similarity index threshold

    vel_thresh: float
        maximum velocity magnitude allowed to be measured; this will help the program to exlude rockfall velocity.
        hence, only calculating displacement velocity.

    image_sensor_resolution: float
        Resolution of the satallite image raster resolution in millimeters. 
        for instance Planetlabs ortho imagery 1 pixel=3125.0 mm 
    udm_mask_option: bool
        True or False

    cmap: str
        matplotlib colormap such as "jet", "hsv", etc...

    Median_Filter: bool
        True or False

    Set_fig_MinMax: bool
        True or False

    show_figure: bool
        True or False

    plot_option: str
        "origional",  "resampled"
    
    xres: int

    yres: int

    Returns
    -------
    Rasters
         velocity in X direction(EW)
         Velocity in Y direction(NS)

    Figures  
        Initial Timesereis Figures (those figures are only intermediate products needs calibration)
    

    """
    
    def binary_mask(raster_path , shape_path, output_path, file_name ):
    
        import rasterio
        from rasterio.features import rasterize
        import geopandas as gpd
        import numpy as np
        import os
        
        if not os.path.exists( output_path):
            os.makedirs( output_path)

        # Load the shapefile into a GeoDataFrame
        shapefile = gpd.read_file(shape_path)

        # Load the raster that you want to use as the base for the mask
        with rasterio.open(raster_path) as src:
            transform = src.transform
            mask = src.read(1)
            mask = mask.astype('float32')
            mask[mask == src.nodata] = 0

        # Create a binary mask using the shapefile
        mask = mask.copy()
        mask[:] = 1
        for shape in shapefile.geometry:
            mask = rasterize(
                [(shape, 0)],
                out_shape=mask.shape,
                transform=transform,
                fill=1,
                all_touched=True,
                default_value=0,
                dtype=np.uint8
            )

        # Save the binary mask as a TIFF file
        binary_mask_file=output_path+"/" + file_name+".tif"
        with rasterio.open(binary_mask_file, "w", driver="GTiff", height=mask.shape[0], width=mask.shape[1],
                        count=1, dtype=str(mask.dtype), crs=src.crs, transform=transform, nodata=0) as dst:
            dst.write(mask.astype(mask.dtype), 1)

        return binary_mask_file
    import time
    import os
    import rasterio
    import pandas as pd
    from skimage.metrics import structural_similarity as ssim
    import matplotlib.pyplot as plt
    from skimage.registration import optical_flow_tvl1, optical_flow_ilk
    import numpy as np
    from scipy.ndimage import shift
    from pystackreg import StackReg
    from skimage.registration import phase_cross_correlation
    from skimage import exposure
    import cv2
    import os
    from os import listdir
    from os.path import isfile, join
    import numpy as np
    import seaborn_image as sb
    import skimage.transform as st 
    from scipy import interpolate
    from osgeo import gdal
    import numpy as np
    import matplotlib.pyplot as plt
    import cmocean
    import matplotlib.colors as mcolors
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import sys
    import warnings
    import earthpy.spatial as es
    import rasterio as rio
    import gc
    from scipy import ndimage
    
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    
    
    if not os.path.exists( out_dir + "/" +'georeferenced_folder/flow_xn'):
        os.makedirs(out_dir + "/" + 'georeferenced_folder/flow_xn')
    if not os.path.exists( out_dir + "/"+ 'georeferenced_folder/flow_yn'):
        os.makedirs( out_dir + "/"+ 'georeferenced_folder/flow_yn')
   
    if not os.path.exists(out_dir + "/"+ "georeferenced_folder/VEL_Triplets"):
        os.makedirs(out_dir + "/"+ "georeferenced_folder/VEL_Triplets") 

    

    
            
    # if not os.path.exists("Results_OpticalFlow\gif_dir\georeferenced_folder\direction"):
    #     os.makedirs("Results_OpticalFlow\gif_dir\georeferenced_folder\direction")
           
    output_dirflowxn=out_dir + "/"+ "georeferenced_folder/flow_xn"
    output_dirflowyn=out_dir + "/"+ "georeferenced_folder/flow_yn"
    output_VEL_Triplets=out_dir + "/"+ "georeferenced_folder/VEL_Triplets"
    
    
    # output_dir = r"Results_OpticalFlow"
    # gif_dir= r'Results_OpticalFlow/gif_dir'
    # img_list=glob.glob(Path_working_Directory+"/" + "*.tif")
    # udm2_mask_list=glob.glob(Path_UDM2_folder + "/"+ "*.tif") f for f in sorted(os.listdir(RGBFolder_jpg)) if isfile(join(RGBFolder_jpg, f))
    
    img_list = [f for f in sorted(os.listdir(Path_working_Directory)) if isfile(join(Path_working_Directory, f))]
    
    if udm_mask_option==True:
        udm2_mask_list=[f for f in sorted(os.listdir(Path_UDM2_folder)) if isfile(join(Path_UDM2_folder, f))]
    else:
        print("user selected to ignore using mask")
     ####################################################
        #Remove outliers


    # def is_outlier(points, thresh=1):
    #     """
    #     Returns a boolean array with True if points are outliers and False 
    #     otherwise.

    #     Parameters:
    #     -----------
    #         points : An numobservations by numdimensions array of observations
    #         thresh : The modified z-score to use as a threshold. Observations with
    #             a modified z-score (based on the median absolute deviation) greater
    #             than this value will be classified as outliers.

    #     Returns:
    #     --------
    #         mask : A numobservations-length boolean array.

    #     References:
    #     ----------
    #         Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
    #         Handle Outliers", The ASQC Basic References in Quality Control:
    #         Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    #     """
    #     if len(points.shape) == 3.5:
    #         points = points[:,None]
    #     median = np.median(points, axis=0)
    #     diff = np.sum((points - median)**2, axis=-1)
    #     diff = np.sqrt(diff)
    #     med_abs_deviation = np.median(diff)

    #     modified_z_score = 0.6745 * diff / med_abs_deviation

    #     return np.where(modified_z_score > thresh, True, False) #modified_z_score > thresh 

    
    f=int(1)
    for n in range(0, len(img_list)):
        # for item2 in img_list[n+1:]:
        #     for item3 in img_list[n+2:]:
        
        item1=img_list[n]
        item2=img_list[n+1]
        item3=img_list[n+2]
        img_src1=rasterio.open(join(Path_working_Directory,img_list[n]), "r+", masked=True)
        img_src2=rasterio.open(join(Path_working_Directory,img_list[n+1]), "r+", masked=True)
        img_src3=rasterio.open(join(Path_working_Directory,img_list[n+2]), "r+", masked=True)

        img1=img_src1.read(1)
        img2=img_src2.read(1)
        img3=img_src3.read(1)
        
        

        if udm_mask_option==True:
            udm2_src1=rasterio.open(join(Path_UDM2_folder,udm2_mask_list[n]))
            udm2_src2=rasterio.open(join(Path_UDM2_folder,udm2_mask_list[n+1]))
            udm2_src3=rasterio.open(join(Path_UDM2_folder,udm2_mask_list[n+2]))
            udm_img1=udm2_src1.read(1)
            udm_img2=udm2_src2.read(1)
            udm_img3=udm2_src3.read(1)

        

        
        # if udm_mask_option==False:
        #     udm_img1[udm_img1==0]=1
        #     udm_img2[udm_img2==0]=1
        #     udm_img3[udm_img3==0]=1

            img1[udm_img1==0]=0
            img2[udm_img1==0]=0
            img3[udm_img1==0]=0

            img1[udm_img2==0]=0
            img2[udm_img2==0]=0
            img3[udm_img2==0]=0

            img1[udm_img3==0]=0
            img2[udm_img3==0]=0
            img3[udm_img3==0]=0
        # else:
        #     udm_img1=udm_img1
        #     udm_img2=udm_img2
        #     udm_img3=udm_img3


        
       
        bin_mask_img1=binary_mask(raster_path=join(Path_working_Directory,img_list[n]), shape_path=AOI_shapefile, output_path=out_dir + "/"+'bin_mask', file_name=item1)
        bin_mask_img2=binary_mask(raster_path=join(Path_working_Directory,img_list[n+1]), shape_path=AOI_shapefile, output_path=out_dir + "/"+'bin_mask', file_name=item2)
        bin_mask_img3=binary_mask(raster_path=join(Path_working_Directory,img_list[n+2]), shape_path=AOI_shapefile, output_path=out_dir + "/"+'bin_mask', file_name=item3)
        
        img1[bin_mask_img2==0]=0
        img1[bin_mask_img3==0]=0

        img2[bin_mask_img1==0]=0
        img2[bin_mask_img3==0]=0

        img3[bin_mask_img1==0]=0
        img3[bin_mask_img2==0]=0
        
        

        ####Creating Image Title and labels based on YYYYDDMM and HHmm
        # filepath1, img1_name = os.path.split(item1)
        # filepath2, img2_name = os.path.split(item2)
        # filepath3, img3_name = os.path.split(item3)
        item1=item1[:10]
        item2=item2[:10]
        item3=item3[:10]
        
        item1=item1.replace("-","")
        item2=item2.replace("-","")
        item3=item3.replace("-","")
        
        
        img1_name = item1
        img2_name = item2
        img3_name = item3

        #2022-08-12.tif
        #20220812.tif
        # if len(img1_name)==12:    
        #     Date1_YYYY=img1_name[:-8]
        #     Date1_MM=img1_name[4:-6]
        #     Date1_DD=img1_name[6:-4]
        # elif len(img1_name)==14:
        #     Date1_YYYY=img1_name[:-10]
        #     Date1_MM=img1_name[5:-7]
        #     Date1_DD=img1_name[8:-4]
            
        # if len(img2_name)==12:
               
        #     Date2_YYYY=img2_name[:-8]
        #     Date2_MM=img2_name[4:-6]
        #     Date2_DD=img2_name[6:-4]
            
        # elif len(img2_name)==14:
        #     Date2_YYYY=img2_name[:-10]
        #     Date2_MM=img2_name[5:-7]
        #     Date2_DD=img2_name[8:-4]
        
        # if len(img3_name)==12:
             
        #     Date3_YYYY=img3_name[:-8]
        #     Date3_MM=img3_name[4:-6]
        #     Date3_DD=img3_name[6:-4]
            
        # elif len(img3_name)==14:
        #     Date3_YYYY=img3_name[:-10]
        #     Date3_MM=img3_name[5:-7]
        #     Date3_DD=img3_name[8:-4]
            


        #convert dates to number of days in the year for image1
        #YMD= Date1_YYYY+Date1_MM+Date1_DD
        date1 = pd.to_datetime(img1_name, format='%Y%m%d')
        new_year_day = pd.Timestamp(year=date1.year, month=1, day=1)
        day_of_the_year_date1 = (date1 - new_year_day).days + 1

        #convert dates to number of days in the year for image2
        #YMD= Date2_YYYY+Date2_MM+Date2_DD
        date2 = pd.to_datetime(img2_name, format='%Y%m%d')
        new_year_day = pd.Timestamp(year=date2.year, month=1, day=1)
        day_of_the_year_date2 = (date2 - new_year_day).days + 1

            #convert dates to number of days in the year for image3
        #YMD= Date3_YYYY+Date3_MM+Date3_DD
        date3 = pd.to_datetime(img3_name, format='%Y%m%d')
        new_year_day = pd.Timestamp(year=date3.year, month=1, day=1)
        day_of_the_year_date3 = (date3 - new_year_day).days + 1

        Delta_DD= (date3-date1).days
        Delta_DD=int(Delta_DD)
        if Delta_DD < 0:
            Delta_DD=Delta_DD*-1
        else:
            Delta_DD=Delta_DD*1

        print ("day_of_the_year_date1: ", day_of_the_year_date1)
        print("---------------------")
        print ("day_of_the_year_date2: ", day_of_the_year_date2)
        print("---------------------")
        print ("day_of_the_year_date3: ", day_of_the_year_date3)
        print("---------------------")
        print('Delta_DD: '+ str(Delta_DD))

        ##Image resgistration
        # #Affine Body transformation First attempt for image alignement using stackreg affine method
        sr = StackReg(StackReg.AFFINE)
        img2 = sr.register_transform(img1, img2)

        
        img1[bin_mask_img2==0]=0
        img1[bin_mask_img3==0]=0

        img2[bin_mask_img1==0]=0
        img2[bin_mask_img3==0]=0

        img3[bin_mask_img1==0]=0
        img3[bin_mask_img2==0]=0

        img3 = sr.register_transform(img2, img3)

        img1[bin_mask_img2==0]=0
        img1[bin_mask_img3==0]=0

        img2[bin_mask_img1==0]=0
        img2[bin_mask_img3==0]=0

        img3[bin_mask_img1==0]=0
        img3[bin_mask_img2==0]=0

        if udm_mask_option==True:
            
            img1[udm_img1==0]=0
            img2[udm_img1==0]=0
            img3[udm_img1==0]=0

            img1[udm_img2==0]=0
            img2[udm_img2==0]=0
            img3[udm_img2==0]=0

            img1[udm_img3==0]=0
            img2[udm_img3==0]=0
            img3[udm_img3==0]=0
        
        # Registration of the two images: image2 to image1
        shifts12, error, phasediff = phase_cross_correlation(img1, img2, upsample_factor=20, overlap_ratio=3)
        
        img2 = shift(img2, shift=(shifts12[0], shifts12[1]), mode='constant', prefilter=True)

        if udm_mask_option==True:
        
            img1[udm_img1==0]=0
            img2[udm_img1==0]=0
            img3[udm_img1==0]=0

            img1[udm_img2==0]=0
            img2[udm_img2==0]=0
            img3[udm_img2==0]=0

            img1[udm_img3==0]=0
            img2[udm_img3==0]=0
            img3[udm_img3==0]=0

        print ('shift:im1 and im2',shifts12, 'phasediff:', phasediff, 'error:', error )
        
        # Registration of the two images: image3 to image1
        shifts13, error, phasediff = phase_cross_correlation(img2, img3, upsample_factor=20, overlap_ratio=3)
        img3 = shift(img3, shift=(shifts13[0], shifts13[1]), mode='constant', prefilter=True)
        print ('shift:im1 and im3',shifts13, 'phasediff:', phasediff, 'error:', error )

        #Run similarity Threshold Map
        data_range = np.ptp(img1)  # compute range of img1
        mssim, grad, S12 = ssim(img1, img2, gradient=True, 
        full=True, use_sample_covariance=True, gaussian_weights=True )

        mssim, grad, S13 = ssim(img1, img3, gradient=True, 
        full=True, use_sample_covariance=True, gaussian_weights=True)

        # fig, ax = plt.subplots()
        # img = ax.imshow(S13)
        # fig.colorbar(img, ax=ax) 
        # plt.show()

        #Apply optical Flow 

        

        img1[bin_mask_img2==0]=0
        img1[bin_mask_img3==0]=0

        img2[bin_mask_img1==0]=0
        img2[bin_mask_img3==0]=0

        img3[bin_mask_img1==0]=0
        img3[bin_mask_img2==0]=0

        if udm_mask_option==True:

            img1[udm_img1==0]=0
            img2[udm_img1==0]=0
            img3[udm_img1==0]=0

            img1[udm_img2==0]=0
            img2[udm_img2==0]=0
            img3[udm_img2==0]=0

            img1[udm_img3==0]=0
            img2[udm_img3==0]=0
            img3[udm_img3==0]=0



        # flow1 = optical_flow_tvl1(img1, img2, attachment=15, tightness=0.3, num_warp=5, num_iter=10,
        #                     tol=0.0001, prefilter=True, dtype='float32')
        # flow2 = optical_flow_tvl1(img1, img3,attachment=15, tightness=0.3, num_warp=5, num_iter=10,
        #                     tol=0.0001, prefilter=True, dtype='float32')

    
        # '''

        # 0.4- image pyramid or simple image scale
        # 1 is the number of pyramid layers. 1 means that flow is calculated only from the previous image. 
        # 12 is window size. Flow is computed over the window larger value is more robust to the noise. 
        # 2 mean number of iteration of the algorithm
        # 8 is polynomial degree expansion recommended value is 5 - 7
        # 1.2 standard deviation used to smooth used derivatives recommended values from 1.1 - 1,5


        # '''
        flow1 = cv2.calcOpticalFlowFarneback(img1, img2, flow=None, pyr_scale=0.5, levels=5, winsize=12,iterations= 2, poly_n=7, poly_sigma= 1.5, flags=0)
        flow2= cv2.calcOpticalFlowFarneback(img1, img3, flow=None, pyr_scale=0.5, levels=5, winsize=12,iterations= 2, poly_n=7, poly_sigma= 1.5, flags=0)
        magnitud1, angle1 = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
        magnitud2, angle2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])

        # display dense optical flow
        # flow_x1 = flow1[1, :, :]
        # flow_y1 = flow1[0, :, :]
        flow_x1=flow1[..., 0]
        flow_y1=flow1[..., 1]

        print(flow_x1.shape, ": ",flow_y1.shape, ": ", img1.shape)

        flow_x1[img1==0]=np.nan
        flow_x1[img2==0]=np.nan
        flow_x1[img3==0]=np.nan
        flow_y1[img1==0]=np.nan
        flow_y1[img2==0]=np.nan
        flow_y1[img3==0]=np.nan

    
        flow_x2=flow2[..., 0]
        flow_y2=flow2[..., 1]
        # flow_x2 = flow2[1, :, :]
        # flow_y2 = flow2[0, :, :]

        flow_x2[img1==0]=np.nan
        flow_x2[img2==0]=np.nan
        flow_x2[img3==0]=np.nan
        flow_y2[img1==0]=np.nan
        flow_y2[img2==0]=np.nan
        flow_y2[img3==0]=np.nan

        ####Coh Thresh
        flow_x1[S12<Coh_Thresh]=np.nan
        flow_y1[S12<Coh_Thresh]=np.nan

        flow_x1[S13<Coh_Thresh]=np.nan
        flow_y1[S13<Coh_Thresh]=np.nan
        
        flow_x2[S12<Coh_Thresh]=np.nan
        flow_y2[S12<Coh_Thresh]=np.nan

        flow_x2[S13<Coh_Thresh]=np.nan
        flow_y2[S13<Coh_Thresh]=np.nan

        ####3  ####Vel thresh

        def vel_threshold(a, thresh= 0.063):
            h=a.shape[0]
            w=a.shape[1]
            print(a.shape)
            for k in range(h):
                for q in range(w):
                    if a[k,q] > thresh:
                        a[k,q]=np.nan
                    elif a[k,q] < -thresh:
                        a[k,q]=np.nan
            return a

        # flow_x1=vel_threshold(flow_x1, thresh=vel_thresh)
        # flow_x2=vel_threshold(flow_x2, thresh=vel_thresh)

        # flow_y1=vel_threshold(flow_y1, thresh=vel_thresh)
        # flow_x2=vel_threshold(flow_x2, thresh=vel_thresh)


        #############
        # flow_x1[flow_y1==np.nan]=np.nan
        # flow_y1[flow_x1==np.nan]=np.nan
        # flow_x2[flow_y2==np.nan]=np.nan
        # flow_y2[flow_x2==np.nan]=np.nan


        #################

        # from scipy import ndimage
        # flow_x1=ndimage.median_filter(flow_x1, size=20)
        # flow_x2=ndimage.median_filter(flow_x2, size=20)
        # flow_y1=ndimage.median_filter(flow_y1, size=20)
        # flow_y2=ndimage.median_filter(flow_y2, size=20)

        # flow_x1=cv2.bilateralFilter(flow_x1, 9, 11,11)
        # flow_x2=cv2.bilateralFilter(flow_x2, 9, 11,11)
        # flow_y1=cv2.bilateralFilter(flow_y1, 9, 11,11)
        # flow_y2=cv2.bilateralFilter(flow_y2, 9, 11,11)
    
        flow_x=[flow_x1,flow_x2]
        flow_x = np.stack(flow_x)
        

        flow_y=[flow_y1,flow_y2]
        flow_y=np.stack(flow_y)

    

        # flow_x=np.ma.array(flow_x, mask=(flow_x==0))
        # flow_y=np.ma.array(flow_y, mask=(flow_y==0))
        # std_x=np.nanstd(flow_x, axis=0)
        # std_x[std_x > 1]=np.nan
        # std_y=np.nanstd(flow_y, axis=0)
        # std_y[std_y > 1]=np.nan

        # mask_x=np.isnan(std_x)
        # mask_y=np.isnan(std_y)

        flow_x= np.mean(flow_x, axis=0)
        flow_y= np.mean(flow_y, axis=0)
        flow_y=flow_y*-1
        if Median_Filter==True:
            
            flow_x=ndimage.median_filter(flow_x, size=20)
            flow_y=ndimage.median_filter(flow_y, size=20)
    

        flow_y=vel_threshold(flow_y, thresh=vel_thresh)
        flow_x=vel_threshold(flow_x, thresh=vel_thresh)
        
        maskx=np.isnan(flow_x)
        masky=np.isnan(flow_y)
        flow_x[masky]=np.nan
        flow_y[maskx]=np.nan

    

        
        # Taking a matrix of size 3 as the kernel
        # kernel = np.ones((3,3), np.float32)

        # flow_x = cv2.morphologyEx(flow_x, cv2.MORPH_OPEN, kernel)
        # flow_y = cv2.morphologyEx(flow_y, cv2.MORPH_OPEN, kernel)
        # flow_x = cv2.morphologyEx(flow_x, cv2.MORPH_CLOSE, kernel)
        # flow_y = cv2.morphologyEx(flow_y, cv2.MORPH_CLOSE, kernel)

        

        ##############

        flow_x[img1==0]=np.nan
        flow_x[img2==0]=np.nan
        flow_x[img3==0]=np.nan
        flow_y[img1==0]=np.nan
        flow_y[img2==0]=np.nan
        flow_y[img3==0]=np.nan

        ##Coh Thresh
        flow_x[S12<Coh_Thresh]=np.nan
        flow_y[S12<Coh_Thresh]=np.nan
        flow_x[S13<Coh_Thresh]=np.nan
        flow_y[S13<Coh_Thresh]=np.nan

        # number_imageframes=3
        # flow_x=flow_x/number_imageframes
        # flow_y=flow_y/number_imageframes

        ###

        # flow_x=flow_x/Delta_DD  
        # flow_y=flow_y/Delta_DD
        #Save Flowx and Flow y rasters
        filename=(str(item1) + "_" +
        str(item2)+ "_"+ str(item3)+".tif")
        
        print ("Now Processing Triplet Dates: ", filename[:-4])
        
        #Read DEM file
        pathhr=Path_to_DEMFile
        with rasterio.open(pathhr, 'r') as r1:
            demfile = r1.read(1)
            meta = r1.meta
        
        flow_xgr=flow_x /3/int(Delta_DD) * image_sensor_resolution #convert pixel to cm(mm) and divide by difference of number of days between frame1 and frame3
        flow_ygr=flow_y /3/ int(Delta_DD) * image_sensor_resolution

    
        mag_map1 = np.hypot(flow_xgr, flow_ygr)  # magnitude
        if udm_mask_option==True:
            mag_map1[udm_img1==0]=np.nan
            mag_map1[udm_img2==0]=np.nan
            mag_map1[udm_img3==0]=np.nan

        mag_map1[img1==0]=np.nan
        mag_map1[img2==0]=np.nan
        mag_map1[img3==0]=np.nan
        mag_map1[S12<Coh_Thresh]=np.nan
        mag_map1[S13<Coh_Thresh]=np.nan
        

        flow_ygr[mag_map1 == np.nan]=np.nan
        flow_xgr[mag_map1 == np.nan]=np.nan

        flow_ygr[flow_x==np.nan]=np.nan
        flow_xgr[flow_y==np.nan]=np.nan


        # mag_map1 = mag_map1[~is_outlier(mag_map1)]
        # flow_xgr = flow_xgr[~is_outlier(flow_xgr)]
        # flow_ygr = flow_ygr[~is_outlier(flow_ygr)]

        # flow_ygr[flow_ygr == 0]=np.nan
        # flow_xgr[flow_xgr == 0]=np.nan
        # mag_map1[mag_map1 == 0]=np.nan

        
    

        # def interpolate_missing_pixels(
        #         image: np.ndarray,
        #         mask: np.ndarray,
        #         method: str = 'nearest',
        #         fill_value: int = 0):
        #     """
        #     :param image: a 2D image
        #     :param mask: a 2D boolean image, True indicates missing values
        #     :param method: interpolation method, one of
        #         'nearest', 'linear', 'cubic'.
        #     :param fill_value: which value to use for filling up data outside the
        #         convex hull of known pixel values.
        #         Default is 0, Has no effect for 'nearest'.
        #     :return: the image with missing values interpolated
        #     """
            

        #     h, w = image.shape[:2]
        #     xx, yy = np.meshgrid(np.arange(w), np.arange(h))

        #     known_x = xx[~mask]
        #     known_y = yy[~mask]
        #     known_v = image[~mask]
        #     missing_x = xx[mask]
        #     missing_y = yy[mask]


        #     #######

        #     xvalues = np.linspace(int(1), int(1+w), w) 
        #     yvalues = np.linspace(int(1), int(1+h), h)

        #     missing_x, missing_y = np.meshgrid(xvalues, yvalues)
            



        #     interp_values = interpolate.griddata(
        #         (known_x, known_y), known_v, (missing_x, missing_y),
        #         method=method, fill_value=fill_value
        #     )

        #     # interp_image = image.copy()
        #     # interp_image[missing_y, missing_x] = interp_values

        #     interp_image=interp_values

        #     return interp_image

        ##Reapply bad mask

        mag_map1[flow_xgr==np.nan]=np.nan

        flow_xgr[img1==0]=np.nan
        flow_ygr[img1==0]=np.nan
        mag_map1[img1==0]=np.nan

        if udm_mask_option==True:

            flow_xgr[udm_img1==0]=np.nan
            flow_ygr[udm_img1==0]=np.nan
            mag_map1[udm_img1==0]=np.nan

            flow_xgr[udm_img2==0]=np.nan
            flow_ygr[udm_img2==0]=np.nan
            mag_map1[udm_img2==0]=np.nan


            flow_xgr[udm_img3==0]=np.nan
            flow_ygr[udm_img3==0]=np.nan
            mag_map1[udm_img3==0]=np.nan


        with rasterio.open(output_VEL_Triplets + "/" + str(filename), 'w', **meta) as dst:
            dst.write(mag_map1, indexes=1)

        with rasterio.open(output_dirflowxn + "/" + str(filename), 'w', **meta) as dst:
            dst.write(flow_xgr, indexes=1)
        with rasterio.open(output_dirflowyn + "/" + str(filename), 'w', **meta) as dst:
            dst.write(flow_ygr, indexes=1)

###########################################################################Interpolation of data below
        ################
        # xgrmask=np.isnan(flow_x)
        # ygrmask=np.isnan(flow_y)
        # mag_mask=np.isnan(mag_map1)

        
        # flow_xgr=interpolate_missing_pixels(flow_xgr, xgrmask , "cubic")
        # flow_ygr_int=interpolate_missing_pixels(flow_ygr, ygrmask , "cubic")

        # flow_ygr[mag_map1 == np.nan]=np.nan
        # flow_xgr[mag_map1 == np.nan]=np.nan

        # mag_map1=interpolate_missing_pixels(mag_map1, ygrmask , "cubic")

        # ###Apply velocity threshold
        # flow_xgr_int=vel_threshold(flow_xgr_int, thresh=vel_thresh * 3125.0)
        # flow_ygr_int=vel_threshold(flow_ygr_int, thresh=vel_thresh * 3125.0)
        # mag_map1_int=vel_threshold(mag_map1_int, thresh=vel_thresh * 3125.0)

        #  ##Reapply bad mask

        # mag_map1[flow_xgr==np.nan]=np.nan

        # flow_xgr[img1==0]=np.nan
        # flow_ygr[img1==0]=np.nan
        # mag_map1[img1==0]=np.nan

        # flow_xgr[udm_img1==0]=np.nan
        # flow_ygr[udm_img1==0]=np.nan
        # mag_map1[udm_img1==0]=np.nan

        # flow_xgr[udm_img2==0]=np.nan
        # flow_ygr[udm_img2==0]=np.nan
        # mag_map1[udm_img2==0]=np.nan


        # flow_xgr[udm_img3==0]=np.nan
        # flow_ygr[udm_img3==0]=np.nan
        # mag_map1[udm_img3==0]=np.nan


        # ######Interpolation iter2

        #  ################
        # xgrmask=np.isnan(flow_x)
        # ygrmask=np.isnan(flow_y)
        # mag_mask=np.isnan(mag_map1)
        
        # flow_xgr_int=interpolate_missing_pixels(flow_xgr_int, xgrmask , "nearest")
        # flow_ygr_int=interpolate_missing_pixels(flow_ygr_int, ygrmask , "nearest")

        # flow_ygr_int[mag_map1 == np.nan]=np.nan
        # flow_xgr_int[mag_map1 == np.nan]=np.nan

        # mag_map1_int=interpolate_missing_pixels(mag_map1_int, ygrmask , "nearest")

        # flow_xgr_int=vel_threshold(flow_xgr_int, thresh=vel_thresh * 3125.0)
        # flow_ygr_int=vel_threshold(flow_ygr_int, thresh=vel_thresh * 3125.0)
        # mag_map1_int=vel_threshold(mag_map1_int, thresh=vel_thresh * 3125.0)


        ################



        # mag_map1_int[flow_xgr==np.nan]=np.nan

        # flow_xgr_int[img1==0]=np.nan
        # flow_ygr_int[img1==0]=np.nan
        # mag_map1_int[img1==0]=np.nan

        # flow_xgr_int[udm_img1==0]=np.nan
        # flow_ygr_int[udm_img1==0]=np.nan
        # mag_map1_int[udm_img1==0]=np.nan

        # flow_xgr_int[udm_img2==0]=np.nan
        # flow_ygr_int[udm_img2==0]=np.nan
        # mag_map1_int[udm_img2==0]=np.nan


        # flow_xgr_int[udm_img3==0]=np.nan
        # flow_ygr_int[udm_img3==0]=np.nan
        # mag_map1_int[udm_img3==0]=np.nan


        


        # mag_map1 = mag_map1[~is_outlier(mag_map1)]
        # flow_xgr = flow_xgr[~is_outlier(flow_xgr)]
        # flow_ygr = flow_ygr[~is_outlier(flow_ygr)]

        # mag_map1=np.percentile(mag_map1, 80)
        # flow_xgr=np.percentile(flow_xgr, 80)
        # flow_ygr=np.percentile(flow_ygr, 80)

        #############################################
        xname=output_dirflowxn + "/" + str(filename)
        yname=output_dirflowyn + "/" + str(filename)
        vname=output_VEL_Triplets + "/" + str(filename)

    
    
        # ###Save interpolated rasters
        # with rasterio.open(xname_int, 'w', **meta) as dst:
        #     dst.write(flow_xgr_int, indexes=1)

        # with rasterio.open(yname_int, 'w', **meta) as dst:
        #     dst.write(flow_ygr_int, indexes=1)
        # with rasterio.open(vname_int, 'w', **meta) as dst:
        #     dst.write(mag_map1_int, indexes=1)


    

        # print("max", mag_map1.max())
        # print("min", mag_map1.min())
        # print("std_mag", np.std(mag_map1))
        # print("mean_mag", np.mean(mag_map1))
        # print("-----------------------------")
        # print("Dates: ", item1[:-4], item2[:-4],  item3[:-4])
        # # print('default Flow_x Min', flow_xgr.min())
        # # print('default Flow_y Min', flow_xgr.min())

        # print("---------------------------------")



        ###########Resample data to remove noise and fill gaps
        def resample(input_raster="", output_raster="" , xres=3.125 , yres=3.125):
            

            ds = gdal.Open(input_raster)

            # resample
            dsRes = gdal.Warp(output_raster, ds, xRes = xres, yRes = yres, 
                            resampleAlg = "bilinear")

            # visualize
            # array = dsRes.GetRasterBand(1).ReadAsArray()
            # plt.figure()
            # plt.imshow(array)
            # plt.colorbar()
            # plt.show()

            # close your datasets!
            dsRes =None
            return output_raster

        if plot_option=="resampled":
            if not os.path.exists(output_dirflowxn + "/resampled" ):
                os.makedirs(output_dirflowxn + "/resampled")

            if not os.path.exists(output_dirflowyn + "/resampled" ):
                os.makedirs(output_dirflowyn + "/resampled")

            if not os.path.exists(output_VEL_Triplets + "/resampled" ):
                os.makedirs(output_VEL_Triplets + "/resampled")


            xname_int=output_dirflowxn + "/resampled" + "/" + str(filename)
            yname_int=output_dirflowyn + "/resampled" + "/" + str(filename)
            vname_int=output_VEL_Triplets + "/resampled" + "/" + str(filename)
            ew=resample(xname, xname_int, xres=xres, yres=yres)
            ns=resample(yname, yname_int, xres=xres, yres=yres)
            _2dvel=resample(vname, vname_int, xres=xres, yres=yres)
        else:
            print(" User selected to ignore resampling raster images ")
        #####Plot Timeseries

        #Function to read dem and prepare it to create hillshade for basemap

        def raster2array(geotif_file):
            metadata = {}
            dataset = gdal.Open(geotif_file)
            metadata['array_rows'] = dataset.RasterYSize
            metadata['array_cols'] = dataset.RasterXSize
            metadata['bands'] = dataset.RasterCount
            metadata['driver'] = dataset.GetDriver().LongName
            metadata['projection'] = dataset.GetProjection()
            metadata['geotransform'] = dataset.GetGeoTransform()
            mapinfo = dataset.GetGeoTransform()
            metadata['pixelWidth'] = mapinfo[1]
            metadata['pixelHeight'] = mapinfo[5]
        #     metadata['xMin'] = mapinfo[0]
        #     metadata['yMax'] = mapinfo[3]
        #     metadata['xMax'] = mapinfo[0] + dataset.RasterXSize/mapinfo[1]
        #     metadata['yMin'] = mapinfo[3] + dataset.RasterYSize/mapinfo[5]  
            metadata['ext_dict'] = {}
            metadata['ext_dict']['xMin'] = mapinfo[0]
            metadata['ext_dict']['xMax'] = mapinfo[0] + dataset.RasterXSize/mapinfo[1]
            metadata['ext_dict']['yMin'] = mapinfo[3] + dataset.RasterYSize/mapinfo[5]
            metadata['ext_dict']['yMax'] = mapinfo[3]
            
            metadata['extent'] = (metadata['ext_dict']['xMin'],metadata['ext_dict']['xMax'],
                                metadata['ext_dict']['yMin'],metadata['ext_dict']['yMax'])
            
            if metadata['bands'] == 1:
                raster = dataset.GetRasterBand(1)
                metadata['noDataValue'] = raster.GetNoDataValue()
                metadata['scaleFactor'] = raster.GetScale()
                
                # band statistics
                metadata['bandstats'] = {} #make a nested dictionary to store band stats in same 
                stats = raster.GetStatistics(True,True)
                metadata['bandstats']['min'] = round(stats[0],2)
                metadata['bandstats']['max'] = round(stats[1],2)
                metadata['bandstats']['mean'] = round(stats[2],2)
                metadata['bandstats']['stdev'] = round(stats[3],2)
                
                array = dataset.GetRasterBand(1).ReadAsArray(0,0,metadata['array_cols'],metadata['array_rows']).astype('float32')
                array[array==(metadata['noDataValue'])]=np.nan 
                # array = array/metadata['scaleFactor']
                return array, metadata
            
            elif metadata['bands'] > 1:
                print('More than one band ... fix function for case of multiple bands')

        # def hillshade(array,azimuth,angle_altitude):
        #     azimuth = 360.0 - azimuth 
        #     x, y = np.gradient(array)
        #     slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
        #     aspect = np.arctan2(-x, y)
        #     azimuthrad = azimuth*np.pi/180.
        #     altituderad = angle_altitude*np.pi/180.
        #     shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)

           # return 255*(shaded + 1)/2

        
        
        
        # # Open the DEM with Rasterio
        # with rio.open(Path_to_DEMFile) as src:
        #     elevation = src.read(1)
        #     # Set masked values to np.nan
        #     elevation[elevation < 0] = np.nan
        
        
        
        dtm_array, dtm_metadata = raster2array(Path_to_DEMFile)
        if plot_option=="origional":

            srcx, src_meta = raster2array(xname)
            srcy, src_meta = raster2array(yname)
            srcv, src_meta = raster2array(vname)
        elif plot_option=="resampled":
            srcx, src_meta = raster2array(xname_int)
            srcy, src_meta = raster2array(yname_int)
            srcv, src_meta = raster2array(vname_int)
        
        srcx = st.resize(srcx, dtm_array.shape, mode='constant')
        srcy = st.resize(srcy, dtm_array.shape, mode='constant')
        srcv = st.resize(srcv, dtm_array.shape, mode='constant')

        # srcx_int, src_meta = raster2array(xname_int)
        # srcy_int, src_meta = raster2array(yname_int)
        # srcv_int, src_meta = raster2array(vname_int)
        
        # Use hillshade function on a DTM Geotiff
        #hs_array = hillshade(dtm_array,335,45)
        hs_array=es.hillshade(dtm_array)
        
        def plot(hillshade, velocity, filename="" , title="EW Velocity", outputfolder=r"" , Delta_DD=1, cmap=cmap, masked=False, cbar_unit="mm/day"):
            
           
            outputfolder='Figs_analysis/timeseriesFigs' + "/" + outputfolder
        
            if not os.path.exists(outputfolder ):
                os.makedirs(outputfolder)

            velocity=velocity.astype(np.int16)
            diff = mcolors.LinearSegmentedColormap.from_list("", ["blue","white","red"])
            max=velocity.max()
            min=velocity.min()
            #print("min: ", min, " max: ", max)
            
            if min < 0 and max >0 :
                if Set_fig_MinMax==True:
                    min_n=-50
                    max_n=50
                    min=min_n
                    max=max_n
                    offset = mcolors.TwoSlopeNorm(vmin=min,
                                vcenter=0., vmax=max)  
                else:
                    offset = mcolors.TwoSlopeNorm(vmin=min,
                                vcenter=0., vmax=max)
                    
            else  : 
                if Set_fig_MinMax==True:
                    min_n=0
                    max_n=100
                    min=min_n
                    max=max_n
                    offset=mcolors.Normalize(vmin=min, vmax=max)
                else:
                    offset=mcolors.Normalize(vmin=min, vmax=max)

            # elif max <= 0 :
            #     offset=mcolors.Normalize()
                
            #scale = {"dx" : pixel_resolution_meter, "units" : "m"}
            fig, ax=plt.subplots(figsize=(10,5), nrows=1, ncols=1)

            axins = inset_axes(
            ax,
            width="5%",  # width: 3% of parent_bbox width
            height="75%",  # height: 50%
            loc="lower left",
            bbox_to_anchor=(1.05, 0., 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0)
            
            ax0=sb.imshow(hillshade, ax=ax,  cmap="gray", cbar=False, dx=3.125, units='m', color="k")

            if masked==True:
                velocity=velocity.astype("float32")
                velocity[velocity==0]=np.nan
            ax1=ax.imshow(velocity,  alpha=0.75, norm=offset, cmap=cmap )
            ax.set_title(str(title))
            # plt.colorbar(ax1, orientation="horizontal")
            a=fig.colorbar(ax1, cax=axins, extend="both")
            a.set_label(cbar_unit, labelpad=2, y=0.5, rotation=90)
            # title
            
            #set the subtitle
            dates_title="Dates:" +str(item1[:10]) + "-"+ str(item2[:10]) + "-"+ str(item3[:10])
        
            ####
            # ax1=sb.imgplot(hillshade, ax=ax[1],  cmap="gray", cbar=False)
            # ax1=sb.imgplot(velocity_interpolated, ax=ax[1], alpha=0.75, cbar_label="mm/"+"day", cmap="gist_rainbow", **scale, showticks=True, orientation="h")
            plt.grid('on') 
            #ax[0].ticklabel_format(useOffset=True, style='plain') #do not use scientific notation #
            #ax[1].set_title("Interpolated: " + str(title) + str(filename[:-4]))
            # scalebar = ScaleBar(3.125, color="k") # 1 pixel = 3.125 meter
            # plt.gca().add_artist(scalebar)
            figname=filename[:-4]+".jpg"
            
            bn=dtm_array.shape
            xs=bn[1]
            ys=bn[0]
            ax.set_xlim(0, xs)
            ax.set_ylim(ys,0)
            ax.text(0.5, - 0.07, dates_title,horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
            bbox=dict(facecolor='red', alpha=0.5))   


            plt.savefig(outputfolder + "/" + figname)

            if show_figure==True:
                plt.show()
            else:
                plt.close(fig)
            del hillshade
            del velocity
            gc.collect()

        _1=plot(hs_array, srcx, filename, title="EW Mean-Velocity", outputfolder="EW", Delta_DD= int(Delta_DD), cmap=cmap , masked=True, cbar_unit="mm" )
        _2=plot(hs_array, srcy, filename, title="NS Mean-Velocity", outputfolder="NS" , Delta_DD= int(Delta_DD), cmap=cmap, masked=True, cbar_unit="mm")
        _3=plot(hs_array, srcv, filename, title="2D Mean-Velocity", outputfolder="2D" , Delta_DD= int(Delta_DD),  cmap=cmap, masked=True, cbar_unit="mm")

    

        S12=exposure.rescale_intensity(S12, out_range=(0, 100)).astype(np.float32)
        S13=exposure.rescale_intensity(S13, out_range=(0, 100)).astype(np.float32)
        S12[S12==100]=np.nan
        S13[S13==100]=np.nan
        S12[S12<Coh_Thresh*100]=np.nan
        S13[S13<Coh_Thresh*100]=np.nan
        #########3
        S12 = st.resize(S12, dtm_array.shape, mode='constant')
        S13 = st.resize(S13, dtm_array.shape, mode='constant')
        ############
        
        _4=plot(hs_array, S12, filename, title="Similarity Index Between Dates 1&2", outputfolder="S12" , Delta_DD= int(Delta_DD) , cmap=cmap, masked=True, cbar_unit="Similarity-Index %")
        _5=plot(hs_array, S13, filename, title="Similarity Index Between Dates 1&3", outputfolder="S13" , Delta_DD= int(Delta_DD), cmap=cmap, masked=True, cbar_unit="Similarity-Index %")


        # fig, ax = plt.subplots()
        # img = ax.imshow(S13)
        # fig.colorbar(img, ax=ax) 
        # plt.show()
        del hs_array
        del S12
        del S13
        del shifts12
        del shifts13
        del dtm_array
        del flow1
        del flow2
        del flow_x
        del flow_y
        del flow_x1
        del flow_x2
        del flow_y1
        del flow_y2
        del flow_xgr
        del flow_ygr
        del srcx
        del srcy
        del srcv
        del xname
        del yname
        del vname
        
        
        f=f+1
        
        # plt.imshow(img3, cmap="gray")
        # plt.colorbar(plt.imshow(mag_map1, cmap="gist_rainbow", alpha=0.7))
        # plt.title(Delta_DD)
        #plt.show()
        
        if (len(img_list) - 2) == f:
            break
        print("Finished Processing Triplet Dates: ", filename[:-4])
        os.system("printf '\033c'")
        
        gc.collect()
        
        
        
    print (" process is compeleted")
   
        

