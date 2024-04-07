
###Start###
import glob
from pathlib import Path
import os
from glob import glob
from osgeo import gdal
import glob
import re
from osgeo import gdal, ogr
import os
import numpy as np
import os 
import glob
from pathlib import Path
from tqdm import tqdm

def Mosaic(Path_to_WorkingDir=r"", output_MosaicDir=r"" , img_mode=1,  file_ex=".tif"):

    """
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

    """
   


#5851965_1062413_2022-08-12_24a4_BGRN_SR_clip.tif
#5851965_1062413_2022-08-12_24a4_udm2_clip.tif
    
    Working_Dir=Path_to_WorkingDir
    output_MosaicDir=output_MosaicDir
    if img_mode==1:
        file_ex=file_ex
        #count_left=16
        #count_right=-22
    elif img_mode==0:
        file_ex=file_ex
        count_left=16
        count_right=-19
    else:
        print("""image mode is invalide 
        please enter 1 to process image data or 
        enter 0 to processes UDM2 Mask data. Alternatively, manually set the file_ex""")
        file_ex=file_ex

    #imglist = sorted(glob.glob(Working_Dir +"/"+ ext))
    def get_file_paths(directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(file_ex):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    # replace 'your_directory_path' with the path to the directory you want to iterate through
    imglist = get_file_paths(Working_Dir)
    
    
    
    
    if not os.path.exists(Working_Dir):
        os.makedirs(Working_Dir)

    if not os.path.exists(output_MosaicDir):
        os.makedirs(output_MosaicDir)

    outputfolder=output_MosaicDir
    
    
    # Regular expression pattern to extract dates
    date_pattern = re.compile(r"\d{4}[-]?\d{2}[-]?\d{2}")

    # Extract dates from file names
    dates = [re.search(date_pattern, file_name).group() for file_name in imglist]


    for idx, item1 in enumerate( imglist):
        #for item2 in imglist[idx+1:]:
            
        filepath1, filename1 = os.path.split(imglist[idx])
        if img_mode==1:
            
            #track_dates1=filename1[:10]
            # Regular expression pattern to extract dates
            date_pattern = re.compile(r"\d{4}[-]?\d{2}[-]?\d{2}")
            # Extract dates from the string
            track_dates1 = re.findall(date_pattern, filename1)


            #img_similar_datesList = [s for s in imglist if track_dates1 in s]
            img_similar_datesList = [s for s in imglist if ' '.join(str(date) for date in track_dates1) in s]

            merged_name= outputfolder + "/" + str(dates[idx] )  + ".tif"
            print("Mosaic file Name: " , img_similar_datesList )
            vrt = gdal.BuildVRT("merged1.vrt", img_similar_datesList)
            gdal.Translate(merged_name, vrt, xRes = 3.125, yRes = -3.125)
            vrt = None 
                     
        elif img_mode==0:
            #ext="*udm2*.tif"
            #count_left=16
            #count_right=-19
            #track_dates1=filename1[:10]
            # Regular expression pattern to extract dates
            date_pattern = re.compile(r"\d{4}[-]?\d{2}[-]?\d{2}")
            # Extract dates from the string
            track_dates1 = re.findall(date_pattern, filename1)


            #img_similar_datesList = [s for s in imglist if track_dates1 in s]
            img_similar_datesList = [s for s in imglist if ' '.join(str(date) for date in track_dates1) in s]

            merged_name= outputfolder + "/" + str(dates[idx] )  + ".tif"
            print("Mosaic file Name: " , img_similar_datesList )
            vrt = gdal.BuildVRT("merged1.vrt", img_similar_datesList)
            gdal.Translate(merged_name, vrt, xRes = 3.125, yRes = -3.125)
            vrt = None 
           
        
      
         
           
        
      
        



def rasterClip(rasterpath, aoi, outfilename):
    """
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
    """
    try:
        # Open the input raster and read its projection
        srcRaster = gdal.Open(rasterpath)
        srcProj = srcRaster.GetProjection()

        # Open the shapefile
        srcShape = ogr.Open(aoi)
        layer = srcShape.GetLayer()

        # Create a temporary file to store the clipping geometry
        tmp = 'temp.shp'
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(tmp):
            driver.DeleteDataSource(tmp)
        outShape = driver.CreateDataSource(tmp)
        outLayer = outShape.CreateLayer('temp', geom_type=ogr.wkbPolygon)
        outLayer.CreateFields(layer.schema)
        outDefn = outLayer.GetLayerDefn()
        outFeature = ogr.Feature(outDefn)

        # Loop through the features in the shapefile
        for inFeature in layer:
            geom = inFeature.GetGeometryRef()
            outFeature.SetGeometry(geom)
            outLayer.CreateFeature(outFeature)

        # Close the shapefiles
        outShape = None
        srcShape = None
        
        

        # Clip the raster
        gdal.Warp(outfilename, srcRaster, cutlineDSName=aoi, cropToCutline=True, dstNodata = np.nan)

        # Close the raster
        srcRaster = None

        # Delete the temporary shapefile
        driver.DeleteDataSource(tmp)
    except Exception as e:
        print(f"An error occurred: {e}") 

def Crop_to_AOI(Path_to_WorkingDir=r'', Path_to_AOI_shapefile=r"", output_CroppedDir=r"" , file_ex= '.tif'):

    """
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

    """
   

    if not os.path.exists(output_CroppedDir):
        os.makedirs(output_CroppedDir)
        
    Path_to_WorkingDir=Path_to_WorkingDir
    output_CroppedDir=output_CroppedDir

    cropped_dest=output_CroppedDir
    #imglist=glob.glob(Path_to_WorkingDir+ "/"+ '*.tif')
    #imglist.sort(key=os.path.getctime)
    def get_file_paths(directory):
        file_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(file_ex):
                    file_paths.append(os.path.join(root, file))
        return file_paths

    # replace 'your_directory_path' with the path to the directory you want to iterate through
    imglist = get_file_paths(Path_to_WorkingDir)
    
            
    Path_to_AOI_shapefile=Path_to_AOI_shapefile
    
    for  idx, item in tqdm(enumerate(imglist), total=len(imglist)):
        item=imglist[idx]
        filepath1, filename = os.path.split(item)
        name= cropped_dest + '/' + filename
        raster_path=item
        #print(name, " index: ", idx)
        path_to_file = name
        path = Path(path_to_file)
        rasterClip(raster_path, Path_to_AOI_shapefile ,name )
        
