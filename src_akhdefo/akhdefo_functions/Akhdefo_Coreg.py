###Start###
import cv2
import numpy as np
import os
import shutil
from osgeo import gdal
import os
from arosics import COREG_LOCAL

def raster_alignment(src_directory, ref_filename, delete_files=False):
    """
    Aligns raster images in a given source directory to a reference image.

    Args:
    src_directory (str): Path to the source directory containing images to align.
    ref_filename (str): Path to the reference image.
    delete_files (bool): If True, deletes the temporary directory created for alignment process. Defaults to False.

    Returns:
    str: Path to the directory containing all registered images.
    """
    # Get the current working directory
    current_dir = os.getcwd()

    # Create a temporary directory inside the current working directory
    temp_dir = os.path.join(current_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    ref = cv2.imread(ref_filename, cv2.IMREAD_COLOR) # Load the reference image in color
    gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) # Convert to grayscale for SIFT
    sift = cv2.SIFT_create()
    ref_kp, ref_des = sift.detectAndCompute(gray_ref, None) # Find keypoints in grayscale image

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Get the georeference info of the reference image
    ref_raster = gdal.Open(ref_filename)
    ref_geotransform = ref_raster.GetGeoTransform()
    ref_projection = ref_raster.GetProjection()

    # Iterate over each file in the directory
    for filename in os.listdir(src_directory):
        src_filename = os.path.join(src_directory, filename)

        src = cv2.imread(src_filename, cv2.IMREAD_COLOR) # Load the source image in color
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) # Convert to grayscale for SIFT
        src_kp, src_des = sift.detectAndCompute(gray_src, None) # Find keypoints in grayscale image

        matches = flann.knnMatch(src_des, ref_des, k=2)

        distances = [m.distance for m, _ in matches]
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        z_scores = [(m.distance - mean_distance) / std_distance for m, _ in matches]

        # Keep only the top 10 matches based on z-scores
        top_matches = [matches[i] for i in np.argsort(z_scores)[:10]]

        src_pts = np.float32([src_kp[m.queryIdx].pt for m, _ in top_matches]).reshape(-1, 1, 2)
        ref_pts = np.float32([ref_kp[m.trainIdx].pt for m, _ in top_matches]).reshape(-1, 1, 2)
        best_h, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)

        if len(top_matches) == 0:
            print(f"No good matches found for {filename}")
            continue

        aligned_src = cv2.warpPerspective(src, np.float32(best_h), (ref.shape[1], ref.shape[0]))

        # Crop the aligned source image to the size of the reference image
        height, width = ref.shape[:2]
        aligned_src = aligned_src[:height, :width, :]

        # Get the base name without extension
        base_filename = os.path.splitext(os.path.basename(src_filename))[0]
        # Use TIFF format for the registered image
        registered_filename = os.path.join(temp_dir, f"{base_filename}.tif")

        # Create a new georeferenced raster
        driver = gdal.GetDriverByName('GTiff')
        out_raster = driver.Create(registered_filename, width, height, 3, gdal.GDT_Byte)

        # Set the geo-transform to the one from the reference raster
        out_raster.SetGeoTransform(ref_geotransform)

        # Set the projection to the one from the reference raster
        out_raster.SetProjection(ref_projection)

        # Write the aligned and cropped image data to the raster bands
        for i in range(3):
            out_band = out_raster.GetRasterBand(i + 1)
            out_band.WriteArray(aligned_src[:, :, i])

        # Close the raster file
        out_raster = None

    if delete_files:
        shutil.rmtree(temp_dir)

    return temp_dir  # returns the directory containing all registered images




def Coregistration(input_Folder="", output_folder="", grid_res=20, min_reliability=60, 
                   window_size=(64,64), path_figures="", showFig=False, 
                   no_data=[0,0], single_ref_path="", step_size=3):
    """
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

    """

   

    def create_directory(directory):
        """Helper function to create a directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def generate_filename(ref_path, target_path, title_suffix):
        """Helper function to generate a clean filename for the saved figure."""
        ref_name = os.path.basename(ref_path)[:-4]
        target_name = os.path.basename(target_path)[:-4]
        return f"Coregistration_{ref_name}_{target_name}_{title_suffix}.jpg"

    def coregister_images(ref_path, target_path, out_path, path_figures, title_suffix):
        """Helper function to perform coregistration and optionally save results."""
        kwargs = {
            'grid_res': grid_res,
            'window_size': window_size,
            'path_out': out_path,
            'fmt_out': 'GTIFF',
            'min_reliability': min_reliability,
            'nodata': no_data
        }

        CRL = COREG_LOCAL(ref_path, target_path, **kwargs)
        CRL.correct_shifts()

        title = f"Coregistration: {os.path.basename(ref_path)[:-4]}_{os.path.basename(target_path)[:-4]}_{title_suffix}"
        CRL.view_CoRegPoints(figsize=(15,15), backgroundIm='ref', title=title, 
                             savefigPath=os.path.join(path_figures, generate_filename(ref_path, target_path, title_suffix)), 
                             showFig=showFig)

    # Create required directories
    create_directory(output_folder)
    create_directory(path_figures)

    img_list = [f for f in sorted(os.listdir(input_Folder)) if os.path.isfile(os.path.join(input_Folder, f))]
    
    if not img_list:
        raise ValueError("No raster images found in the provided input folder.")
    
    if single_ref_path:
        for item in img_list:
            coregister_images(single_ref_path, os.path.join(input_Folder, item), 
                              os.path.join(output_folder, item), path_figures, 'single_ref')
    else:
        title_suffix = 1
        
        # Register the first image to itself
        ref_image = os.path.join(input_Folder, img_list[0])
        coregister_images(ref_image, ref_image, 
                        os.path.join(output_folder, img_list[0]), 
                        path_figures, str(title_suffix))
        title_suffix += 1

        for i in range(0, len(img_list) - (step_size - 1), step_size): 
            ref_image = os.path.join(input_Folder, img_list[i])
            
            for offset in range(1, step_size): # Adjust the loop for the user-specified step size
                if i + offset < len(img_list):
                    target_image = os.path.join(input_Folder, img_list[i + offset])
                    coregister_images(ref_image, target_image, 
                                    os.path.join(output_folder, img_list[i + offset]), 
                                    path_figures, str(title_suffix))
            
            title_suffix += 1

        
    print("Process is completed")

# Example usage
# Coregistration(input_Folder="/path_to_input", output_folder="/path_to_output")
     
