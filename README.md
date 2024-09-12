
# Akhdefo


[<img src="./_static/akhdefo_logo.svg" alt="Akhdefo Project Image" align="right" width="200px"/>](https://akhdefo.readthedocs.io/en/latest/index.html)
<em align="right">Click on the Logo to Navigate to the Main Page</em>



## Computer Vision for Slope Stability: Land Deformation Monitoring

## Background of Akh-Defo

**Akh-Defo** is derived from two distinct words: 
1. 'Akh' in the Kurdish language, representing land, earth, or soil (originating from the Kurdish Badini dialect).
2. 'Defo', a shorthand for the English term 'deformation'.

## Recommended Citation

Muhammad M, Williams-Jones G, Stead D, Tortini R, Falorni G, and Donati D (2022) Applications of Image-Based Computer Vision for Remote Surveillance of Slope Instability. *Front. Earth Sci.* 10:909078. doi: [10.3389/feart.2022.909078](https://doi.org/10.3389/feart.2022.909078)

## Updates

- **Deprecated:** Akhdefo version one. *Current recommendation:* Use Akhdefo version 2.
- **New Feature:** Cloud-based real-time processing capabilities.
- **Expansion:** Over 20 modules for end-to-end Python-based GIS and Image Processing, and Customized Figure generation.
- **Integration:** Access, filter, and download capabilities for Planet Labs data using the Planet Lab API.
- **Enhancement:** Orthorectification feature for satellite images.

------

# Installation of AkhDefo Software Method 1 on Linux Operating System

**Method 1 works well on Linux Operating System** **Use method 2 to install on Windows Operating System**

This guide provides step-by-step instructions on how to create a Conda environment using the provided [akhdefo_conda_env.yml](akhdefo_conda_env.yml) file. This file specifies all the necessary packages, including their versions, required for the project.

## Prerequisites

- **Conda**: Make sure you have either Anaconda or Miniconda installed on your system. If you do not have Conda installed, you can download it from the [official Anaconda website](https://www.anaconda.com/products/individual) or the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

## Creating the Environment

1. **Download the YAML File**: Ensure you have the `akhdefo_conda_env.yml` [Click Here to Download](akhdefo_conda_env.yml) file saved on your computer. This file should be in the directory where you intend to set up the environment, or you should note its path.

2. **Open a Terminal**: Open your terminal (or Command Prompt/PowerShell on Windows) and navigate to the directory containing the [akhdefo_conda_env.yml](akhdefo_conda_env.yml) file. You can navigate to the directory using the `cd` command followed by the path to the directory.

3. **Create the Environment**: Execute the following command in your terminal:

***Make sure to activate conda base environment***


```bash
conda activate base # if base environment not activated

conda install -c conda-forge mamba

mamba env create -f akhdefo_conda_env.yml

```

This command instructs Conda to create a new environment with the name specified in the YAML file (`akhdefo_env`) and install all the listed packages along with their dependencies.

4. **Activate the Environment**: After the environment has been successfully created, you can activate it using the following command:

```bash
conda activate akhdefo_env
```

Activating the environment will set up your terminal session to use the packages and their specific versions installed in this environment.

## Verifying the Installation

To ensure that the environment has been set up correctly and all packages are installed, you can list the installed packages using:

```bash
conda list
```

This command displays a list of all packages installed in the active conda environment.

## Troubleshooting

If you encounter any issues during the installation process, make sure you have the latest version of Conda installed, and try updating Conda using the following command:

```bash
conda update conda
```

If problems persist, refer to the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more detailed information on managing environments with YAML files.

## Conclusion

You now have a dedicated Conda environment for this project, containing all the necessary packages. This environment helps maintain project dependencies isolated from other projects, ensuring reproducibility and consistency.


If you have any questions or need further assistance, please feel free to open an issue in this repository

---

# Installation of AkhDefo Software Method 2 on Windows Operating System

This guide provides step-by-step instructions on how to create a Conda environment using the provided [ENV_Win.yml](ENV_Win.yml) file. This file specifies all the necessary packages, including their versions, required for the project.

## Prerequisites

- **Conda**: Make sure you have either Anaconda or Miniconda installed on your system. If you do not have Conda installed, you can download it from the [official Anaconda website](https://www.anaconda.com/products/individual) or the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

## Creating the Environment

1. **Download the YAML File**: Ensure you have the `ENV_Win.yml` [Click Here to Download](ENV_Win.yml) file saved on your computer. This file should be in the directory where you intend to set up the environment, or you should note its path.

2. **Open a Terminal**: Open your terminal (or Command Prompt/PowerShell on Windows) and navigate to the directory containing the [ENV_Win.yml](ENV_Win.yml) file. You can navigate to the directory using the `cd` command followed by the path to the directory.

3. **Create the Environment**: Execute the following command in your terminal:

***Make sure to activate conda base environment***


```bash
conda activate base # if base environment not activated

conda env create -f ENV_Win.yml

```

This command instructs Conda to create a new environment with the name specified in the YAML file (`akhdefo_env`) and install all the listed packages along with their dependencies.

4. **Activate the Environment**: After the environment has been successfully created, you can activate it using the following command:

```bash
conda activate akhdefo_env
```

Activating the environment will set up your terminal session to use the packages and their specific versions installed in this environment.

## Verifying the Installation

To ensure that the environment has been set up correctly and all packages are installed, you can list the installed packages using:

```bash
conda list
```

This command displays a list of all packages installed in the active conda environment.

## Troubleshooting

If you encounter any issues during the installation process, make sure you have the latest version of Conda installed, and try updating Conda using the following command:

```bash
conda update conda
```

If problems persist, refer to the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more detailed information on managing environments with YAML files.

## Conclusion

You now have a dedicated Conda environment for this project, containing all the necessary packages. This environment helps maintain project dependencies isolated from other projects, ensuring reproducibility and consistency.

---

**Alternatively you can manually create conda environment and install dependencies**

***Follow these steps to install the Akhdefo software manually:***

1. Create a new Python Anaconda environment using the command:

   ```python
   conda create --name akhdefo_env

   conda activate akhdefo_env
   ```

2. Create  Anaconda environment and install the following libraries with Anaconda

```yaml

# Install Python 3.8 and essential packages
#!conda install python=3.8 pip -y

# Install scientific and plotting libraries
!conda install -c conda-forge numpy matplotlib pandas scipy seaborn statsmodels plotly tqdm -y

# Install geospatial libraries
!conda install -c conda-forge geopandas shapely gdal rasterio rioxarray pykrige gstools -y

# Install additional tools for geospatial analysis and visualization
!pip install opencv-python earthpy cmocean -y

# Install web frameworks and general libraries
!conda install flask requests -y

# Install Sphinx and documentation tools
!conda install -c conda-forge recommonmark sphinx nbsphinx sphinx-book-theme myst-parser -y

# Install machine learning libraries
!conda install -c conda-forge scikit-learn scikit-image  -y

!pip install -U tensorflow -y

# Install widget and interface tools
!conda install -c conda-forge ipywidgets json5 -y

# Install miscellaneous libraries
!conda install -c conda-forge hyp3_sdk glob2 xmltodict -y


! pip install akhdefo_functions arosics asf_search cmocean datetime earthpy flask geocube geopandas glob2 
! pip install gstools hyp3_sdk imutils ipywidgets json3 matplotlib matplotlib_scalebar numpy opencv-python pandas 
! pip install pathlib pillow planet plotly pykrige pyproj python-dateutil rasterio regex requests 
! pip install rioxarray scikit-image scikit-learn scipy seaborn shapely shutils simplekml statsmodels 
! pip install subprocess32 tensorflow



```
3. Download the Python package requirement file: [pip_req.txt](pip_req.txt).

4. Install required Python packages with the command:

   ```python
   pip install -r pip_req.txt
   ```

5. Install Akhdefo using the following command:

   ```python
   pip install akhdefo-functions
   ```