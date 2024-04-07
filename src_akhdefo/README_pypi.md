
# Akhdefo


[<img src="https://akhdefo.readthedocs.io/en/latest/_images/akhdefo_logo.svg" alt="Akhdefo Project Image" align="right" width="200px"/>](https://akhdefo.readthedocs.io/en/latest/index.html)
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

## Installation of AkhDefo Software Method 1 through Creating a Conda Environment from the YAML File

This guide provides step-by-step instructions on how to create a Conda environment using the provided [akhdefo_conda_env.yml](akhdefo_conda_env.yml) file. This file specifies all the necessary packages, including their versions, required for the project.

## Prerequisites

- **Conda**: Make sure you have either Anaconda or Miniconda installed on your system. If you do not have Conda installed, you can download it from the [official Anaconda website](https://www.anaconda.com/products/individual) or the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html).

## Creating the Environment

1. **Download the YAML File**: Ensure you have the `akhdefo_conda_env.yml` file saved on your computer. This file should be in the directory where you intend to set up the environment, or you should note its path.

2. **Open a Terminal**: Open your terminal (or Command Prompt/PowerShell on Windows) and navigate to the directory containing the [akhdefo_conda_env.yml](akhdefo_conda_env.yml) file. You can navigate to the directory using the `cd` command followed by the path to the directory.

3. **Create the Environment**: Execute the following command in your terminal:

```bash
conda env create -f akhdefo_conda_env.yml
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

If you have any questions or need further assistance, please feel free to open an issue in this repository


## Installation of AkhDefo Software Method 2

Follow these steps to install the Akhdefo software:

1. Create a new Python Anaconda environment using the command:

   ```python
   conda create --name akhdefo_env
   ```

2. Create  Anaconda environment and install the following libraries with Anaconda

```yaml

dependencies:
  - python=3.8  # Assuming Python 3.8, can be changed as needed
  - cmocean
  - pip
  - opencv
  - earthpy
  - flask
  - geopandas
  - glob2
  - gstools
  - hyp3_sdk
  - ipywidgets
  - json5
  - matplotlib
  - numpy
  - gdal
  - pandas
  - recommonmark
  - sphinx
  - nbsphinx
  - sphinx-book-theme
  - myst-parser
  - plotly
  - pykrige
  - rasterio
  - requests
  - rioxarray
  - scipy
  - seaborn
  - shapely
  - scikit-image  # skimage
  - scikit-learn  # sklearn
  - statsmodels
  - tensorflow
  - tqdm
  - xmltodict

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
