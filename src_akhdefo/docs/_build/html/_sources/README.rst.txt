Akhdefo
=======

Computer Vision for Slope Stability: Land Deformation Monitoring

.. image:: _static/akhdefo_logo.svg
   :alt: Akhdefo Project Image
   :align: center
   :width: 100px
   

Background of Akh-Defo
----------------------

Akh-Defo is a combination of two different words: 1) 'Akh' in Kurdish language means land, earth, or soil (origin of the word is from Kurdish Badini dialect) 2) 'Defo' is short for the English word 'deformation'.

Recommended Citation
--------------------

Muhammad M, Williams-Jones G, Stead D, Tortini R, Falorni G, and Donati D (2022) Applications of Image-Based Computer Vision for Remote Surveillance of Slope Instability. Front. Earth Sci. 10:909078. doi: 10.3389/feart.2022.909078

Updates
-------

* Akhdefo version one is deprecated; please use Akhdefo version 2.
* Akhdefo can now run on the cloud for real-time processing.
* Akhdefo now consists of more than 20 modules that perform end-to-end python-based GIS and Image Processing, and Customized Figure generation.
* Capability to access, filter, and download Planet Labs data using Planet Lab API.
* Capability to orthorectify satellite images.

Installation of Akhdefo Software
--------------------------------

1. `Download the Anaconda environment file akhdefov2.yml <akhdefov2.yml>`_ 
2. `Download the Python package requirement file pip_req.txt <pip_req.txt>`_ 
3. Create a new Python Anaconda environment using the command below:

.. code-block:: python

    conda env create -f akhdefov2.yml

4. Install required Python packages using the command below:

.. code-block:: python

    pip install -r pip_req.txt

5. Now install Akhdefo using the command below:

.. code-block:: python

    pip install akhdefo-functions
