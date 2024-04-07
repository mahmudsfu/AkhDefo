# from setuptools import setup


# from setuptools import setup, Command
# import subprocess

# class BuildDoc(Command):
#     """A custom command to build Sphinx documentation."""

#     description = 'build Sphinx documentation'
#     user_options = []

#     def initialize_options(self):
#         pass

#     def finalize_options(self):
#         pass

#     def run(self):
#         subprocess.call(['sphinx-build', '-b', 'html', 'docs', 'docs/_build/html'])
# # from sphinx.setup_command import BuildDoc
# cmdclass = {'build_sphinx': BuildDoc}



# with open("README.md", "r") as fh:
#     long_description = fh.read()

# setup(
    
#     name='akhdefo_functions',
#     version='2.2.58',    
#     description='Land Deformation Monitoring Using Optical Satellite Imagery',
    
#     url='https://github.com/mahmudsfu/AkhDefo',
#     author='Mahmud Mustafa Muhammad',
#     author_email='mahmud.muhamm1@gmail.com',
#     license='Academic Free License (AFL)',
#     packages=['akhdefo_functions'],
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     zip_safe=False,
#     classifiers=[
#         'Development Status :: 3 - Alpha',
#         'Intended Audience :: Science/Research',
#         'License :: OSI Approved :: Academic Free License (AFL)',  
#         'Operating System :: Microsoft :: Windows :: Windows 10',
#         'Operating System :: POSIX :: Linux',        
#         'Programming Language :: Python :: 3',  'Programming Language :: Python :: 3.6',  
#         'Programming Language :: Python :: 3.7', "Programming Language :: Python :: 3.8",
#         "Programming Language :: Python :: 3.9",
#         "Programming Language :: Python :: 3.10",

#     ], 

#    )


import subprocess

from setuptools import Command, setup


class BuildDoc(Command):
    """A custom command to build Sphinx documentation."""
    
    description = 'build Sphinx documentation'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.call(['sphinx-build', '-b', 'html', 'docs', 'docs/_build/html'])


# Define command class for building Sphinx documentation
cmdclass = {'build_sphinx': BuildDoc}

# Read the long description from the README file
with open("./README_pypi.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup configuration
setup(
    name='akhdefo_functions',
    version='2.5.0',
    description='Land Deformation Monitoring Using Optical Satellite Imagery',
    url='https://github.com/mahmudsfu/AkhDefo',
    author='Mahmud Mustafa Muhammad',
    author_email='mahmud.muhamm1@gmail.com',
    license='Academic Free License (AFL)',
    packages=['akhdefo_functions'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Academic Free License (AFL)',  
        'Operating System :: Microsoft :: Windows :: Windows 10',
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',  
        'Programming Language :: Python :: 3.6',  
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ], 
    cmdclass=cmdclass
)
