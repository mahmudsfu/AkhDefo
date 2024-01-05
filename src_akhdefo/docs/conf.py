# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os 
import sys 

sys.path.insert(0, os.path.abspath('..'))

project = 'AkhDefo Software'
copyright = '2024, Mahmud Mustafa Muhammad'
author = 'Mahmud Mustafa Muhammad'
release = '2.2.61'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc', 'myst_parser' , 'nbsphinx' , 'autodocsumm' ,  "pallets_sphinx_themes"]


html_short_title = "Akhdefo"


html_logo = '_static/akhdefo_logo.svg'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme='sphinx_book_theme'
# import sphinx_adc_theme
#html_theme = 'furo'
# html_theme_path = [sphinx_adc_theme.get_html_theme_path()]

# import hachibee_sphinx_theme
# html_theme = 'hachibee'
# html_theme_path = [hachibee_sphinx_theme.get_html_themes_path()]


# Activate the theme.
html_theme = 'flask'

html_static_path = ['_static']

# autodoc_default_options = {
#     'autosummary': True,
# }

nbsphinx_allow_errors = True

####################
