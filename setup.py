#!/usr/bin/env python
import pathlib
import sys

from setuptools import find_packages, setup

__author__ = "Russell Tsuchida"
__copyright__ = "Copyright, Russell Tsuchida"
__credits__ = ["Russell Tsuchida", "Phuong Le"]
__license__ = "BSD"
__version__ = "2021.09.01"
__maintainer__ = "Russell Tsuchida"
__email__ = "russell.tsuchida@data61.csiro.au"
__status__ = "Development"

if sys.version_info < (3, 6):
    py_version = ".".join([str(n) for n in sys.version_info])
    raise RuntimeError(
        "Python-3.6 or greater is required, Python-%s used." % py_version
    )

# REPLACE WITH YOUR PROJECT DETAILS
PROJECT_URLS = {
    "Source Code": "https://github.com/RussellTsuchida/klr",
}

# REPLACE WITH YOUR PROJECT NAME
short_description = "Kernel Logistic Regression"

# REPLACE WITH YOUR PROJECT README NAME
readme_path = pathlib.Path(__file__).parent / "README.md"

long_description = readme_path.read_text()

PACKAGE_DIR = "code"

setup(
    name="klr",  # REPLACE WITH YOUR PROJECT NAME
    version=__version__,
    author="Russell Tsuchida",
    author_email="russell.tsuchida@data61.csiro.au",
    description=short_description,
    long_description=long_description,
    long_description_content_type="markdown",  # change if it's in markdown format
    platforms=["any"],
    license=__license__,
    keywords=["science", "logging"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        # "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(where="code"),
    package_dir={"": PACKAGE_DIR},
    url="https://github.com/RussellTsuchida/klr",  # REPLACE WITH YOUR PROJECT DETAILS
    project_urls=PROJECT_URLS,
    # REPLACE WITH YOUR PROJECT DEPENDENCIES
    install_requires=["numpy"],
)
