# to build run : python3 setup.py sdist bdist_wheel
import setuptools
from emagpy import EMagPy_version


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emagpy",
    version=EMagPy_version,
#    version="1.1.11",
    author="HKEx",
    description="API for FDEM inversion and data manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/hkex/emagpy",
    packages=setuptools.find_packages(),
    install_requires=['numpy','matplotlib','pandas','scipy', 'pyproj', 'spotpy','joblib','tqdm','rasterio'],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
