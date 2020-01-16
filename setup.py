import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datacube-NN",
    version="0.1",
    author="Richard Taylor",
    author_email="richard.taylor@ga.gov.au",
    description="Useful scripts for training models and making predictions on geospatial raster data stored in a datacube",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richardt94/datacube-NN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    
    install_requires = [
        'numpy',
        'pandas',
        'datacube',
        'tifffile'
    ]
    
)