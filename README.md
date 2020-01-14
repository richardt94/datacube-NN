# datacube-NN
Bits and pieces for training and predicting with CNNs (via Keras) on data from opendatacube databases.

## Dependencies
Functionality will depend generally on having a working [opendatacube](https://github.com/opendatacube/datacube-core) installation with desired datasets indexed.

Example applications in Jupyter notebooks use the [Digital Earth Australia](https://docs.dea.ga.gov.au/) datacube installation on [NCI](https://nci.org.au/) and therefore require an NCI account with read permissions for the DEA datacube (provided through project code v10).

Other requirements:
- Python 3.6
- Tensorflow (developed on 1.14.0)
