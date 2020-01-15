import pytest
import datacube
import pandas as pd
import numpy as np
import tifffile
from datacube_NN import model_development

"""
Tests for datacube_NN.model_development. These tests assume access to the
production Digital Earth Australia datacube on NCI, in order to load products
for test cases.
"""

@pytest.fixture
def dea_dc():
    return datacube.Datacube()

@pytest.fixture
def harvester_fc(dea_dc):
    return model_development.DatacubeHarvester(dea_dc,'fc_percentile_albers_annual',time=('2013-05-01','2014-05-01'))

def test_data_harvester_metadata(harvester_fc):
    #make sure all the things are correct given the product we created the harvester with.
    
    assert harvester_fc.resolutions['x'] == 25
    assert harvester_fc.resolutions['y'] == -25
    
    assert harvester_fc.spatial_buffer['x'] == 250
    assert harvester_fc.spatial_buffer['y'] == -250

def test_save_from_dataframe(harvester_fc, dea_dc, tmpdir):
    test_df = pd.read_pickle('tests/test_table.pkl')
    #process the dataframe and save the resulting tiffs
    harvester_fc.process(str(tmpdir),test_df,'points','TC')
    
    out_df = pd.read_pickle(str(tmpdir.join('samples.pkl')))
    #none of the samples should have failed to load
    assert len(out_df)==len(test_df)
    #Make sure the y-values are the same
    assert (out_df['Value'] == test_df['TC']).all()
    #make sure the correct window of the product is being loaded
    
    for loc,fname in zip(out_df['Point'],out_df['Filename']):
        sitex,sitey = loc.coords[0]
        test_dc_xr = dea_dc.load(product='fc_percentile_albers_annual',
                         x=(sitex-267.5,sitex+267.5),
                         y=(sitey-267.5,sitey+267.5),
                         crs='EPSG:3577',
                         time=('2013-05-01','2014-05-01')).isel(time=0).to_array()
        test_dc_xr = test_dc_xr.reindex(x=sitex+np.arange(-250,275,25),y=sitey+np.arange(250,-275,-25),method='nearest',tolerance=12.5)
        
        test_dc_array = test_dc_xr.data
        
        test_harvested_array = tifffile.imread(str(tmpdir.join(fname)))
        
        assert test_dc_array.shape == test_harvested_array.shape
        assert np.array_equal(test_dc_array,test_harvested_array)