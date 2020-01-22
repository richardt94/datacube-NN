"""
Script to perform prediction on a single tile. To be used with GPU nodes on NCI.
This is a simple case using a single hardcoded tile index.
"""

import numpy as np
import xarray as xr
import datacube

from tensorflow.keras.models import load_model

from tensorflow.keras import backend as K
import tensorflow

K.set_session(tensorflow.Session(config=tensorflow.ConfigProto(intra_op_parallelism_threads=48,
                                                               inter_op_parallelism_threads=48))) 

def get_padded_tile(gw,tilvals,padding=(25,25),**query):
    """
    Returns an xarray.DataArray corresponding to the tile index provided in tilevals.
    
    args:
    gw: a GridWorkflow object which can map tile indices to netCDF files we can load.
    tilvals: The central tile index we want to get some padding around.
    padding: padding values (in pixels). Given in order (y,x)
    query: query for the gridworkflow object. This is not optional, list_tiles() refuses
    to work without at least being passed a product
    
    TODO handle if the tile exists and can be loaded but one of its neighbours cannot - 
    in this case pad with some constant value
    """

    tls = gw.list_tiles(**query)
    
    tiles = []

    for yoffset in range(1,-2,-1):
        row = []
        for xoffset in range(-1,2):
            tilekey = (tilvals[0]+xoffset,tilvals[1]+yoffset,tilvals[2])
            tile = gw.load(tls[tilekey])
            tile = tile.to_array().isel(time=0)
            row.append(tile)
        tiles.append(row)
        
    tilepx = tiles[1][1].sizes
    
    #combine everything together
    single_array = xr.concat([xr.concat(row,dim='x') for row in tiles],dim='y')
    
    #sort x-ascending y-descending
    single_array = single_array.sortby('y',ascending=False)
    single_array = single_array.sortby('x',ascending=True)
    
    #select padded region
    single_array = single_array.isel(y=slice(tilepx['y']-padding[0],2*tilepx['y']+padding[0]),
                                    x=slice(tilepx['x']-padding[1],2*tilepx['x']+padding[1]))
    
    single_array = single_array.transpose('y','x','variable')
    return single_array

tilvals = (13, -36, np.datetime64('2000-01-01T00:00:00.000000000'))

dc = datacube.Datacube()
gw = datacube.api.GridWorkflow(dc.index,product='fc_percentile_albers_annual')

padsize = (25,25)
convshape = tuple([2*pad + 1 for pad in padsize])

tila = get_padded_tile(gw, tilvals, padding = padsize,
                         time=('1999-07-01','2000-06-30'),
                         product='fc_percentile_albers_annual')


pxvals = np.zeros([tila.shape[0]-convshape[0]+1,tila.shape[1]-convshape[1]+1])

#batch size advantage will increase for GPU compute.
batch_len = 64

model = load_model('SOC_national_model.h5')

for cy in range(pxvals.shape[0]):
    for batch in range(pxvals.shape[1]//batch_len + 1):
        batch_input = []
        cur_batch_size = min(batch_len,pxvals.shape[1]-batch*batch_len)

        if cur_batch_size <= 0:
            continue
            
        #slice the image for the whole batch now
        batch_pred_img = tila.isel(y=slice(cy,cy+convshape[0]),
                                   x=slice(batch*batch_len,
                                           convshape[1]+batch*batch_len+cur_batch_size)
                                  ).data
        
        #build the batch in ingestible np array form
        for cx in range(cur_batch_size):
            predimg = batch_pred_img[:,cx:cx+convshape[1],:]
            batch_input.append(predimg)
        
        batch_input = np.array(batch_input)

        batch_pred = model.predict(batch_input)
        pxvals[cy,batch*batch_len:batch*batch_len+cur_batch_size] = batch_pred[:,0]
        print(batch_pred.shape)
        print('predicted pixels',cy,str(batch*batch_len)+':'+str(batch*batch_len+cur_batch_size))
        
np.save('tile_out.npy',pxvals)

