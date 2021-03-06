from datacube.utils.geometry import CRS
import numpy as np
import tifffile
import pandas as pd
import xarray as xr
import os
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import apply_affine_transform

class DatacubeHarvester:
    """
    Harvests labelled examples from geospatial raster data stored in a 
    datacube into a format easily ingested by Keras models.
    """
    def __init__(self, dc, product, pixel_buffer = (10,10),
                 geotiffs=[], dask_chunks={'x':1000,'y':1000},
                 interpolation = None, **query):
        """
        Initialise a DatacubeHarvester, which contains attributes relevant
        to *all* samples: the datacube handle itself, product to be loaded
        from the datacube, relevant variables within that product, and
        other parts of the datacube query (e.g. time range for products
        with a time dimension). Only the first observation in the time range
        for products with a time dimension will be used - this is to ensure
        consistency between samples where different locations may have
        different numbers of observations within the same timeframe.
        
        geotiffs is an array of geotiff filenames. These should have the same
        CRS as the datacube product being loaded. They are assumed to be v big
        and dask chunking is used to open them by default to deal with this.
        
        Geotiffs may be interpolated (by default nearest-neighbour, no interp)
        when resampling to the same grid as the datacube product samples.
        """
        self.dc = dc
        self.prod = product
        self.query = query
        self.pixel_buffer = pixel_buffer
        
        prod_df = dc.list_products()
        
        self.native_crs = CRS(prod_df[prod_df['name']==product]['crs'].array[0])
        spdims = prod_df[prod_df['name']==product]['spatial_dimensions'].array[0]
        resolution = prod_df[prod_df['name']==product]['resolution'].array[0]
        
        res_dict = {}
        for dim,res in zip(spdims,resolution):
            res_dict[dim] = res
        
        #we want everything as 'x','y' for consistency
        if 'long' in res_dict:
            res_dict['x'] = res_dict.pop('long')
            res_dict['y'] = res_dict.pop('lat')
        self.resolutions = res_dict
        
        spatial_buffer = {}
        #y-first indexing for consistency with DEA datacube (maybe all ODCs)
        spatial_buffer['y'] = pixel_buffer[0]*res_dict['y']
        spatial_buffer['x'] = pixel_buffer[1]*res_dict['x']
        
        #precompute the offsets from the sample site for each pixel in the final image
        self.reindex_masks = { dim : np.arange(-spatial_buffer[dim],
                                               spatial_buffer[dim]+res_dict[dim],
                                               res_dict[dim])
                              for dim in spatial_buffer }
        #include the tolerance in the dc.load to avoid missing values on
        #edge of the spatial buffer
        self.tol =   {
                'x': abs(self.resolutions['x'])/2,
                'y': abs(self.resolutions['y'])/2
                }
        
        self.qbuffer = {
            'x': abs(spatial_buffer['x'])+self.tol['x'],
            'y': abs(spatial_buffer['y'])+self.tol['y']
        }
        
        self.extra_data = False
        if len(geotiffs):
            self.extra_data = True
            self.tiff_list = []
            self.indexing_order = [] #to assist in slicing the xarray dataset correctly
            self.tiff_res=[]
            for fname in geotiffs:
                tiffds = xr.open_rasterio(fname, chunks=dask_chunks)
                #are the coordinates in descending order? -1 for yes 1 for no
                self.indexing_order.append({'x': 1-2*(tiffds.x[-1]<tiffds.x[0]),
                                            'y': 1-2*(tiffds.y[-1]<tiffds.y[0])})
                self.tiff_list.append(tiffds)
                self.tiff_res.append({'x': abs(tiffds.x[1] - tiffds.x[0]),
                                     'y': abs(tiffds.y[1] - tiffds.y[0])})
                
            
        
    def process(self,savedir, xyw_df, x_col, y_col, w_col = None, nan_value = 0):
        """
        Save labelled examples defined by the list of (point, value) pairs xy,
        and optionally weights, in a format that can be ingested by a Keras
        ImageDataGenerator.
        """
        
        def _load_sample(point):
            #load a single point's sample data and return it as a numpy array.
            point = point.to_crs(self.native_crs)
            (sitex,sitey) = point.coords[0]
            
            
            covars = self.dc.load(product = self.prod,
                             x = (sitex-self.qbuffer['x'],
                                  sitex+self.qbuffer['x']),
                             y = (sitey-self.qbuffer['y'],
                                  sitey+self.qbuffer['y']),
                             crs = self.native_crs,
                             **self.query)
            
            if len(covars.variables) == 0: #this is probably an empty dataset from a bad query
                return None
            
            covars = covars.reindex(x = self.reindex_masks['x']+sitex,
                                    method = 'nearest',
                                    tolerance = self.tol['x'])
            covars = covars.reindex(y = self.reindex_masks['y']+sitey,
                                    method = 'nearest',
                                    tolerance = self.tol['y'])
            
            
            if 'time' in covars.dims:
                covars = covars.isel(time=0)
                covars = covars.reset_coords(drop=True)
            
            #needs to be a dataarray to turn to numpy
            covars = covars.to_array()
            
            if self.extra_data:
                for res,iord,tiffds in zip(self.tiff_res, self.indexing_order, self.tiff_list):
                    #select a window from the tiff and load to memory
                    tiffw = tiffds.sel(
                        x=slice(sitex-iord['x']*self.qbuffer['x'],
                                sitex+iord['x']*self.qbuffer['x']),
                        y=slice(sitey-iord['y']*self.qbuffer['y'],
                                sitey+iord['y']*self.qbuffer['y'])
                    ).compute()
                    #reindex to be compatible with covars, tolerance to one pixel
                    #of the tiff
                    
                    tiffw = tiffw.reindex(x = self.reindex_masks['x']+sitex,
                                        method = 'nearest',
                                        tolerance = res['x'])
                    tiffw = tiffw.reindex(y = self.reindex_masks['y']+sitey,
                                        method = 'nearest',
                                        tolerance = res['y'])
                    tiffw = tiffw.rename({'band':'variable'})
                    
                    covars = xr.concat([covars, tiffw], dim='variable')
            covars = covars.fillna(nan_value)                    
            
            return covars.data

        if not os.path.exists(savedir):
            os.makedirs(savedir)
            
        if not os.path.isdir(savedir):
            raise ValueError('Savedir cannot point to a file')
        
        metadata_tab = []
        for idx, point in enumerate(xyw_df[x_col]):
            point_data = _load_sample(point)
            
            if point_data is not None:
                #save the image
                fname = str(idx)+'.tif'
                tifffile.imsave(savedir+'/'+fname,data=point_data)
                #append all relevant metadata
                metadata = [point,xyw_df[y_col][idx],fname]
                if w_col is not None:
                    metadata.append(xyw_df[w_col][idx])
                
                metadata_tab.append(metadata)
                
        headers = ['Point','Value','Filename']
        if w_col is not None:
            headers.append('Sample Weight')
        
        metadata_df = pd.DataFrame(metadata_tab,columns=headers)
        
        #save the sample data as a csv for loading and reuse with
        #Keras flow_from_dataframe
        metadata_df.to_pickle(savedir+'/'+'samples.pkl')
        

class MultiTiffGenerator(Sequence):
    """
    Feed images created by the DatacubeHarvester. This functionality
    is necessary because Pillow does not support images with more than
    four channels. Keras does have *some* preprocessing layers which
    may be able to help with restoring some of the built-in transformations
    available to the normal ImageDataGenerator in Keras.
    """
    
    def __init__(self, gen_df, x_col = 'Filename', y_col = 'Value',
                 w_col = None, batch_size = 32, shuffle = True,
                 img_dir = '', rotate = False):
        self.batch_size = batch_size
        
        self.length = len(gen_df)//batch_size
        
        self.shuffle = shuffle
        
        if self.shuffle:
            self.gen_df = gen_df.sample(frac=1).reset_index(drop=True)
        else:
            self.gen_df = gen_df
            
        self.fnames = gen_df[x_col].to_numpy()
        self.yvals = gen_df[y_col]
        
        if w_col is not None:
            self.weights = gen_df[w_col]
            
        self.x_col = x_col
        self.y_col = y_col
        self.w_col = w_col
        
        self.img_dir = img_dir
        
        self.rotate = rotate
    
    def __getitem__(self,index):
        minidx,maxidx = (index*self.batch_size,(index+1)*self.batch_size)
        y = self.yvals[minidx:maxidx]
        
        if self.w_col is not None:
            w = self.weights[minidx:maxidx]
        
        #vectorising file I/O is stupid so let's use a for loop
        samples = []
        for fname in self.fnames[minidx:maxidx]:
            image = tifffile.imread(os.path.join(self.img_dir,fname))
            if self.rotate:
                image = apply_affine_transform(image,theta=np.random.randint(4)*90)
            samples.append(image)
        
        X = np.stack(samples)
        
        #TF wants channels_last and tifffile reads channels-first
        X = np.moveaxis(X,1,3)
        
        if self.w_col is not None:
            return (X,y,w)
        else:
            return (X,y)
        
        
    def __len__(self):
        return self.length
    
    def on_epoch_end(self):
        if self.shuffle:
            self.gen_df = self.gen_df.sample(frac=1).reset_index(drop=True)
            self.fnames = self.gen_df[self.x_col].to_numpy()
            self.yvals = self.gen_df[self.y_col].to_numpy()
        
            if self.w_col is not None:
                self.weights = self.gen_df[self.w_col]
