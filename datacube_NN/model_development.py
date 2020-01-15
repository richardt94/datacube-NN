from datacube.utils.geometry import CRS
import numpy as np
import tifffile as tif
import pandas as pd

class DatacubeHarvester:
    """
    Harvests labelled examples from geospatial raster data stored in a 
    datacube into a format easily ingested by Keras models.
    """
    def __init__(self, dc, product, pixel_buffer = (10,10), **query):
        """
        Initialise a DatacubeHarvester, which contains attributes relevant
        to *all* samples: the datacube handle itself, product to be loaded
        from the datacube, relevant variables within that product, and
        other parts of the datacube query (e.g. time range for products
        with a time dimension). Only the first observation in the time range
        for products with a time dimension will be used - this is to ensure
        consistency between samples where different locations may have
        different numbers of observations within the same timeframe.
        """
        self.dc = dc
        self.prod = product
        self.query = query
        self.pixel_buffer = pixel_buffer
        
        prod_df = dc.list_products()
        
        self.native_crs = CRS(prod_df[prod_df['name']==prod_df]['crs'].array[0])
        spdims = prod_df[prod_df['name']==prod_df]['spatial_dimensions'].array[0]
        resolution = prod_df[prod_df['name']==prod_df]['resolution'].array[0]
        
        res_dict = {}
        for dim,res in zip(spdims):
            res_dict[dim] = res
        
        #we want everything as 'x','y' for consistency
        if 'long' in res_dict:
            res_dict['x'] = res_dict.pop('long')
            res_dict['y'] = res_dict.pop('lat')
        self.resolutions = res_dict
        
        self.spatial_buffer = {}
        #y-first indexing for consistency with DEA datacube (maybe all ODCs)
        self.spatial_buffer['y'] = pixel_buffer[0]*res_dict['y']
        self.spatial_buffer['x'] = pixel_buffer[1]*res_dict['x']
        
        #precompute the offsets from the sample site for each pixel in the final image
        self.reindex_masks = { dim : np.arange(-self.spatial_buffer[dim],
                                               self.spatial_buffer[dim]+res_dict[dim],
                                               res_dict[dim])
                              for dim in self.spatial_buffer }
        
    def process(savedir, xyw_df, x_col, y_col, w_col = None, nan_value = 0):
        """
        Save labelled examples defined by the list of (point, value) pairs xy,
        and optionally weights, in a format that can be ingested by a Keras
        ImageDataGenerator.
        """
        
        def _load_sample(point):
            #load a single point's sample data and return it as a numpy array.
            point = point.to_crs(self.native_crs)
            (sitex,sitey) = point.coords
            
            covars = dc.load(product = product,
                             x = (sitex-self.spatial_buffer['x'],sitex+self.spatial_buffer['x']),
                             y = (sitey-self.spatial_buffer['y'],sitey+self.spatial_buffer['y']),
                             crs = self.native_crs,
                             **query)
            
            if len(covars.variables) == 0: #this is probably an empty dataset from a bad query
                return None
            
            covars = covars.reindex(x = self.reindex_masks['x']+sitex,
                                    method = 'nearest',
                                    tolerance = abs(self.resolutions['x']/2))
            covars = covars.reindex(y = self.reindex_masks['y']+sitey,
                                    method = 'nearest',
                                    tolerance = abs(self.resolutions['y']/2))
            
            covars = covars.fillna(nan_value)
            
            if 'time' in covars.dims:
                covars = covars.isel(time=0)
            
            #don't do this - tifffile requires channels-last for saving
            #transpose to 'channels-first' format (default for TensorFlow)
            #covars = covars.to_array().transpose('y','x','variable')
            
            return covars.data

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
        metadata_df.to_csv(savedir+'/'+'samples.csv')
        