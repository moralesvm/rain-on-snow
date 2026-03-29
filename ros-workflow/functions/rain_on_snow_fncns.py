import os
import pandas
import xarray
import s3fs
#import rasterio
from rasterio import features
import pyproj
import geopandas
import exactextract
from exactextract import exact_extract
from shapely.geometry import box
#import rioxarray

# --- CONSTANTS ---
# NWM projection
nwm_proj = pyproj.Proj(proj='lcc',
                       lat_1=30.,
                       lat_2=60.,
                       lat_0=40.0000076293945, lon_0=-97.,
                       a=6370000, b=6370000)

def read_nwmData(awsPath,variables,timerange):

    s3_path = awsPath
    # Connect to S3
    #---------------
    s3 = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=s3_path, s3=s3, check=False)

    # Lazy load dataset
    #--------------------
    ds = xarray.open_zarr(store=store, consolidated=True)

    # NWM data doesn't have a CRS explicitely assigned, we'll do it here
    ds = ds.rio.write_crs(nwm_proj.crs)
    ds = ds.rio.write_coordinate_system()

    # Domain subset
    #---------------
    max_lon = ds["x"].max()
    max_lat = ds["y"].max()

    # NOTE: Domain is hardcoded for now
    ds_sub = ds[variables].sel(
    time = timerange,
    x = slice(1.4e6,max_lon),
    y = slice(0,1.5e6))

    return ds_sub

def ros_musselman(dataset):

       # Convert units
       #---------------
       # NOTE:
       # QRAIN = Rainfall rate on the ground (mm/s)
       # SNEQV = Snowfall water equivalent (kg/m2)
       # 1 kg/m² = 1 mm water equivalent
       dataset["QRAIN_mm"] = dataset["QRAIN"] * 3 * 3600
       dataset["QRAIN_mm"].attrs["units"] = "mm"

       # Summarize to daily
       #--------------------
       rain_daily = dataset["QRAIN_mm"].resample(time="1D").sum()
       sneqv_daily = dataset["SNEQV"].resample(time="1D").mean()

       # Combine them back to a single dataset to ease computations
       ds_daily = xarray.Dataset({
           "QRAIN_daily_mm": rain_daily,
           "SNEQV_daily_mm": sneqv_daily})

       # ROS condition - Binary flag per grid-cell
       #--------------------------------------------
       ros_daily_mask = ((ds_daily["QRAIN_daily_mm"] > 10) &
                   (ds_daily["SNEQV_daily_mm"] > 10)).astype(int)

       # Assign NWM projection
       #------------------------
       ros_daily_mask = ros_daily_mask.rio.write_crs(nwm_proj.crs)

       return ros_daily_mask

def define_ros_zone(daily_ros_mask, threshold):

    # ROS zone:
    # All grid cells w/ at least 1 ROS day per year in average
    # Example: A grid cell must contain at least 43 ROS days over the full NWM v3
    # Retrospective period (2010-2022) to be included in the ROS zone
    #annual_count = ros_mask.groupby('time.year').sum(dim='time')

    yearly_presence = daily_ros_mask.groupby('time.year').any(dim='time').compute() # At least 1 ROS day per year
    ros_zone_mask = (yearly_presence.sum(dim='year') == threshold).astype(int)

    # Assign NWM projection
    ros_zone_mask = ros_zone_mask.rio.write_crs(nwm_proj.crs)

    return ros_zone_mask

def get_ros_basins(ros_zone,shpPath):

    # Read basins shapefile
    #----------------------
    shp = geopandas.read_file(shpPath)
    # Reproject basins to NWM projection
    shp_prj = shp.to_crs(nwm_proj.crs)

    # Select basins within the domain
    #---------------------------------
    # Get raster bounds
    raster_bounds = box(*ros_zone.rio.bounds())
    # Filter polygons intersecting raster
    shp_prj_subset = shp_prj[shp_prj.intersects(raster_bounds)]

    # Extract % of ROS zone per basin
    #---------------------------------
    #ros_zone_bsns_df = exact_extract(ros_zone, shp_prj_subset, ['sum','count','mean'], # For testing
    ros_zone_bsns_df = exact_extract(ros_zone, shp_prj_subset, ['mean'],
                                     include_cols='GAGE_ID', output='pandas')

    ros_zone_bsns_df['Perc_ROS'] = (ros_zone_bsns_df['mean'] * 100).round(0)
    ros_zone_bsns_df.drop(columns=['mean'], inplace=True)

    return ros_zone_bsns_df

def get_ros_events(ros_daily_mask,shpPath,t_chunks):

    # Read basins shapefile
    #----------------------
    shp = geopandas.read_file(shpPath)
    # Reproject basins to NWM projection
    shp_prj = shp.to_crs(nwm_proj.crs)

    # Select basins within the domain
    #---------------------------------
    # Get raster bounds
    raster_bounds = box(*ros_daily_mask.rio.bounds())
    # Filter polygons intersecting raster
    shp_prj_subset = shp_prj[shp_prj.intersects(raster_bounds)]

    # Let's re-chunk the ros daily dataset for faster processing
    #------------------------------------------------------------
    ros_daily_mask = ros_daily_mask.chunk({'time': t_chunks, 'x': -1, 'y': -1}).persist()

    # Process % of ros per day and basin
    #-----------------------------------
    df_ros_evs = exact_extract(ros_daily_mask, shp_prj_subset, ['mean'],include_cols='GAGE_ID',
                               output='pandas',strategy='raster-sequential')

    # Create a mapping dictionary: band_index = Date
    date_map = {f"band_{i+1}_mean": d for i, d in enumerate(ros_daily_mask.time.values)}

    # Reshape the results (exactextract returns wide format for time)
    daily_ros_evs = df_ros_evs.melt(id_vars='GAGE_ID', var_name='layer', value_name='mean_ros')

    # Final Percentage calculation
    daily_ros_evs['Perc_ROS'] = (daily_ros_evs['mean_ros'] * 100).round(1)

    # Add time to the final df and clean columns and order
    daily_ros_evs['Date'] = daily_ros_evs['layer'].map(date_map)
    daily_ros_evs.drop(['layer', 'mean_ros'], axis=1, inplace=True)
    daily_ros_evs = daily_ros_evs[['GAGE_ID', 'Date', 'Perc_ROS']]

    return daily_ros_evs





