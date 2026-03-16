import os
import pandas
import xarray
import s3fs
#import rasterio
from rasterio import features
import pyproj
import geopandas
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

    # Create a lookup table for later: {index: GAGE_ID}
    lookup = shp_prj['GAGE_ID'].to_dict()

    # Rasterize shp polygons
    #-------------------------
    # This will make it easier to mask later
    # Create a list of (geometry, identifier) tuples
    shapes = zip(shp_prj.geometry, shp_prj.index)

    # Create a manual mask aligned with the Xarray coordinates
    mask = features.rasterize(
        shapes=shapes,
        out_shape=(ros_zone.rio.height, ros_zone.rio.width),
        transform=ros_zone.rio.transform(),
        fill=-1, # Areas outside any polygon
        all_touched=False)

    # Add the mask into to ROS zone Xarray for easy grouping
    #--------------------------------------------------------
    ros_zone_mask_wBsns = ros_zone
    ros_zone_mask_wBsns["bsn_mask"] = (("y", "x"), mask)

    # Now let's get the % of ROS grid cells within each basin for the ROS zone
    #--------------------------------------------------------------------------
    # Thet's ignode cells outside the basins
    valid_pixels = ros_zone_mask_wBsns.where(ros_zone_mask_wBsns.bsn_mask != -1)

    def compute_stats(group):
        total_cells = group.count()
        ones_count = (group == 1).sum()
        percentage = (ones_count / total_cells) * 100

        return percentage

    # Apply the calculation across the masked groups
    results = valid_pixels.groupby("bsn_mask").apply(compute_stats)

    # Turn % into easy to read table
    #--------------------------------
    results_df = results.compute().to_dataframe(name='percentage')
    # Filter out the background mask
    results_df = results_df[results_df.index != -1]
    results_df['GAGE_ID'] = results_df.index.map(lookup)

    return results_df, mask, lookup

def get_ros_events(ros_daily_mask,shpRaster, lookupTable):

    # Add the rasterized basins to the ros daily mask for easy grooping
    #--------------------------------------------------------------------
    mask = shpRaster
    ros_mask_wBsns = ros_daily_mask
    ros_mask_wBsns["bsn_mask"] = (("y", "x"), mask)

    # Let's re-chunk this data for easy processing
    ros_daily_mask = ros_daily_mask.chunk({'time': -1, 'x': 452, 'y': 500}) # HARDCODED, FIX LATER

    # Since the basins are always the same, and the grids too (same resolution.
    # I'll use the first time slice to count total pixels per basin
    # We filter out -1 (background) immediately
    total_cells_static = ros_mask_wBsns["bsn_mask"].where(ros_mask_wBsns["bsn_mask"] != -1).groupby("bsn_mask").count().compute()

    # Compute number of 1s (ROS grid-cell) and % per basin and per day
    #---------------------------------------------------------------------
    ones_count = ros_daily_mask.groupby(ros_mask_wBsns["bsn_mask"]).sum().compute()

    # Let's filter computation of cells outside basins and get the percentage
    ones_count = ones_count.sel(bsn_mask=ones_count.bsn_mask != -1)

    # Combine everything in a single data frame
    #-------------------------------------------
    # Convert ones_count to a 'long' dataframe
    df_final = ones_count.to_dataframe(name="ones_count").reset_index()

    # Convert static denominator to a DF for easy merging
    df_total = total_cells_static.to_dataframe(name="total_cells").reset_index()

    # Merge the static basin info into the daily time series
    df_final = df_final.merge(df_total, on="bsn_mask")

    # Calculate the percentage column
    df_final["percentage"] = (df_final["ones_count"] / df_final["total_cells"]) * 100

    # Map the basin indices to your GAGE_IDs
    df_final["GAGE_ID"] = df_final["bsn_mask"].map(lookupTable)

    # Organize columns and sort for report
    df_final = df_final[["GAGE_ID", "time", "ones_count", "total_cells", "percentage"]]
    df_final = df_final.sort_values(["GAGE_ID", "time"])

    return df_final





