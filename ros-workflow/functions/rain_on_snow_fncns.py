import os
import pandas as pd
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

def prepare_spatial_assets(ds, shp_path):
    """
    Reads, crop and reproject input basins to be used along the dataset
    """
    # 1.Read basins shapefile
    shp = geopandas.read_file(shp_path)
    # Reproject basins to NWM projection
    shp_prj = shp.to_crs(nwm_proj.crs)

    # 2. Select basins within the domain
    # Get raster bounds
    raster_bounds = box(*ds.rio.bounds())
    # Filter polygons intersecting raster
    shp_prj_subset = shp_prj[shp_prj.intersects(raster_bounds)]

    print(f"Automatic Setup Complete: {len(shp_prj_subset)} basins selected.")

    return shp_prj_subset

def daily_resampler(dataset):
    """
    Computes daily means for selected variables in an input dataset.
    Returns a daily raster - Xarray
    """
    # 1. Detect the time step (in seconds) automatically
    # Taken from the difference between the first two timestamps
    delta_t = dataset.time.diff('time').dt.seconds.values[0]

    daily_vars = {}

    for var_name in dataset.data_vars:
        # Check for 'Rate' variables (Precip, Rain, Snowfall)
        # We look for keywords that imply a mass flux (mm/s or kg/m2/s)
        is_rate = any(k in var_name.upper() for k in ["RAIN", "PRECIP", "PRCP"])

        if is_rate:
            # Convert Rate (per second) to totals (per time step)
            # e.g., (mm/s) * (3600 s) = mm per hour
            data_depth = dataset[var_name] * delta_t

            # Resample using SUM for total daily totals
            resampled = data_depth.resample(time="1D").sum()

            # Metadata update
            new_name = f"{var_name}_daily_mm"
            resampled.attrs["units"] = "mm"
            resampled.attrs["description"] = f"Daily total calculated from {delta_t/3600}h intervals"
            daily_vars[new_name] = resampled

        else:
            # Resample using MEAN for states (Temp, Soil Moisture, SWE)
            resampled = dataset[var_name].resample(time="1D").mean()

            new_name = f"{var_name}_daily"
            daily_vars[new_name] = resampled

    # Put variables in final dataset
    ds_daily = xarray.Dataset(daily_vars)
    ds_daily = ds_daily.rio.write_crs(nwm_proj.crs)

    return ds_daily

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

def get_ros_basins(ros_zone,shp):

    # Extract % of ROS zone per basin
    #---------------------------------
    #ros_zone_bsns_df = exact_extract(ros_zone, shp_prj_subset, ['sum','count','mean'], # For testing
    ros_zone_bsns_df = exact_extract(ros_zone, shp, ['mean'],
                                     include_cols='GAGE_ID', output='pandas')

    ros_zone_bsns_df['Perc_ROS'] = (ros_zone_bsns_df['mean'] * 100).round(0)
    ros_zone_bsns_df.drop(columns=['mean'], inplace=True)

    return ros_zone_bsns_df

def get_ros_events(ros_daily_mask,shp):#,t_chunks):

    # Let's re-chunk the ros daily dataset for faster processing
    #------------------------------------------------------------
    #ros_daily_mask = ros_daily_mask.chunk({'time': t_chunks, 'x': -1, 'y': -1})#.persist()

    # Process % of ros per day and basin
    #-----------------------------------
    df_ros_evs = exact_extract(ros_daily_mask, shp, ['mean'],include_cols='GAGE_ID',
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

def batch_processor(ds, func, batch_size_years, **kwargs):
    """
    Excecutes a function in bacthes
    ds: A lazy xarray dataset
    func: The function to be excecuted (e.g., get_ros_events or get_precip_means)
    kwargs: Any extra arguments your function needs (like shp_path)
    """
    all_results = []
    years = sorted(ds.time.dt.year.to_series().unique())

    for i in range(0, len(years), batch_size_years):
        batch_years = years[i : i + batch_size_years]
        print(f"--- Processing {batch_years[0]} to {batch_years[-1]} ---")

        # Slice and Compute to get a dataset that exact_extract can digest immediately.
        subset = ds.sel(time=ds.time.dt.year.isin(batch_years)).compute()

        # Run the specific function passed as an argument
        # **kwargs passes things like shp_path automatically
        result_df = func(subset, **kwargs)

        all_results.append(result_df)

    return pd.concat(all_results, ignore_index=True)

def extract_hydrologic_properties(ds, events_df, shp):

    # 1. Get the dates present in this dataset
    ds_dates = ds.time.values

    # 2. Filter the ROS events for ONLY the dates in this dataset
    relevant_events = events_df[events_df['Date'].isin(ds_dates)]

    if relevant_events.empty:
        return pd.DataFrame()

    # 3. Filter the Dataset to ONLY these dates
    active_dates_list = relevant_events['Date'].unique()
    ds_subset = ds.sel(time=active_dates_list)

    # 4. Run exact_extract (mean values of the hydrologic variables)
    # Only for the selected dates
    df_wide = exact_extract(ds_subset, shp, ['mean'],
                            include_cols='GAGE_ID', output='pandas')

    # 5. Melt the wide dataframe
    # This turns [GAGE_ID, temp_band_1_mean, precip_band_1_mean] into
    # [GAGE_ID, column_name, value]
    df_long = df_wide.melt(id_vars='GAGE_ID', var_name='column_name', value_name='value')

    # 6. Parse the variable name and band number from the column name
    # We split 'temp_band_1_mean' into ['temp', '1']
    # We use regex or string splitting
    parsed = df_long['column_name'].str.extract(r'^(.*)_band_(\d+)_mean$')
    df_long['variable'] = parsed[0]
    df_long['band_idx'] = parsed[1].astype(int) - 1 # Back to 0-indexed for Python

    # 7. Map the Date using the band index
    date_lookup = {i: d for i, d in enumerate(ds_dates)}
    df_long['Date'] = df_long['band_idx'].map(date_lookup)

    # 8. Pivot back so each variable has its own column (Optional but cleaner)
    # This gives you: [GAGE_ID, Date, temp, precip, etc]
    df_final = df_long.pivot(index=['GAGE_ID', 'Date'],
                             columns='variable',
                             values='value').reset_index()

    df_final.columns = [f"{col}_mean" if col not in ['GAGE_ID', 'Date'] else col for col in df_final.columns]

    # 9. Inner Join with your ROS events
    evs_prop = pd.merge(relevant_events, df_final, on=['GAGE_ID', 'Date'], how='inner')

    return evs_prop





