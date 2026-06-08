import os
import pandas as pd
import numpy as np
import xarray
import s3fs
#import rasterio
from rasterio import features
import pyproj
import geopandas
import exactextract
from exactextract import exact_extract
from shapely.geometry import box
from dataretrieval import waterdata
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

    # Create a shallow copy so the original ds_ne is protected
    ds = dataset.copy()

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
    ros_daily_mask = ((ds_daily["QRAIN_daily_mm"] >= 10) &
                      (ds_daily["SNEQV_daily_mm"] >= 10)).astype(int)

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

    # Filter only basins with ROS % > 0
    fltr_ros_zone_bsns = ros_zone_bsns_df[ros_zone_bsns_df['Perc_ROS'] > 0.0].copy()

    return fltr_ros_zone_bsns

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

    # Filter only days with ROS % > 0
    fltr_daily_ros_evs = daily_ros_evs[daily_ros_evs['Perc_ROS'] > 0.0]

    return fltr_daily_ros_evs

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

def extract_dly_hydrologic_properties(ds, events_df, shp):

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

def extract_hydrologic_properties(ds, events_df, shp):

    # 1. Create a "Day" version of the times in the dataset
    ds_times_full = ds.time.values
    ds_days = ds.time.dt.floor("D").values  # Floored to YYYY-MM-DD 00:00

    # 2. Filter the ROS events for ONLY the dates in this dataset
    relevant_events = events_df[events_df['Date'].isin(ds_days)]

    if relevant_events.empty:
        return pd.DataFrame()

    # 3. Filter the Dataset to ONLY these dates
    active_dates_list = relevant_events['Date'].unique()
    # Find every hourly timestamp that falls on each event days
    time_mask = np.isin(ds_days, active_dates_list)
    ds_subset = ds.sel(time=time_mask)

    subset_times = ds_subset.time.values

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
    date_lookup = {i: d for i, d in enumerate(subset_times)}
    df_long['DateTime'] = df_long['band_idx'].map(date_lookup)

    # 8. Pivot back so each variable has its own column (Optional but cleaner)
    # This gives you: [GAGE_ID, Date, temp, precip, etc]
    df_final = df_long.pivot(index=['GAGE_ID', 'DateTime'],
                             columns='variable',
                             values='value').reset_index()

    df_final.columns = [f"{col}_mean" if col not in ['GAGE_ID', 'DateTime'] else col for col in df_final.columns]

    # Since our final df is hourly and events are in daily frequency, we create a temporary column
    # to join both data frames
    df_final['Date_Join'] = pd.to_datetime(df_final['DateTime']).dt.floor('D')

    # 9. Inner Join with your ROS events
    evs_prop = pd.merge(relevant_events, df_final,
                        left_on=['GAGE_ID', 'Date'],
                        right_on=['GAGE_ID', 'Date_Join'],
                        how='inner')

    # Drop the temporary join key
    evs_prop.drop(columns=['Date_Join'], inplace=True)

    return evs_prop

def fetch_usgs_data(site_list, start_date, end_date, parameter_code, batch_size,
                    max_workers, output_path=None):
    """
    Fetches continuous USGS streamflow data for a large number of sites over long
    timeframes. Requests are split into site batches and 1000-day date chunks to
    avoid NWIS server timeouts, and all (batch, chunk) combinations are issued
    concurrently to reduce wall time.

    When output_path is provided, each completed request is saved as an individual
    parquet file inside a '<stem>_chunks/' directory next to the final output
    (an empty parquet is written when a site/date range genuinely has no data).
    This allows the run to be resumed after an interruption or rate-limit failure:
    already-saved chunk files are detected and skipped automatically on the next
    call. A request that exhausts its rate-limit retries writes a '.failed' marker
    instead of a parquet; on the next call the marker is cleared and only that chunk
    is retried, so a multi-run job converges without re-fetching completed chunks.
    The completed chunks are merged into output_path on every call; the chunks
    directory is removed only once a call finishes with zero failures.

    Args:
        site_list: List of USGS site IDs in 'USGS-XXXXXXXX' format.
        start_date: Start of the retrieval period as 'YYYY-MM-DD' string.
        end_date: End of the retrieval period as 'YYYY-MM-DD' string.
        parameter_code: USGS parameter code to retrieve (e.g., '00060' for discharge).
        batch_size: Number of sites per batch request (recommended <= 40).
        max_workers: Number of concurrent HTTP requests (default 10). Increase for
                     faster downloads; lower if the NWIS server returns errors.
        output_path: File path for the final merged parquet. A '<stem>_chunks/'
                     subdirectory is created alongside it for intermediate files.
                     If None, all results are accumulated in memory and returned.

    Returns:
        pd.DataFrame or None: Concatenated USGS records for all sites and date chunks.
        Returns None when output_path is set. Returns an empty DataFrame if no data
        was retrieved.
    """
    # Build 1000-day date windows
    date_range = pd.date_range(start=start_date, end=end_date)
    date_chunks = [date_range[i : i + 1000] for i in range(0, len(date_range), 1000)]

    # Build site batches
    site_batches = [site_list[i : i + batch_size] for i in range(0, len(site_list), batch_size)]

    total_tasks = len(site_batches) * len(date_chunks)

    # Set up chunk directory for intermediate files when output_path is given
    chunk_dir = None
    if output_path is not None:
        output_path = Path(output_path)
        chunk_dir = output_path.parent / (output_path.stem + '_chunks')
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for f in chunk_dir.glob('batch_*.failed'):
            f.unlink()

    def _chunk_path(bi, ci):
        return chunk_dir / f'batch_{bi:03d}_chunk_{ci:03d}.parquet'

    def _fetch_one(batch, chunk):
        chunk_start = chunk[0].strftime('%Y-%m-%d')
        chunk_end   = chunk[-1].strftime('%Y-%m-%d')
        max_retries = 5
        try:
            result = waterdata.get_continuous(
                monitoring_location_id=batch,
                parameter_code=parameter_code,
                time=f'{chunk_start}/{chunk_end}',
            )
            if result and len(result[0]) > 0:
                return result[0]
            return pd.DataFrame()  # genuinely no data for this site/period
        except ChunkInterrupted as exc:
            for attempt in range(max_retries):
                retry_after = exc.retry_after or 300
                stagger = random.uniform(0, min(retry_after, 300))
                wait = retry_after + stagger
                print(f'  [RATE LIMIT] {chunk_start}→{chunk_end}: '
                      f'waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})...')
                time.sleep(wait)
                try:
                    result = exc.call.resume()
                    if result and len(result[0]) > 0:
                        return result[0]
                    return pd.DataFrame()  # genuinely no data after retry
                except ChunkInterrupted as next_exc:
                    exc = next_exc
            print(f'  [ERROR] {chunk_start} to {chunk_end} ({len(batch)} sites): '
                  f'exhausted {max_retries} retries after rate limiting')
            # Return None (not the partial frame) so this chunk is marked .failed and
            # re-fetched in full next run. Saving the partial would mark it complete and
            # silently drop the remaining pages, since cursor state is not persisted.
            return None
        except Exception as e:
            print(f'  [ERROR] {chunk_start} to {chunk_end} ({len(batch)} sites): {e}')
        return None

    # Skip (bi, ci) pairs whose chunk file already exists from a previous run
    pending = []
    skipped = 0
    for bi, batch in enumerate(site_batches):
        for ci, chunk in enumerate(date_chunks):
            if chunk_dir is not None and (
                _chunk_path(bi, ci).exists() or
                _chunk_path(bi, ci).with_suffix('.failed').exists()
            ):
                skipped += 1
            else:
                pending.append((bi, batch, ci, chunk))

    print(f"Total tasks: {total_tasks} | Already done: {skipped} | "
          f"Submitting: {len(pending)} | Workers: {max_workers}")

    all_data = []
    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_one, batch, chunk): (bi, ci, batch, chunk)
            for bi, batch, ci, chunk in pending
        }

        for future in as_completed(futures):
            df = future.result()
            bi, ci, batch, chunk = futures[future]

            with lock:
                completed += 1
                if chunk_dir is not None:
                    if df is None:
                        chunk_start = chunk[0].strftime('%Y-%m-%d')
                        chunk_end   = chunk[-1].strftime('%Y-%m-%d')
                        _chunk_path(bi, ci).with_suffix('.failed').write_text(
                            f'batch={bi} chunk={ci} sites={len(batch)} '
                            f'dates={chunk_start}..{chunk_end}\n'
                        )
                    else:
                        df.to_parquet(_chunk_path(bi, ci), index=False)
                elif df is not None and len(df) > 0:
                    all_data.append(df)
                if completed % 20 == 0 or completed == len(pending):
                    print(f'  {completed}/{len(pending)} requests complete.')

    # Merge completed chunks into output_path. The NWIS service returns several
    # response layouts (native-typed columns with a 'continuous_id'/'time_series_id'
    # vs. all-string columns with an 'id' column, plus an int 'value' variant), so
    # each chunk is normalized to one canonical schema before being written. The
    # merge streams one chunk at a time through a single ParquetWriter so peak
    # memory stays at ~one chunk instead of loading all chunks at once. Only
    # finalize (delete chunks and remove the directory) when there are zero
    # failures this run; otherwise keep the completed chunks so the next run skips
    # them and retries only the failures.
    canon_schema = pa.schema([
        ('monitoring_location_id', pa.string()),
        ('parameter_code',         pa.string()),
        ('statistic_id',           pa.string()),
        ('time',                   pa.timestamp('us', tz='UTC')),
        ('value',                  pa.float64()),
        ('unit_of_measure',        pa.string()),
        ('approval_status',        pa.string()),
        ('qualifier',              pa.string()),
    ])

    def _qualifier_to_string(val):
        # qualifier arrives as null (None) or list<string> depending on the layout;
        # flatten to a ';'-joined string, or None when there are no flags.
        if val is None or (np.isscalar(val) and pd.isna(val)):
            return None
        if isinstance(val, (list, tuple, np.ndarray)):
            items = [str(x) for x in val if x is not None and str(x) != '']
            return ';'.join(items) if items else None
        s = str(val)
        return s if s and s.lower() != 'none' else None

    def _normalize_chunk(f):
        # Read one chunk and coerce it to canon_schema; return None when empty so
        # the genuine no-data sentinels are skipped without contributing rows.
        df = pd.read_parquet(f)
        if df.shape[0] == 0:
            return None
        out = pd.DataFrame()
        out['monitoring_location_id'] = df['monitoring_location_id'].astype('string')
        out['parameter_code']         = df['parameter_code'].astype('string')
        out['statistic_id']           = df['statistic_id'].astype('string')
        # format='ISO8601' so both second- and microsecond-precision time strings
        # parse (string-layout chunks mix '...T00:00:00+00:00' and
        # '...T06:22:03.754229+00:00'); ignored when 'time' is already a timestamp.
        out['time']                   = pd.to_datetime(df['time'], utc=True, format='ISO8601')
        out['value']                  = pd.to_numeric(df['value'], errors='coerce')
        out['unit_of_measure']        = df['unit_of_measure'].astype('string')
        out['approval_status']        = df['approval_status'].astype('string')
        out['qualifier'] = (df['qualifier'].map(_qualifier_to_string).astype('string')
                            if 'qualifier' in df.columns else pd.Series([None] * len(df), dtype='string'))
        return pa.Table.from_pandas(out, schema=canon_schema, preserve_index=False)

    if chunk_dir is not None:
        chunk_files  = sorted(chunk_dir.glob('batch_*.parquet'))
        failed_files = sorted(chunk_dir.glob('batch_*.failed'))

        print(f'\nMerging {len(chunk_files)} chunk files (streaming, normalized schema)...')
        writer = None
        written_rows = 0
        empty_count = 0
        try:
            for f in chunk_files:
                tbl = _normalize_chunk(f)
                if tbl is None:
                    empty_count += 1
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(output_path, canon_schema)
                writer.write_table(tbl)
                written_rows += tbl.num_rows
        finally:
            if writer is not None:
                writer.close()

        if writer is not None:
            suffix = f' ({empty_count} empty skipped)' if empty_count else ''
            print(f'Merged → {output_path} ({written_rows:,} rows{suffix})')
        else:
            print('\n[WARNING] No non-empty chunk files yet.')

        if failed_files:
            # Keep all completed chunks for resume; clear only the .failed markers
            # so those (and only those) chunks are retried on the next call.
            print(f'\n[WARNING] {len(failed_files)} chunk(s) failed this run — '
                  f're-run this cell to retry only those:')
            for f in failed_files:
                print(f'  {f.read_text().strip()}')
            for f in failed_files:
                f.unlink()
            print(f'{len(chunk_files)} completed chunk(s) preserved in {chunk_dir} for resume.')
            return None

        # No failures: the dataset is complete. Finalize and clean up.
        for f in chunk_files:
            f.unlink()
        chunk_dir.rmdir()
        print('Done! All chunks complete.')
        return None

    if not all_data:
        print('\n[WARNING] No data was retrieved.')
        return pd.DataFrame()

    full_df = pd.concat(all_data, axis=0, ignore_index=True)
    print(f'\nDone! Retrieved {len(full_df):,} total rows.')
    return full_df
