import os
import gc
import time
from pathlib import Path
import pandas as pd
import numpy as np
import xarray
import rioxarray  # registers the .rio accessor used for CRS handling
import s3fs
#import rasterio
from rasterio import features
import pyproj
import geopandas
import exactextract
from exactextract import exact_extract
from shapely.geometry import box
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sparse
#import rioxarray

# --- CONSTANTS ---
# NWM projection
nwm_proj = pyproj.Proj(proj='lcc',
                       lat_1=30.,
                       lat_2=60.,
                       lat_0=40.0000076293945, lon_0=-97.,
                       a=6370000, b=6370000)

def read_nwmData(awsPath, variables, timerange, x_range=None, y_range=None):
    """
    Lazy-loads NWM v3 retrospective zarr from S3, assigns the NWM LCC CRS,
    and optionally subsets to a spatial domain defined by coordinate ranges.

    The domain is given as x/y ranges in NWM LCC meters. When both ranges are
    None, the full CONUS domain is returned.

    Args:
        awsPath: S3 path to the NWM zarr store (e.g., 's3://noaa-nwm-retrospective-3-0-pds/...').
        variables: List of variable names to load (e.g., ['QRAIN', 'SNEQV']).
        timerange: xarray-compatible time slice for subsetting the time dimension.
        x_range: Optional (min_x, max_x) tuple in NWM LCC meters; a None on either
                 side means "open to the data edge". None for the whole tuple means
                 the full x extent.
        y_range: Optional (min_y, max_y) tuple in NWM LCC meters; a None on either
                 side means "open to the data edge". None for the whole tuple means
                 the full y extent.

    Returns:
        xarray.Dataset: Lazy dataset with NWM LCC CRS assigned, subset to the
        given ranges (full CONUS when no ranges are given).
    """
    # With no ranges given, both stay (None, None) -> full domain.
    x_range = x_range or (None, None)
    y_range = y_range or (None, None)

    s3_path = awsPath
    # Connect to S3
    s3 = s3fs.S3FileSystem(anon=True)
    store = s3fs.S3Map(root=s3_path, s3=s3, check=False)

    # Lazy load dataset
    ds = xarray.open_zarr(store=store, consolidated=True)

    # NWM data doesn't have a CRS explicitely assigned. It is assigned below
    ds = ds.rio.write_crs(nwm_proj.crs)
    ds = ds.rio.write_coordinate_system()

    # Build the spatial slices from the resolved ranges; a None bound falls back
    # to the data edge. NWM x and y are both ascending, so slices run low -> high.
    x0 = ds["x"].min().item() if x_range[0] is None else x_range[0]
    x1 = ds["x"].max().item() if x_range[1] is None else x_range[1]
    y0 = ds["y"].min().item() if y_range[0] is None else y_range[0]
    y1 = ds["y"].max().item() if y_range[1] is None else y_range[1]

    ds_sub = ds[variables].sel(
        time=timerange,
        x=slice(x0, x1),
        y=slice(y0, y1))

    return ds_sub

def prepare_spatial_assets(ds, shp_path):
    """
    Reads a basin shapefile, reprojects it to NWM LCC, and clips it to the
    spatial extent of the input dataset.

    Args:
        ds: xarray.Dataset with a valid CRS and spatial coordinates (x, y).
        shp_path: Path to the input basin shapefile (e.g., GAGES II .shp file).

    Returns:
        geopandas.GeoDataFrame: Basins reprojected to NWM LCC and filtered to
        those intersecting the raster domain.
    """
    # Read basins shapefile
    shp = geopandas.read_file(shp_path)
    # Reproject basins to NWM projection
    shp_prj = shp.to_crs(nwm_proj.crs)

    # Select basins within the domain
    # Get raster bounds
    raster_bounds = box(*ds.rio.bounds())
    # Filter polygons intersecting raster
    shp_prj_subset = shp_prj[shp_prj.intersects(raster_bounds)]

    print(f"Automatic Setup Complete: {len(shp_prj_subset)} basins selected.")

    return shp_prj_subset

def daily_resampler(dataset):
    """
    Resamples a sub-daily NWM dataset to daily frequency.
    Rate variables (RAIN, PRECIP, PRCP) are summed to daily totals (mm);
    state variables (SWE, temperature, etc.) are averaged to daily means.

    Args:
        dataset: xarray.Dataset with a sub-daily 'time' dimension and NWM LCC CRS.

    Returns:
        xarray.Dataset: Daily dataset with rate variables renamed to *_daily_mm
        and state variables renamed to *_daily, with NWM LCC CRS preserved.
    """
    # Detect the time step (in seconds) automatically
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
    """
    Applies the Musselman ROS detection method to produce a daily binary mask.
    A grid cell is flagged as ROS when daily QRAIN >= 10 mm AND daily SNEQV >= 10 mm.

    Args:
        dataset: xarray.Dataset containing 'QRAIN' (mm/s) and 'SNEQV' (kg/m²) variables
                 with a sub-daily time dimension and NWM LCC CRS.

    Returns:
        xarray.DataArray: Daily binary ROS mask (1 = ROS, 0 = no ROS) with NWM LCC CRS.
    """
    # NOTE: We operate on `dataset` directly below. This only adds a derived
    # 'QRAIN_mm' variable to the dataset; the existing QRAIN/SNEQV data
    # is never modified. If no modification on dataset is ever needed, work on a copy:
    # ds = dataset.copy()

    # Convert units
    # NOTE:
    # QRAIN = Rainfall rate on the ground (mm/s)
    # SNEQV = Snowfall water equivalent (kg/m2)
    # 1 kg/m² = 1 mm water equivalent
    dataset["QRAIN_mm"] = dataset["QRAIN"] * 3 * 3600
    dataset["QRAIN_mm"].attrs["units"] = "mm"

    # Summarize to daily
    rain_daily = dataset["QRAIN_mm"].resample(time="1D").sum()
    sneqv_daily = dataset["SNEQV"].resample(time="1D").mean()

    # Combine them back to a single dataset to ease computations
    ds_daily = xarray.Dataset({
        "QRAIN_daily_mm": rain_daily,
        "SNEQV_daily_mm": sneqv_daily})

    # ROS condition - Binary flag per grid-cell
    ros_daily_mask = ((ds_daily["QRAIN_daily_mm"] >= 10) &
                      (ds_daily["SNEQV_daily_mm"] >= 10)).astype(int)

    # Assign NWM projection
    ros_daily_mask = ros_daily_mask.rio.write_crs(nwm_proj.crs)

    return ros_daily_mask

def define_ros_zone(daily_ros_mask, threshold):
    """
    Identifies the ROS zone as grid cells where at least one ROS day per year
    occurs across a minimum number of years equal to the threshold.

    Args:
        daily_ros_mask: xarray.DataArray with a daily binary ROS flag (from ros_musselman)
                        and a 'time' dimension spanning multiple years.
        threshold: Minimum number of years in which a cell must have at least one
                   ROS day to be included in the zone (e.g., 43 for the full 1979-2022 period).

    Returns:
        xarray.DataArray: Binary ROS zone mask (1 = in zone, 0 = outside) with NWM LCC CRS.
    """
    # ROS zone:
    # All grid cells that have at least 1 ROS day in each of `threshold` years.
    # Example: with threshold=43, a grid cell must have >=1 ROS day in 43 distinct
    # years to be included in the ROS zone (43 of the full NWM v3 retrospective
    # period, 1979-2022).

    yearly_presence = daily_ros_mask.groupby('time.year').any(dim='time').compute() # At least 1 ROS day per year
    ros_zone_mask = (yearly_presence.sum(dim='year') == threshold).astype(int)

    # Assign NWM projection
    ros_zone_mask = ros_zone_mask.rio.write_crs(nwm_proj.crs)

    return ros_zone_mask

def get_ros_basins(ros_zone, shp):
    """
    Computes the percentage of each basin's area that falls within the ROS zone
    and returns only basins with non-zero ROS coverage.

    Args:
        ros_zone: xarray.DataArray binary ROS zone mask (from define_ros_zone)
                  in NWM LCC projection.
        shp: geopandas.GeoDataFrame of basin polygons in NWM LCC projection
             with a 'GAGE_ID' column.

    Returns:
        pd.DataFrame: Columns ['GAGE_ID', 'Perc_ROS'] for basins with Perc_ROS > 0.
    """
    # Extract % of ROS zone per basin
    #ros_zone_bsns_df = exact_extract(ros_zone, shp, ['sum','count','mean'], # For testing
    ros_zone_bsns_df = exact_extract(ros_zone, shp, ['mean'],
                                     include_cols='GAGE_ID', output='pandas')

    ros_zone_bsns_df['Perc_ROS'] = (ros_zone_bsns_df['mean'] * 100).round(0)
    ros_zone_bsns_df.drop(columns=['mean'], inplace=True)

    # Filter only basins with ROS % > 0
    fltr_ros_zone_bsns = ros_zone_bsns_df[ros_zone_bsns_df['Perc_ROS'] > 0.0].copy()

    return fltr_ros_zone_bsns

def get_ros_events(ros_daily_mask, shp):
    """
    Extracts daily ROS coverage per basin over the full time period and returns
    only days where at least one basin has non-zero ROS coverage.

    Args:
        ros_daily_mask: xarray.DataArray daily binary ROS mask (from ros_musselman)
                        in NWM LCC projection, with a 'time' dimension.
        shp: geopandas.GeoDataFrame of basin polygons in NWM LCC projection
             with a 'GAGE_ID' column.

    Returns:
        pd.DataFrame: Columns ['GAGE_ID', 'Date', 'Perc_ROS'] for basin-days with
        Perc_ROS > 0.
    """
    # Let's re-chunk the ros daily dataset for faster processing
    #ros_daily_mask = ros_daily_mask.chunk({'time': t_chunks, 'x': -1, 'y': -1})#.persist()

    # Process % of ros per day and basin
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

def add_water_year(df, date_col='Date'):
    """
    Adds a 'water_year' column based on the US water year convention: the water
    year starts on October 1 and is labeled by the calendar year in which it ends
    (e.g., Oct 1999 through Sep 2000 are all water year 2000).

    Used to group ROS event-days so that inter-event gaps are never measured across
    the Oct 1 water-year boundary. This single definition is shared between the
    exploratory analysis and the later event-delineation step to keep them consistent.

    Args:
        df: pd.DataFrame containing a datetime column.
        date_col: Name of the datetime column to derive the water year from
                  (default 'Date').

    Returns:
        pd.DataFrame: A copy of df with an added integer 'water_year' column.
    """
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    out['water_year'] = dates.dt.year + (dates.dt.month >= 10).astype(int)
    return out

def batch_processor(ds, func, batch_size_years, **kwargs):
    """
    Iterates over a lazy xarray dataset in yearly batches, computing each batch
    before passing it to a processing function, then concatenates all results.
    Required for full 43-year runs to avoid loading the entire dataset into memory.

    Args:
        ds: xarray.Dataset (lazy) with a 'time' dimension spanning multiple years.
        func: Callable to apply to each computed batch (e.g., get_ros_events).
        batch_size_years: Number of years to include in each batch (e.g., 5).
        **kwargs: Additional keyword arguments forwarded to func (e.g., shp=shp_gdf).

    Returns:
        pd.DataFrame: Concatenated results from all yearly batches.
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
    """
    Legacy: extracts daily mean hydrologic properties for basins on ROS event days.
    No chunking — loads all event dates at once. Use extract_hydrologic_prop for
    large datasets.

    Args:
        ds: xarray.Dataset (computed) containing the hydrologic variables to extract,
            with a daily 'time' dimension in NWM LCC projection.
        events_df: pd.DataFrame with columns ['GAGE_ID', 'Date', 'Perc_ROS'].
        shp: geopandas.GeoDataFrame of basin polygons in NWM LCC projection
             with a 'GAGE_ID' column.

    Returns:
        pd.DataFrame: Columns ['GAGE_ID', 'Date', '<var>_mean', ...] joined with
        Perc_ROS. Returns an empty DataFrame if no matching dates are found.
    """
    # Get the dates present in this dataset
    ds_dates = ds.time.values

    # Filter the ROS events for ONLY the dates in this dataset
    relevant_events = events_df[events_df['Date'].isin(ds_dates)]

    if relevant_events.empty:
        return pd.DataFrame()

    # Filter the Dataset to ONLY these dates
    active_dates_list = relevant_events['Date'].unique()
    ds_subset = ds.sel(time=active_dates_list)

    # Run exact_extract (mean values of the hydrologic variables)
    # Only for the selected dates
    df_wide = exact_extract(ds_subset, shp, ['mean'],
                            include_cols='GAGE_ID', output='pandas')

    # Melt the wide dataframe
    # This turns [GAGE_ID, temp_band_1_mean, precip_band_1_mean] into
    # [GAGE_ID, column_name, value]
    df_long = df_wide.melt(id_vars='GAGE_ID', var_name='column_name', value_name='value')

    # Parse the variable name and band number from the column name
    # We split 'temp_band_1_mean' into ['temp', '1']
    # We use regex or string splitting
    parsed = df_long['column_name'].str.extract(r'^(.*)_band_(\d+)_mean$')
    df_long['variable'] = parsed[0]
    df_long['band_idx'] = parsed[1].astype(int) - 1 # Back to 0-indexed for Python

    # Map the Date using the band index
    date_lookup = {i: d for i, d in enumerate(ds_dates)}
    df_long['Date'] = df_long['band_idx'].map(date_lookup)

    # Pivot back so each variable has its own column (Optional but cleaner)
    # This gives you: [GAGE_ID, Date, temp, precip, etc]
    df_final = df_long.pivot(index=['GAGE_ID', 'Date'],
                             columns='variable',
                             values='value').reset_index()

    df_final.columns = [f"{col}_mean" if col not in ['GAGE_ID', 'Date'] else col for col in df_final.columns]

    # Inner Join with your ROS events
    evs_prop = pd.merge(relevant_events, df_final, on=['GAGE_ID', 'Date'], how='inner')

    return evs_prop


def _build_coverage_weights(template_2d, basins, id_col='GAGE_ID'):
    """
    Computes each basin's fractional pixel-coverage ONCE and packs it into a sparse
    weight matrix, so the per-timestep zonal mean becomes a single matrix multiply
    instead of a repeated exactextract call.

    The coverage fractions are obtained from exactextract itself (the 'coverage' op),
    so the resulting weighted average is identical to exactextract's 'mean' op,
    including partial-pixel (fractional-coverage) weighting along basin edges.

    Two details make this robust on real NWM rasters:
      * Coverage is computed on a clean all-ones copy of the grid. exactextract drops
        nodata cells, so running it on a real data slice (which has NaNs over ocean)
        would empty/scramble the per-cell output. Ones carry no nodata.
      * Cells are placed by their absolute center_x/center_y coordinates mapped through
        the grid affine, NOT by exactextract's 'cell_id' (which is local to each
        feature's window, not a global raster index).

    Args:
        template_2d: 2-D xarray.DataArray (a single timestep's spatial frame) with a CRS
                     and 1-D 'x' (ascending) / 'y' (descending) coords on a uniform grid.
                     Defines the grid the weights index into.
        basins: geopandas.GeoDataFrame of basin polygons (same CRS) with an id_col column.
        id_col: Name of the basin identifier column.

    Returns:
        (basin_ids, W): basin_ids is a list of basin identifiers in row order; W is a
        scipy.sparse.csr_matrix of shape (n_basins, n_pixels) holding coverage fractions,
        where n_pixels == ny*nx raveled in C order (matching <data>.reshape(nt, -1)).
    """
    # Clean geometry-only raster (no nodata) so exactextract keeps every cell.
    # Having a grid of ones avoids possible problems with NaN
    geom = xarray.ones_like(template_2d).compute()
    if template_2d.rio.crs is not None:
        geom = geom.rio.write_crs(template_2d.rio.crs)

    # Computes grid coordinates and grid spacing
    xs = np.asarray(template_2d['x'].values, dtype=float)
    ys = np.asarray(template_2d['y'].values, dtype=float)
    nx, ny = xs.size, ys.size
    x0, dx = xs[0], xs[1] - xs[0]          # x ascending  -> dx > 0
    y0, dy = ys[0], ys[0] - ys[1]          # y descending -> dy > 0

    # Get pixel coverage with exact_extract (only once)
    cov = exact_extract(geom, basins, ['cell_id', 'coverage', 'center_x', 'center_y'],
                        include_cols=id_col, output='pandas')
    basin_ids = cov[id_col].tolist()
    rows, cols, data = [], [], [] # Create empty containers, wich will be the Sparse matrix
    for r, (weights, cx, cy) in enumerate(zip(cov['coverage'], cov['center_x'], cov['center_y'])):
        weights = np.asarray(weights, dtype=np.float64)
        if weights.size == 0:
            continue
        # Convert coordinates to grid indices
        ix = np.rint((np.asarray(cx, dtype=float) - x0) / dx).astype(np.int64)
        iy = np.rint((y0 - np.asarray(cy, dtype=float)) / dy).astype(np.int64)
        rows.append(np.full(weights.size, r, dtype=np.int64))
        # Convert 2D indices to 1D indices
        cols.append(iy * nx + ix)          # global C-order flat index over (y, x)
        data.append(weights)

    if rows:
        rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    else:
        rows = cols = data = np.empty(0)
    # Build sparse matrix
    W = sparse.csr_matrix((data, (rows, cols)),
                          shape=(len(basin_ids), nx * ny), dtype=np.float64)
    return basin_ids, W


def extract_hydrologic_prop(ds, variables, events_df, shp, chunk_size_hours=None,
                            output_path=None, target_mem_gb=3.0, read_dtype='float32'):
    """
    Extracts coverage-weighted basin means of one or more NWM variables over ROS event
    timesteps. Basin pixel-coverage weights are computed ONCE (via exactextract) and
    reused as a sparse matrix, so each timestep's basin means are a single matrix
    multiply rather than a repeated exactextract pass. Results are numerically identical
    to calling exactextract's 'mean' op per timestep (partial-pixel weighting preserved),
    but far faster and lighter, which is what makes a CONUS-scale run feasible.

    Args:
        ds: xarray.Dataset (lazy, from read_nwmData) containing all requested variables.
        variables: Variable name or list of variable names to extract (e.g., ['QRAIN', 'SNEQV']).
        events_df: pd.DataFrame with columns ['GAGE_ID', 'Date', 'Perc_ROS'].
        shp: geopandas.GeoDataFrame of basin polygons in NWM LCC projection
             with a 'GAGE_ID' column.
        chunk_size_hours: Optional upper bound on the number of timesteps per read block.
                          The block is additionally capped to stay within target_mem_gb,
                          so a large value here will not cause an out-of-memory read.
                          If None, the block size is chosen automatically from the budget.
        output_path: File path to stream results to parquet via PyArrow (one block at a
                     time, avoids RAM accumulation). If None, results are accumulated in
                     memory and returned as a DataFrame.
        target_mem_gb: Approximate per-block read budget used to size blocks.
        read_dtype: dtype to read NWM data as. 'float32' (default) halves memory and
                    bandwidth vs the stored float64; means then agree with float64
                    exactextract to ~1e-4 absolute on large SWE values (negligible for
                    this analysis). Pass 'float64' for bit-closer agreement at 2x memory.

    Returns:
        pd.DataFrame or None: Wide-format DataFrame with columns
        ['GAGE_ID', 'Date', 'Perc_ROS', 'DateTime', '<var1>_mean', '<var2>_mean', ...].
        Returns None when output_path is set (results are written to disk instead).
    """
    if isinstance(variables, str):
        variables = [variables]

    # Pre-filter basins to only those that appear in the events dataframe
    active_bsns = events_df['GAGE_ID'].unique()
    subset_basins = shp[shp['GAGE_ID'].isin(active_bsns)].copy()

    # Clip the dataset spatially to the basin set's bounding box (CONUS-scaling lever:
    # read only basin-overlapping pixels instead of the whole domain; harmless for NE).
    # One-pixel padding so edge basins keep their partially covered cells.
    minx, miny, maxx, maxy = subset_basins.total_bounds
    xv = ds['x'].values; yv = ds['y'].values
    res_x = abs(xv[1] - xv[0]); res_y = abs(yv[1] - yv[0])
    xsel = np.where((xv >= minx - res_x) & (xv <= maxx + res_x))[0]
    ysel = np.where((yv >= miny - res_y) & (yv <= maxy + res_y))[0]
    ds_win = ds[variables].isel(x=xsel, y=ysel)
    # Force north-up, west-east orientation so exactextract cell ids (row-major from the
    # top-left) line up with <array>.reshape(nt, -1) when we do the matmul.
    ds_win = ds_win.sortby('y', ascending=False).sortby('x', ascending=True)

    # Restrict the time dimension to ROS event timesteps only
    active_dates = pd.DatetimeIndex(events_df['Date'].unique())
    time_mask = ds_win.time.dt.floor("D").isin(active_dates)
    ds_ros = ds_win.sel(time=time_mask)
    n_times = ds_ros.sizes['time']

    if n_times == 0:
        print("[WARNING] No timesteps match the event dates. Returning empty result.")
        return pd.DataFrame()

    # Build the coverage-weight matrix ONCE from a single spatial frame of the window
    # (geometry only; the helper uses a clean ones grid, so no data read is needed here).
    template = ds_win[variables[0]].isel(time=0)
    n_pixels = int(template.size)
    basin_ids, W = _build_coverage_weights(template, subset_basins, id_col='GAGE_ID')
    basin_ids = np.asarray(basin_ids)

    # Size the read block to the memory budget, capped by chunk_size_hours.
    itemsize = np.dtype(read_dtype).itemsize
    bytes_per_step = n_pixels * len(variables) * itemsize
    mem_block = max(1, int(target_mem_gb * 1024**3 // max(bytes_per_step, 1)))
    block = mem_block if chunk_size_hours is None else min(int(chunk_size_hours), mem_block)

    print(f"Processing {len(variables)} variable(s) over {n_times} event timesteps "
          f"({len(basin_ids)} basins, {n_pixels:,} window pixels) "
          f"in blocks of {block} timesteps...")

    all_results = []
    parquet_writer = None
    nb = len(basin_ids)

    for i in range(0, n_times, block):
        end_i = min(i + block, n_times)
        print(f"--- Processing timesteps {i} to {end_i} ---")

        # Read this block at read_dtype (float32 by default halves bytes vs stored float64)
        subset = ds_ros.isel(time=slice(i, end_i)).astype(read_dtype).compute()
        subset_times = subset.time.values
        nt = subset_times.size

        # Coverage-weighted mean per variable via sparse matmul. The denominator is the
        # per-timestep weight sum over VALID (non-NaN) cells, so nodata pixels are excluded
        # exactly as exactextract's 'mean' does.
        df_block = pd.DataFrame({
            'GAGE_ID': np.repeat(basin_ids, nt),
            'DateTime': np.tile(subset_times, nb),
        })
        for v in variables:
            flat = subset[v].values.reshape(nt, -1).T          # (n_pixels, nt)
            valid = (~np.isnan(flat)).astype(np.float32)
            numer = W @ np.nan_to_num(flat, nan=0.0)           # (nb, nt)
            denom = W @ valid                                  # (nb, nt)
            with np.errstate(invalid='ignore', divide='ignore'):
                means = numer / denom
            df_block[f'{v}_mean'] = means.reshape(-1)          # basin-major, time-minor

        # Join with events_df: links each event-day's timesteps back to the daily ROS event
        df_block['Date_Join'] = pd.to_datetime(df_block['DateTime']).dt.floor('D')
        chunk_result = pd.merge(
            events_df,
            df_block,
            left_on=['GAGE_ID', 'Date'],
            right_on=['GAGE_ID', 'Date_Join'],
            how='inner'
        ).drop(columns=['Date_Join'])

        if not chunk_result.empty:
            if output_path is not None:
                table = pa.Table.from_pandas(chunk_result, preserve_index=False)
                if parquet_writer is None:
                    parquet_writer = pq.ParquetWriter(output_path, table.schema)
                parquet_writer.write_table(table)
            else:
                all_results.append(chunk_result)

        del subset, df_block, chunk_result
        gc.collect()

    if parquet_writer is not None:
        parquet_writer.close()
        print(f"\nDone! Results written to {output_path}")
        return None

    if not all_results:
        print("[WARNING] No matching data found after joining with events.")
        return pd.DataFrame()

    return pd.concat(all_results, ignore_index=True)
