import teehr
import teehr.fetching.nwm.retrospective_points as nwm_retro
from pathlib import Path
import pandas as pd
import numpy as np
import datetime as dt
import xarray as xr
import dask
import sys
import argparse
import calendar
from pathlib import Path
import time


if __name__ == "__main__":  # This avoids infinite subprocess creation

   print("I am the main program")

   # import dask
   from dask.distributed import LocalCluster
   #from dask.distributed import Client
   print("Python has started and Dask is imported")
   #sys.exit(0) # clean exit without any errors

   # memory_limit is PER WORKER, in MB. The result "finalize" task runs on a
   # single worker and needs > 1.9 GB on a full year, so 2048MB caused
   # intermittent KilledWorker failures (1982-85, 1990). 4096MB (= 4 GB) gives
   # that task room. Keep 16 workers for parallelism; the node has ~140 GB so
   # 16 x 4096MB = 64GB fits.
   cluster = LocalCluster(
       n_workers=16,               # one worker per core
       threads_per_worker=1,      # no extra threading — pure multiprocessing
       memory_limit="4096MB",
       )

   client = cluster.get_client()

   print(f"Dask client connected: {client}")

   start = time.perf_counter()

   # Define paths
   #---------------------------------------------------------------------
   base_dir = Path("output", "teehr") # Change to the path where you'd like to save the outputs
   data_dir = Path(base_dir, "nwm_retro_archive_uvm")
   metadata_dir = Path("input") # Change to the path where your inout metadata is
   print(base_dir)
   print(data_dir)

   # Define input arguments
   #-----------------------------
   parser = argparse.ArgumentParser(description="Process year and month input")
   parser.add_argument("--year", type=int, required=True, help="Year to process")
   #parser.add_argument("--month", type=int, required=True, help="Month to process")
   parser.add_argument("--version", type=str, required=True, help="NWM version to process")

   args = parser.parse_args()

   year = args.year
   #month = args.month
   nwm_version = args.version

   # Define comid list
   #-------------------
   # location list (USGS gauge IDs)
   #locations_path = Path(metadata_dir, 'Metadata_GAGESII_ROS.parquet')
   locations_path = Path(metadata_dir, 'Metadata_GAGESII_ROS_wComid.parquet')

   df_locs = pd.read_parquet(locations_path, engine='pyarrow')
   #df_locs_fltr = df_locs[df_locs['comid'].notna()]
   df_locs_fltr = df_locs[df_locs['in_nwm']]
   nwm_ids = df_locs_fltr['comid'].unique().tolist()

   # Only for testing
   #nwm_ids = nwm_ids[0:5]

   print(f"Example: {nwm_ids[0:5]} NWM reach IDs")
   print(f"List contains {len(nwm_ids)} NWM reach IDs")

   #-----------------------------------------------------
   # Dates of availability of each NWM version:
   #     - v1.2: 2018-09-17 - 2019-06-18
   #     - v2.0: 2019-06-19 - 2021-04-19
   #     - v2.1/2.2: 2021-04-20 - 2023-09-18
   #     - v3.0: 2023-09-19 - present
   #--------------------------------------------
   # choose dates to fetch - must be within range for a single nwm version.
   # The v3.0 retrospective runs 1979-02-01 01:00 to 2023-02-01 00:00, so the
   # first and last calendar years are partial: clamp to those bounds or TEEHR
   # raises "start_date must be on or after 1979-02-01 01:00:00".
   RETRO_START = dt.datetime(1979, 2, 1, 1)
   #RETRO_END = dt.datetime(2023, 2, 1)
   start_date = max(dt.datetime(year, 1, 1), RETRO_START)
   end_date = dt.datetime(year, 12, 31)
   #end_date = min(dt.datetime(year, 12, 31), RETRO_END)

   variable_name = "streamflow"

   # run teehr fetching function for point data
   nwm_retro.nwm_retro_to_parquet(
       nwm_version = nwm_version,
       variable_name = variable_name,
       start_date = start_date,
       end_date = end_date,
       location_ids = nwm_ids,
       domain = "CONUS",
       output_parquet_dir = Path(data_dir, "nwm30_retrospective"),
       )

   print("Function ended!")

   end = time.perf_counter()
   runtime = round((end - start), 2)
   if runtime < 60:
       print(f'Runtime: {runtime} seconds')
   else:
       print('Runtime: ' + str(round((runtime/60), 2)) + ' minutes')

   #client.close()
   #cluster.close()
   client.shutdown()
