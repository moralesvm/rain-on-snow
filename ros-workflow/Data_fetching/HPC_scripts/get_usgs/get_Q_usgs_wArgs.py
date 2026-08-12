import sys
# Path to ROS functions
sys.path.append('/gpfs1/home/m/m/mmorale3/netfiles/ROS_project/code_testing/ros-workflow')

import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
from dataretrieval import waterdata
import functions.rain_on_snow_fncns_mirror as ros
import functions.usgs_data_fncns as usgs
import argparse
import calendar
import time

start = time.perf_counter()

#--------------------
# Define paths
#--------------------
# Base directory where input metadata is located
base_dir = Path('../../input')

# Output directory for retrieved data
output_dir = Path('~/scratch/ros_project/get_usgs/output') # Change to the path were you'd like to save the outputs

# location list (USGS gauge IDs)
#locations_path = Path(base_dir, 'Metadata_GAGESII_NE_ROS.parquet')
locations_path = Path(base_dir, 'Metadata_GAGESII_ROS.parquet') # CONUS metadata

# unique ID column header in the locations list
unique_location_id = 'GAGE_ID'

# Read locations to retrieve streamflow time series
#----------------------------------------------------
# create a list of usgs IDs in the format "usgs-xxxxxxxx"
df_ids = pd.read_parquet(locations_path, engine='pyarrow')

# Slect only non-reference basins
df_ids = df_ids.loc[df_ids['CLASS'] == 'Non-ref']

id_list = df_ids[unique_location_id].radd('USGS-').to_list()
print(f'Total of GAGES II sites: {len(id_list)}')

#-----------------------------
# Define input arguments
#-----------------------------
parser = argparse.ArgumentParser(description="Process year and month input")
parser.add_argument("--year", type=int, required=True, help="Year to process")
#parser.add_argument("--month", type=int, required=True, help="Month to process")

args = parser.parse_args()

year = args.year
#month = args.month

#------------------------------------------
# Retrieve observed streamflow time series
#------------------------------------------
# Setting API Key as environment variable to allow higher rate limits
os.environ["API_USGS_PAT"] = "" # Place your API Key here

# Set the parameters for the USGS data retrieval
parameterCode = '00060'  # Discharge

# Sites per request. USGS support flagged timeouts/502s on this key in Aug 2026 and
# advised querying "one or only a few sites at a time" on the continuous endpoint.
# A request is paginated at 10,000 records/page, so 20 sites x 1 year of 15-minute
# data was ~70 pages, and the deep pages are the queries that time out. At 5 sites
# it is ~18 pages. Total requests barely move (~25,300 -> ~25,600 for a full year)
# because the page count follows data volume, not how sites are grouped.
batchSize  = 5

# Concurrent HTTP requests *per job*. The rate limit is per API key, so this
# multiplies by however many year-jobs are running at once: 3 jobs x 3 workers = 9
# concurrent streams. Keep that product under ~10.
nWorkers   = 3

#start_date = dt.datetime(year, month, 1)
#last_day = calendar.monthrange(start_date.year, start_date.month)[1]
#end_date = dt.datetime(year, month, last_day)
startDate = f'{year}-01-01'
#startDate = f'{year}-10-01' # Only for 1979
endDate   = f'{year}-12-31'

usgs_output_path = output_dir / f'usgs_Q_conus_nonref_{year}.parquet'

print(usgs_output_path)

# Record the key's remaining quota in the .out log before the run starts. The limit
# is shared across every concurrently running year-job, so it cannot be inferred
# from this job alone.
usgs.report_rate_limit(id_list[0], parameter_code=parameterCode)

# Intermediate chunk files are saved to <output stem>_chunks/ as each request
# completes. If the run is interrupted, re-running skips already-saved chunks and
# resumes from where it left off. Chunk names are positional, so a manifest.json
# pins the call shape and the resume aborts if batchSize or the site list changed.
usgs.fetch_usgs_data(
    site_list=id_list,
    start_date=startDate,
    end_date=endDate,
    parameter_code=parameterCode,
    batch_size=batchSize,
    max_workers=nWorkers,
    output_path=usgs_output_path
)

end = time.perf_counter()
runtime = round((end - start), 2)
if runtime < 60:
    print(f'Runtime: {runtime} seconds')
else:
    print('Runtime: ' + str(round((runtime/60), 2)) + ' minutes')
