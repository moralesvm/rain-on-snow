"""
Compute ROS event-extraction per-HUC2 CONUS

For each HUC2 region this script crops the NWM domain to the region's GAGES II basin polygons (so no basin is sliced at the domain edge),
computes the daily ROS masks, and extracts the daily ROS event records for the GAGES II
reference and non-reference basins.

This is the only stage that reads NWM from S3, which is why it runs on the VACC. Everything
derivable from the persisted masks -- the ROS zone at every frequency bin, and the ROS-day
counts -- is built afterwards, locally, by ros-workflow/run_zone_thresholds.py. So this script
no longer computes a zone: doing so meant a second full Dask pass over the mask grid to
reproduce something run_zone_thresholds.py already produces as its bin 1.

Outputs are organized by kind, then basin class, under output/:

    output/
      ros_events/ref/    ros_events_gagesii_ref_huc<HH>.parquet     <- daily ROS event records
                 nonref/ ros_events_gagesii_nonref_huc<HH>.parquet
      ros_masks/ ros_masks_huc<HH>.zarr                             <- persisted daily mask grid

Copy both trees into ros-workflow/output/ afterwards, then run run_zone_thresholds.py (zones +
counts) and create_conus_ros_data.py (CONUS event tables).
"""
import os
import sys
import time

# Cuts the "unmanaged memory is high" growth
# This tells the allocator to instantly give unused memory back to the OS, keeping the process's RAM footprint as small as possible.
os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "0")

sys.path.append('path_to/ros-workflow') # Place the path to your ros-workflow
import pandas as pd
import argparse
import functions.rain_on_snow_fncns_mirror as ros
import functions.rain_on_snow_utils as rosutl

#===============================================
# Define configuration and input data paths
#===============================================
S3_PATH = 's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/ldasout.zarr'
VARIABLES = ['QRAIN', 'SNEQV']
TIMERANGE = slice('1979-10-01', '2022-09-30')

SHP_DIR = ('/shapefiles/GAGES_II/'
    'boundaries_shapefiles_by_aggeco/boundaries-shapefiles-by-aggeco/') # CHange to the path to your GAGESII database
METADATA_PATH = '../../input/Metadata_GAGESII_ROS.parquet'

OUTPUT_DIR = '~/scratch/ros_project/output' # Change to the path were you'd like to save the outputs
BSN_CLASS = ('ref', 'nonref')

# ROS masks are saved to avoid recomputing from S3 in the future. They are also the sole input
# to run_zone_thresholds.py (zones at every frequency bin, and the ROS-day counts), so this is
# the artifact that matters most from a VACC run -- copy it back before anything else.
# For now, only mask_ros is saves, but rain and swe mask could be added
# in the future.
SAVE_MASKS = True
MASKS_TO_SAVE = ('mask_ros',)
# Zarr chunking for the mask stores. The read_nwmData spatial subset starts and ends mid-chunk, so
# the chunks inherited from the source are non-uniform; Zarr only allows the final chunk to be
# smaller, so save_masks_zarr rechunks y/x to this uniform size.
MASKS_TIME_CHUNK = 365
MASKS_SPATIAL_CHUNK = 350

# Default temporal batch size for get_ros_events; lower it per region (memory) via BATCH_OVERRIDES.
BATCH_SIZE_YEARS = 2

# Large western regions produce big rasters; drop the batch size if the kernel OOMs.
BATCH_OVERRIDES = {'10U': 1, '10L': 1, '14': 1, '15': 1, '16': 1, '17': 1}

#===============================================
# Dask cluster sizing  (TUNE to your node)
#===============================================
N_WORKERS = 4                   # TUNE
THREADS_PER_WORKER = 4          # N_WORKERS * THREADS_PER_WORKER <= --cpus-per-task
WORKER_MEMORY_LIMIT = "12GB"    # TUNE: per-worker cap; leave the rest of --mem for the client

#=========================
# Define input arguments
#=========================
parser = argparse.ArgumentParser(description="Process ROS % by HUC2 input")
parser.add_argument("--HUC2_LIST", nargs="+", type=str,required=True, help="HUC2 to process")

args = parser.parse_args()

HUC2_LIST = args.HUC2_LIST

#========================================================
# Define main function that cointain all ROS subroutines
#========================================================
def process_huc2(huc2, basins):
    """Run the full event-extraction stage for one HUC2 region and write its outputs."""
    sub = ros.huc2_basins(basins, huc2)
    if len(sub) == 0:
        print(f"[HUC2 {huc2}] no basins with a polygon; skipping.", flush=True)
        return

    x_range, y_range = ros.huc2_domain_bounds(sub)
    print(f"[HUC2 {huc2}] {len(sub)} basins | x={x_range} y={y_range}", flush=True)

    ds = ros.read_nwmData(awsPath=S3_PATH, variables=VARIABLES, timerange=TIMERANGE,
                          x_range=x_range, y_range=y_range)
    masks = ros.ros_musselman(ds)

    # Persist the daily mask grid once. run_zone_thresholds.py reads it back to build the zone
    # at every frequency bin and the ROS-day counts, so nothing zone-related happens here.
    if SAVE_MASKS:
        masks_path = os.path.join(OUTPUT_DIR, 'ros_masks', f'ros_masks_huc{huc2}.zarr')
        rosutl.save_masks_zarr(masks, masks_path, masks_to_save=MASKS_TO_SAVE,
                               time_chunk=MASKS_TIME_CHUNK, spatial_chunk=MASKS_SPATIAL_CHUNK)

    batch_size = BATCH_OVERRIDES.get(huc2, BATCH_SIZE_YEARS)
    for cls in BSN_CLASS:
        shp = sub[sub['CLASS'] == cls]
        if len(shp) == 0:
            continue
        # HUC02 is stamped at merge time (from the filename), so per-region files keep the lean
        # NE schema.
        events_dir = os.path.join(OUTPUT_DIR, 'ros_events', cls)
        os.makedirs(events_dir, exist_ok=True)

        # Daily per-basin ROS event records, batched over years.
        events = ros.batch_processor(ds=masks, func=ros.get_ros_events,
                                     batch_size_years=batch_size, shp=shp)
        events.to_parquet(os.path.join(events_dir, f'ros_events_gagesii_{cls}_huc{huc2}.parquet'), index=False)
        events.to_csv(os.path.join(events_dir, f'ros_events_gagesii_{cls}_huc{huc2}.csv'), index=False)
        print(f"[HUC2 {huc2}] {cls}: {len(shp)} basins -> {len(events):,} event records", flush=True)

#========================================================
# START COMPUTATION
#========================================================

if __name__ == "__main__":  # This avoids infinite subprocess creation

    print("I am the main program...Starting")

    # import dask
    from dask.distributed import LocalCluster
    #from dask.distributed import Client
    print("Python has started and Dask is imported")
    #sys.exit(0) # clean exit without any errors

    # Fewer workers (see the "Dask cluster sizing" block above for why). Spill to
    # node-local disk (TMPDIR) rather than the default cwd, which may be on slow/quota'd NFS.
    cluster = LocalCluster(
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
        memory_limit=WORKER_MEMORY_LIMIT,
        local_directory=os.environ.get("TMPDIR", "/tmp"),
        )

    client = cluster.get_client()

    print("Dask client connected:")
    print(client, flush=True)

    basins = ros.load_conus_basins(SHP_DIR, METADATA_PATH, bsn_class=BSN_CLASS)

    t0 = time.time()
    for huc2 in HUC2_LIST:
        process_huc2(huc2, basins)
    print("Functions ended!")
    print(f"\nAll regions done in {(time.time() - t0) / 60:.1f} min", flush=True)

    client.shutdown()
