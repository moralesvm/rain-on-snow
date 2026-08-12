"""
Input/Output and aggregation helpers for the ROS workflow.

Helper funtions to rain_on_snow_fncns.py. This module keeps everything
that moves results around: writing the mask and count grids to Zarr, rebuilding a
ROS zone from a persisted mask, mosaicking the per-HUC2 tiles into CONUS-wide rasters and
stores, and merging the per-HUC2 parquets into CONUS-wide tables.
"""

import os
import glob

import pandas as pd
import xarray
import rioxarray  # registers the .rio accessor used for CRS handling
from rioxarray.merge import merge_arrays

try:
    import rain_on_snow_fncns as ros
except ModuleNotFoundError:
    import functions.rain_on_snow_fncns as ros


def save_masks_zarr(masks, path, masks_to_save, time_chunk, spatial_chunk):
    """
    Persists selected daily ROS mask variables to a compressed Zarr store.

    Args:
        masks: xarray.Dataset of daily binary masks (e.g. from ros_musselman).
        path: Output Zarr store path (e.g. 'output/ros_masks/ros_masks_huc01.zarr').
        masks_to_save: Which mask variables to persist. ('mask_ros',) keeps the sparse
                       daily ROS-occurrence grid only; pass
                       ('mask_ros', 'mask_rain', 'mask_sneqv') to persist all three.
        time_chunk: Chunk size (number of days) along time.
        spatial_chunk: Chunk size (cells) along y and x. The spatial subset in
                       read_nwmData starts/ends mid-chunk, so the inherited chunks are
                       non-uniform (e.g. y=(144,350,283)); Zarr only allows the final
                       chunk to be smaller, so y/x are rechunked to a uniform size here.

    Returns:
        str: the path written.
    """
    mode = 'w'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds = masks[list(masks_to_save)].chunk(
        {'time': time_chunk, 'y': spatial_chunk, 'x': spatial_chunk})
    # Drop encoding inherited from the source (zarr v2) store: it carries a numcodecs
    # compressor that the zarr 3 writer rejects ("Expected a BytesBytesCodec"), and stale
    # source chunk sizes. Clearing it lets zarr 3 apply its own default (v3) Zstd codec
    # and use the uniform dask chunks set above.
    ds = ds.drop_encoding()
    ds.to_zarr(path, mode=mode, consolidated=True)
    return path


def save_counts_zarr(counts, path, wy_chunk, spatial_chunk):
    """
    Persists the ROS-day count Dataset from count_ros_days to a compressed Zarr store.

    Args:
        counts: xarray.Dataset from count_ros_days.
        path: Output Zarr store path (e.g. 'output/ros_counts/ros_counts_huc01.zarr').
        wy_chunk: Chunk size (number of water years) along water_year.
        spatial_chunk: Chunk size (cells) along y and x.

    Returns:
        str: the path written.
    """
    mode = 'w'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # The NWM source store carries its own 'crs' scalar coordinate, which read_nwmData
    # passes down. It duplicates 'spatial_ref' (identical crs_wkt) and is never read;
    # dropping it leaves one unambiguous grid-mapping variable in the output.
    ds = counts.drop_vars('crs', errors='ignore')
    # Order matters. write_grid_mapping has to run before drop_encoding: with the calls
    # the other way round the grid_mapping attribute survives into the write, xarray does
    # not re-decode 'spatial_ref' as a coordinate on reload, and .rio.crs comes back None
    # -- a silent loss of georeferencing that only shows up downstream.
    ds = ds.rio.write_transform(ds.rio.transform(recalc=True))
    ds = ds.rio.write_grid_mapping('spatial_ref')
    ds = ds.chunk({'water_year': wy_chunk, 'y': spatial_chunk, 'x': spatial_chunk})

    ds = ds.drop_encoding()
    ds.to_zarr(path, mode=mode, consolidated=True)
    return path

def build_zone_from_saved_masks(huc2, threshold, out_path, output_dir):
    """Rebuild one HUC2's ROS zone raster from its persisted daily mask, at a given threshold.

    Returns the in-memory ROS-zone DataArray so the caller can extract per-basin coverage from it
    (as process_huc2 does) without re-reading the raster.
    """
    masks_path = os.path.join(output_dir, 'ros_masks', f'ros_masks_huc{huc2}.zarr')
    # open_zarr restores the persisted NWM LCC CRS; define_ros_zone re-asserts it anyway.
    masks_local = xarray.open_zarr(masks_path, consolidated=True)
    ros_zone = ros.define_ros_zone(daily_ros_mask=masks_local['mask_ros'], threshold=threshold)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ros_zone.rio.to_raster(out_path)
    return ros_zone


def mosaic_ros_zone(output_dir, method, input_rasters_pattern, out_name):
    """Mosaic the per-HUC2 ROS zone rasters into one CONUS zone raster.

    Reads output/ros_zone/ros_zone_raster_huc*.tif (all on the NWM LCC 1 km grid),
    flips each to north-up (they are written y-ascending, which rasterio.merge
    rejects), merges with `max` so a zone cell (1) wins in the overlapping HUC2
    bounding-box borders, and writes output/conus/<out_name>.

    Args:
        output_dir: Root of the CONUS outputs; the mosaic goes to <output_dir>/conus/.
        method: How merge_arrays resolves overlapping cells ('max' for the binary zone).
        input_rasters_pattern: File pattern selecting the rasters to merge, e.g.
                               '<output_dir>/ros_zone/ros_zone_raster_huc*.tif'. Point it
                               at a different set (e.g. a frequency-bin subtree) to merge
                               those instead. The '*' is required -- this is matched with
                               glob, not read as a directory.
        out_name: Filename for the CONUS output inside <output_dir>/conus/.

    Returns:
        str: the path written, or None if no rasters matched.
    """
    conus_dir = os.path.join(output_dir, 'conus')
    os.makedirs(conus_dir, exist_ok=True)
    rasters = sorted(glob.glob(input_rasters_pattern))
    if not rasters:
        print("no ROS zone rasters found; skipping mosaic.", flush=True)
        return
    das = []
    for t in rasters:
        da = rioxarray.open_rasterio(t)
        if da.y.values[0] < da.y.values[-1]:      # y ascending -> flip north-up for merge
            da = da.sortby('y', ascending=False)
            # sortby reorders the coords but leaves the GeoTransform rioxarray cached when it
            # opened the file, which still describes the y-ascending original: positive dy, and
            # an origin one pixel low.
            da = da.rio.write_transform(da.rio.transform(recalc=True))
        das.append(da)
    mosaic = merge_arrays(das, method=method, nodata=0)
    out = os.path.join(conus_dir, out_name)
    mosaic.rio.to_raster(out)
    ncells = int((mosaic.values == 1).sum())
    print(f"mosaicked {len(rasters)} rasters -> {out} "
          f"({mosaic.shape[-2]}x{mosaic.shape[-1]} grid, {ncells:,} zone cells)", flush=True)
    return out


def mosaic_ros_counts(output_dir, huc2_list, wy_out_name, total_out_name, merge_method, merge_nodata):
    """Mosaic the per-HUC2 ROS-day count stores into one CONUS zarr + one CONUS total raster.

    Reads output/ros_counts/ros_counts_huc<HH>.zarr (from get_ros_day_counts in
    run_zone_thresholds.py) and writes, into output/conus/:
        <total_out_name>  a GeoTIFF of ROS days per cell over the whole record
        <wy_out_name>     a zarr holding both ros_days_wy (water_year, y, x) and
                          ros_days_total (y, x), matching the per-HUC2 store layout
    """
    conus_dir = os.path.join(output_dir, 'conus')
    os.makedirs(conus_dir, exist_ok=True)

    # Open every per-HUC store once (lazily); the water-year loop below reuses them.
    stores = []
    for huc2 in huc2_list:
        counts_path = os.path.join(output_dir, 'ros_counts', f'ros_counts_huc{huc2}.zarr')
        if not os.path.exists(counts_path):
            print(f"[counts mosaic] no count store for HUC2 {huc2}; skipping.", flush=True)
            continue
        stores.append(xarray.open_zarr(counts_path, consolidated=True))
    if not stores:
        print("no ROS count stores found; skipping mosaic.", flush=True)
        return

    def _tiles(das):
        """Flip each tile north-up and give it the (band, y, x) shape merge_arrays expects."""
        out = []
        for da in das:
            flipped = da.y.values[0] < da.y.values[-1]
            if flipped:                            # y ascending -> flip north-up for merge
                da = da.sortby('y', ascending=False)
            da = da.rio.write_crs(ros.nwm_proj.crs)
            if flipped:
                # After the flip the transform has to be rewritten from the reordered coords, or
                # merge_arrays clips a row off each y edge (see mosaic_ros_zone). A no-op for the
                # zarr-backed arrays used here -- write_crs above has just replaced spatial_ref, so
                # there is no cached GeoTransform left to be stale -- but it keeps the two mosaics
                # symmetric, and it has to come after write_crs, which would otherwise wipe it.
                da = da.rio.write_transform(da.rio.transform(recalc=True))
            da = da.expand_dims('band')
            out.append(da)
        return out

    # 1. Total counts - one CONUS GeoTIFF.
    total = merge_arrays(_tiles([s['ros_days_total'] for s in stores]),
                         method=merge_method, nodata=merge_nodata)
    total = total.squeeze('band', drop=True).astype('int16')
    total_out = os.path.join(conus_dir, total_out_name)
    total.rio.to_raster(total_out)
    print(f"mosaicked {len(stores)} count tiles -> {total_out} "
          f"({total.shape[-2]}x{total.shape[-1]} grid, max {int(total.max())} ROS days)", flush=True)

    # 2. Per-water-year counts - one CONUS zarr, appended a water year at a time.
    water_years = sorted(set().union(*[set(s.water_year.values.tolist()) for s in stores]))
    wy_out = os.path.join(conus_dir, wy_out_name)
    for i, wy in enumerate(water_years):
        das = [s['ros_days_wy'].sel(water_year=wy) for s in stores if wy in s.water_year.values]
        slab = merge_arrays(_tiles(das), method=merge_method, nodata=merge_nodata)
        slab = slab.squeeze('band', drop=True).astype('int16')
        # merge_arrays records the fill value in .attrs; xarray's zarr writer refuses to
        # serialize a _FillValue that sits in attrs rather than encoding, so drop it here
        # (the GeoTIFF branch above keeps it, where it is the raster's nodata).
        slab.attrs.pop('_FillValue', None)
        slab.attrs.pop('coordinates', None)
        # Restore water_year as a size-1 dimension so the slabs stack along it.
        slab = slab.expand_dims(water_year=[wy]).to_dataset(name='ros_days_wy')
        slab = slab.chunk({'water_year': 1, 'y': slab.sizes['y'], 'x': slab.sizes['x']})
        slab = slab.drop_encoding()
        if i == 0:
            slab.to_zarr(wy_out, mode='w', consolidated=True)
        else:
            slab.to_zarr(wy_out, append_dim='water_year', consolidated=True)
    print(f"mosaicked {len(water_years)} water years -> {wy_out} "
          f"(water_year {water_years[0]}-{water_years[-1]})", flush=True)

    # 3. Add the total to that same store, so the CONUS zarr carries both variables like
    #    the per-HUC2 ones do. It goes in after the water-year loop, not before: appending
    #    along water_year to a store that already holds a variable without that dimension
    #    is not safe. mode='a' checks the y/x coords against what is already there, and
    #    both arrays come from merge_arrays over the same tiles, so they line up.
    total.attrs.pop('_FillValue', None)
    total.attrs.pop('coordinates', None)
    total_ds = total.to_dataset(name='ros_days_total')
    total_ds = total_ds.chunk({'y': total_ds.sizes['y'], 'x': total_ds.sizes['x']})
    total_ds = total_ds.drop_encoding()
    total_ds.to_zarr(wy_out, mode='a', consolidated=True)
    print(f"added ros_days_total to {wy_out}", flush=True)
    return total_out, wy_out


def _huc2_from_filename(path):
    """Parse the HUC2 token from a per-HUC parquet name (..._huc<HH>.parquet)."""
    stem = os.path.basename(path).rsplit('.', 1)[0]
    return stem.split('_huc', 1)[1]


def merge_huc2_outputs(kind, bsn_class, output_dir, in_dir, out_tag, write_csv):
    """Concatenate the per-HUC2 parquets into CONUS-wide files at the output root.

    HUC02 is stamped here (parsed from each file's name) rather than on the per-HUC files, so the
    per-region outputs keep the lean NE schema and HUC02 only appears where regions actually mix.
    Writes one ``<kind>_gagesii_<cls><out_tag>_CONUS.parquet`` per class plus a combined ref+nonref
    ``<kind>_gagesii_all<out_tag>_CONUS.parquet``.

    Args:
        kind: Which per-HUC outputs to merge ('ros_events' or 'ros_zone').
        bsn_class: Basin classes to merge, e.g. ('ref', 'nonref').
        output_dir: Root of the CONUS outputs; merged files go to <output_dir>/conus/.
        in_dir: Directory holding the per-HUC <cls>/ subfolders. Pass None for the standard
                <output_dir>/<kind>/, or a frequency-bin subtree to merge those instead.
        out_tag: Suffix inserted before '_CONUS' in the output names, to keep variants apart
                 (e.g. '_freqbin5'). Pass '' for the standard names.
        write_csv: Also write a .csv next to each .parquet. The CSVs are large (the combined
                   ros_events file is ~83 MB against ~10 MB of parquet), so only the drivers
                   that actually want them ask for them.
    """
    # Per-HUC files live under <in_dir>/<cls>/; merged CONUS files go to output/conus/.
    if in_dir is None:
        in_dir = os.path.join(output_dir, kind)
    conus_dir = os.path.join(output_dir, 'conus')
    os.makedirs(conus_dir, exist_ok=True)
    per_class = []
    for cls in bsn_class:
        files = sorted(glob.glob(os.path.join(in_dir, cls, f'{kind}_gagesii_{cls}_huc*.parquet')))
        if not files:
            continue
        frames = []
        for f in files:
            fr = pd.read_parquet(f)
            fr['HUC02'] = _huc2_from_filename(f)
            frames.append(fr)
        merged = pd.concat(frames, ignore_index=True)
        out = os.path.join(conus_dir, f'{kind}_gagesii_{cls}{out_tag}_CONUS.parquet')
        merged.to_parquet(out, index=False)
        if write_csv:
            merged.to_csv(out.replace('.parquet', '.csv'), index=False)
        print(f"merged {len(files)} files -> {out} ({len(merged):,} rows, "
              f"{merged['GAGE_ID'].nunique()} basins)", flush=True)
        per_class.append(merged)
    if per_class:
        combined = pd.concat(per_class, ignore_index=True)
        out_all = os.path.join(conus_dir, f'{kind}_gagesii_all{out_tag}_CONUS.parquet')
        combined.to_parquet(out_all, index=False)
        if write_csv:
            combined.to_csv(out_all.replace('.parquet', '.csv'), index=False)
        print(f"combined ref+nonref -> {out_all} ({len(combined):,} rows, "
              f"{combined['GAGE_ID'].nunique()} basins)", flush=True)
