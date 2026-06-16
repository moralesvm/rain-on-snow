"""
USGS observational data retrieval for the ROS workflow.

This module is intentionally decoupled from the NWM/ROS spatial pipeline in
rain_on_snow_fncns_mirror.py: its only link to the workflow is the list of USGS
site IDs passed to fetch_usgs_data. NWIS is a national service, so the fetch
logic is identical for the Northeast and full-CONUS site sets — the CONUS work
lives upstream (generating the site list) and in the NWM domain functions, not
here.

Used by notebook 4 (4_get_Q_usgs.ipynb).
"""

import time
import random
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dataretrieval import waterdata
from dataretrieval.waterdata.chunking import ChunkInterrupted
from concurrent.futures import ThreadPoolExecutor, as_completed


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
