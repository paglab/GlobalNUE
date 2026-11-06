#!/usr/bin/env python3
"""Global wheat/maize/rice NUE: independent source-mixture Monte Carlo.

Run: python GlobalNUE_M2_MC_Global_Method5_AnnualOnly.py --config config_GlobalNUE_MC_Global_Method5_Annual_Wheat.yaml

For each draw, choose Yield/F/M/D sources independently using M2 provenance
weights (renormalized over sources available in the year). Perturb each selected
product's values independently by its configured relative-uniform bounds;
perturb SN/BNF independently; calculate:
  Nup = max(min_nup, a*Yield + b*(F + M if include_manure else F) + c)
  NUE = Nup / (F + M + D + SN + BNF)
The manure term ALWAYS remains in the denominator. Missing F is nearest-filled,
missing M/D is zero-filled, and Yield <= 0 is invalid, as in the baseline.
This is a source-mixture predictive distribution, NOT a confidence interval for
the M2 weighted-median estimator. Input source choices are shared across pixels
within each simulation/year; perturbation multipliers are pixelwise independent.

New: independent global Method-5 ratio-of-sums for EACH MC draw,
using year-specific processed NUE mask, harvested area, and baseline M2 Nup.
Annual-only output: save baseline Method 5 and MC global mean/median/P2.5/P97.5/SD/CV
for each year in Excel. Keep the 8 original per-pixel MC GeoTIFFs.
Do not export MC-realization detail rows or produce any plots.
"""
import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
import contextlib
import logging
import math
import re
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

LOG = logging.getLogger('M2_Independent_MC_Global_Method5')
FACTORS = ('yield', 'F', 'M', 'D')
STATS = ('median', 'mean', 'sd', 'cv', 'p2_5', 'p97_5', 'p95_diff', 'n')


def load_config(path):
    with open(path, encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError('YAML root must be a dictionary')
    if not cfg.get('scientific_settings_confirmed', False):
        raise ValueError('Confirm M2 source weights before proceeding')
    if not isinstance(cfg.get('nup_equation'), dict):
        raise ValueError('nup_equation mapping required')
    eq = cfg['nup_equation']
    for key in ('yield_coefficient', 'fertilizer_manure_coefficient', 'intercept'):
        try:
            val = float(eq[key])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f'nup_equation.{key} must be a numeric coefficient') from exc
        if not math.isfinite(val):
            raise ValueError(f'nup_equation.{key} must be finite')
        eq[key] = val
    try:
        floor = float(eq.get('min_nup', 3.17))
    except (TypeError, ValueError) as exc:
        raise ValueError('nup_equation.min_nup must be numeric') from exc
    if not math.isfinite(floor) or floor < 0:
        raise ValueError('nup_equation.min_nup must be finite and >= 0')
    eq['min_nup'] = floor
    if not isinstance(eq.get('include_manure', True), bool):
        raise ValueError('nup_equation.include_manure must be true or false')
    eq['include_manure'] = eq.get('include_manure', True)
    cfg['crop'] = str(cfg.get('crop', 'crop')).strip().lower()
    if not cfg['crop']:
        raise ValueError('crop must not be empty')
    if cfg.get('output_decimals', 6) is not None and not (0 <= int(cfg['output_decimals']) <= 8):
        raise ValueError('output_decimals must be 0..8 or null')
    if not math.isfinite(float(cfg.get('out_nodata', -9999))):
        raise ValueError('out_nodata must be finite')
    # SN and BNF are independent uniform-relative random variables.
    for name in ('SN', 'BNF'):
        base = float(cfg[name])
        if not math.isfinite(base) or base < 0:
            raise ValueError(f'{name} baseline must be finite and non-negative')
        settings = cfg.get('constant_uncertainty', {}).get(name)
        if not isinstance(settings, dict):
            raise ValueError(f'constant_uncertainty.{name} must specify relative bounds')
        if 'mc_relative_lower' not in settings or 'mc_relative_upper' not in settings:
            raise ValueError(f'constant_uncertainty.{name} missing relative bounds')
        lo, hi = float(settings['mc_relative_lower']), float(settings['mc_relative_upper'])
        if not (math.isfinite(lo) and math.isfinite(hi) and -1 <= lo <= hi):
            raise ValueError(f'constant_uncertainty.{name}: invalid relative bounds')
    mc = cfg['monte_carlo']
    if mc.get('scheme') != 'source_mixture_independent':
        raise ValueError('Only scheme=source_mixture_independent is implemented')
    if mc.get('distribution') != 'uniform_relative':
        raise ValueError('Only uniform_relative is implemented')
    if int(mc['n_simulations']) < 2 or int(mc['batch_size']) < 1:
        raise ValueError('n_simulations >= 2 and batch_size >= 1 required')
    if int(cfg['block_height']) <= 0 or int(cfg['block_width']) <= 0:
        raise ValueError('Block dimensions must be positive')
    y0, y1 = cfg['years']
    if int(y0) > int(y1):
        raise ValueError('years must be increasing')
    for factor in FACTORS:
        sources = cfg['sources'][factor]
        group_weights = {}
        source_ids = set()
        for s in sources:
            for field in ('id', 'directory', 'group', 'group_weight', 'within_group_weight', 'mc_relative_lower', 'mc_relative_upper'):
                if field not in s:
                    raise ValueError(f'{factor}/{s.get("id")}: missing {field}')
            if s['id'] in source_ids:
                raise ValueError(f'{factor}: duplicate source ID {s["id"]}')
            source_ids.add(s['id'])
            lo, hi = float(s['mc_relative_lower']), float(s['mc_relative_upper'])
            if not (math.isfinite(lo) and math.isfinite(hi) and -1 <= lo <= hi):
                raise ValueError(f'{factor}/{s["id"]}: invalid relative bounds')
            gw = float(s['group_weight'])
            within = float(s['within_group_weight'])
            if not (math.isfinite(gw) and math.isfinite(within) and gw >= 0 and within >= 0):
                raise ValueError(f'{factor}/{s["id"]}: invalid weight')
            if s['group'] in group_weights and not np.isclose(group_weights[s['group']], gw):
                raise ValueError(f'{factor}: inconsistent group weight for {s["group"]}')
            group_weights[s['group']] = gw
            s['probability'] = gw * float(s['within_group_weight'])
        if not np.isclose(sum(group_weights.values()), 1, atol=1e-6):
            raise ValueError(f'{factor}: group weights must sum to 1')
        for g in group_weights:
            if not np.isclose(sum(float(s['within_group_weight']) for s in sources if s['group'] == g), 1, atol=1e-6):
                raise ValueError(f'{factor}/{g}: within-group weights must sum to 1')
    # Global Method 5 inputs are required; missing folders/files fail early.
    ga = cfg.get('global_aggregation')
    if not isinstance(ga, dict) or ga.get('method') != 'nup_ratio_of_sums':
        raise ValueError('global_aggregation.method must be nup_ratio_of_sums')
    for section in ('reference_nue_mask', 'harvested_area', 'baseline_nup'):
        item = ga.get(section)
        if not isinstance(item, dict) or not item.get('directory') or not item.get('filename_template'):
            raise ValueError(f'global_aggregation.{section}: directory and filename_template required')
        if '{year}' not in str(item['filename_template']):
            raise ValueError(f'global_aggregation.{section}.filename_template must contain {{year}}')
    if not isinstance(ga.get('restrict_to_baseline_valid', True), bool):
        raise ValueError('global_aggregation.restrict_to_baseline_valid must be boolean')
    if float(ga.get('area_threshold', 0)) < 0:
        raise ValueError('global_aggregation.area_threshold must be >= 0')
    # Annual-only Excel is always generated, with no per-realization detail export.
    pcfg = cfg.get('parallel', {})
    if int(pcfg.get('max_workers', 16)) < 1 or int(pcfg.get('max_pending_tasks', 32)) < 1:
        raise ValueError('parallel max_workers and max_pending_tasks must be positive')
    if pcfg.get('scratch_dir') and not Path(pcfg['scratch_dir']).is_dir():
        raise ValueError('parallel.scratch_dir must be an existing folder')
    return cfg


def index_files(cfg):
    indexed = {}
    for factor in FACTORS:
        indexed[factor] = []
        for s in cfg['sources'][factor]:
            root = Path(s['directory'])
            if not root.is_dir():
                raise FileNotFoundError(f'Missing directory: {root}')
            years = {}
            for p in sorted(root.iterdir()):
                if p.suffix.lower() not in ('.tif', '.tiff'):
                    continue
                m = re.search(r'(\d{4})$', p.stem)
                if m:
                    year = int(m.group(1))
                    if year in years:
                        raise ValueError(f'Duplicate year {year} in {root}')
                    years[year] = p
            indexed[factor].append((s, years))
    return indexed


def read_on_grid(path, ref):
    with rasterio.open(path) as src:
        if src.crs is None or ref.crs is None:
            raise ValueError(f'CRS missing: {path}')
        with WarpedVRT(src, crs=ref.crs, transform=ref.transform,
                       width=ref.width, height=ref.height,
                       resampling=Resampling.nearest, nodata=np.nan,
                       dtype='float32') as vrt:
            result = np.ma.filled(vrt.read(1, masked=True), np.nan).astype(np.float32)
    result[~np.isfinite(result)] = np.nan
    return result


def fill_f_nearest(f):
    ok = np.isfinite(f)
    if ok.all():
        return f
    if not ok.any():
        return np.zeros_like(f)
    idx = distance_transform_edt(~ok, return_distances=False, return_indices=True)
    filled = f.copy()
    filled[~ok] = f[tuple(a[~ok] for a in idx)]
    return filled


def windows(height, width, bh, bw):
    for row in range(0, height, bh):
        for col in range(0, width, bw):
            yield Window(col, row, min(bw, width-col), min(bh, height-row))


def source_mixture_tile(blocks, source_choices, bounds, eq, sn, bnf,
                        constant_uncertainty, nsim, batch_size, rng,
                        reference_spatial_mask, global_domain, area):
    """Unchanged source-mixture draws + paired per-draw global Method 5 totals.

    Each per-draw numerator and denominator uses the SAME valid pixels.
    Annual global totals are constructed by summing per-tile totals, not by
    aggregating per-pixel MC quantile rasters.
    """
    nrows, ncols = reference_spatial_mask.shape
    pixels = nrows * ncols
    samples = np.empty((nsim, pixels), dtype=np.float32)
    num = np.zeros(nsim, dtype=np.float64)
    den = np.zeros(nsim, dtype=np.float64)
    counts = np.zeros(nsim, dtype=np.int64)
    pixel_ids = np.arange(pixels)
    flat = {k: a.reshape(a.shape[0], pixels) for k, a in blocks.items()}
    spatial_mask = reference_spatial_mask.reshape(-1)
    domain = global_domain.reshape(-1)
    ha = area.reshape(-1).astype(np.float64, copy=False)
    coef_y = float(eq['yield_coefficient'])
    coef_n = float(eq['fertilizer_manure_coefficient'])
    intercept = float(eq['intercept'])
    min_nup = float(eq['min_nup'])
    include_manure = bool(eq['include_manure'])

    # Preserve the random number generation and call sequence of the former MC.
    for start in range(0, nsim, batch_size):
        b = min(batch_size, nsim - start)
        realized = {}
        for factor in FACTORS:
            ids = source_choices[factor][start:start+b]
            values = flat[factor][ids[:, None], pixel_ids[None, :]].copy()
            lo, hi = bounds[factor]
            multipliers = rng.random((b, pixels), dtype=np.float32)
            multipliers *= (hi[ids] - lo[ids])[:, None]
            multipliers += (1.0 + lo[ids])[:, None]
            values *= multipliers
            realized[factor] = values
        y, f, m, d = (realized[v] for v in FACTORS)
        nup_input = f + m if include_manure else f
        numerator = coef_y * y + coef_n * nup_input + intercept
        numerator = np.maximum(numerator, min_nup)
        sn_bounds = constant_uncertainty['SN']
        bnf_bounds = constant_uncertainty['BNF']
        sn_sample = sn * (1.0 + rng.uniform(
            float(sn_bounds['mc_relative_lower']), float(sn_bounds['mc_relative_upper']),
            size=(b, pixels)).astype(np.float32))
        bnf_sample = bnf * (1.0 + rng.uniform(
            float(bnf_bounds['mc_relative_lower']), float(bnf_bounds['mc_relative_upper']),
            size=(b, pixels)).astype(np.float32))
        denominator = f + m + d + sn_sample + bnf_sample
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            nue = numerator / denominator
        # Consistent with baseline: valid positive NUE, finite Nup, positive HA.
        valid = (spatial_mask[None, :] & np.isfinite(y) & (y > 0)
                 & np.isfinite(numerator) & (numerator >= 0)
                 & np.isfinite(denominator) & (denominator > 0)
                 & np.isfinite(nue) & (nue > 0))
        samples[start:start+b] = np.where(valid, nue, np.nan).astype(np.float32)
        # Exact methodological form: sum(Nup*HA) / sum((Nup/NUE)*HA).
        # Use float64 in BOTH global sums; don't round to 6 decimals first.
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            nup_term = numerator.astype(np.float64) * ha[None, :]
            input_term = (numerator.astype(np.float64) / nue.astype(np.float64)) * ha[None, :]
        paired = (valid & domain[None, :] & np.isfinite(nup_term) & np.isfinite(input_term))
        num[start:start+b] = np.sum(np.where(paired, nup_term, 0.0), axis=1, dtype=np.float64)
        den[start:start+b] = np.sum(np.where(paired, input_term, 0.0), axis=1, dtype=np.float64)
        counts[start:start+b] = paired.sum(axis=1, dtype=np.int64)

    n_per_cell = np.isfinite(samples).sum(axis=0)
    has = n_per_cell > 0
    samples[:, ~has] = 0
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        p2, med, p97 = np.nanpercentile(samples, [2.5, 50.0, 97.5], axis=0, method='linear')
        avg = np.nanmean(samples, axis=0)
        std = np.nanstd(samples, axis=0, ddof=0)
    cv = np.full(pixels, np.nan, dtype=np.float64)
    np.divide(std, avg, out=cv, where=has & (avg != 0))
    results = dict(median=med, mean=avg, sd=std, cv=cv, p2_5=p2,
                   p97_5=p97, p95_diff=p97-p2, n=n_per_cell.astype('float32'))
    for key in results:
        values = np.asarray(results[key], dtype='float32')
        values[~has] = np.nan
        results[key] = values.reshape(nrows, ncols)
    return results, num, den, counts


# Global read-only mapped inputs per worker; initialized after spawn.
_WORKER = {}


def _worker_init(array_paths, source_choices, bounds, eq, sn, bnf,
                 uncertainty, nsim, batch_size, random_seed, year):
    global _WORKER
    # Full inputs, domain mask and area are in read-only on-disk npy memory maps.
    arrays = {factor: np.load(str(path), mmap_mode='r')
              for factor, path in array_paths.items()}
    _WORKER = dict(arrays=arrays, source_choices=source_choices, bounds=bounds,
                   eq=eq, sn=sn, bnf=bnf, uncertainty=uncertainty,
                   nsim=nsim, batch_size=batch_size, seed=random_seed, year=year)


def _worker_tile(task):
    ti, row, col, height, width = task
    state = _WORKER
    arrays = state['arrays']
    region = np.asarray(arrays['reference_spatial_mask'][row:row+height, col:col+width], dtype=bool)
    global_domain = np.asarray(arrays['global_domain'][row:row+height, col:col+width], dtype=bool)
    area = np.asarray(arrays['harvested_area'][row:row+height, col:col+width])
    if not np.any(region):
        empty = {k: np.full((height, width), np.nan, dtype=np.float32) for k in STATS}
        zeros = np.zeros(state['nsim'], dtype=np.float64)
        return ti, row, col, height, width, empty, zeros, zeros.copy(), np.zeros(state['nsim'], dtype=np.int64)
    blocks = {v: np.asarray(arrays[v][:, row:row+height, col:col+width]) for v in FACTORS}
    seed = np.random.SeedSequence([state['seed'], state['year'], row, col])
    rng = np.random.default_rng(seed)
    stats, num, den, counts = source_mixture_tile(
        blocks, state['source_choices'], state['bounds'], state['eq'],
        state['sn'], state['bnf'], state['uncertainty'], state['nsim'],
        state['batch_size'], rng, region, global_domain, area)
    return ti, row, col, height, width, stats, num, den, counts


def _make_memmap_files(full, location):
    location.mkdir(parents=True, exist_ok=True)
    paths = {}
    for factor, arrays in full.items():
        path = location / f'{factor}.npy'
        if isinstance(arrays, (list, tuple)):
            dims = (len(arrays), *arrays[0].shape)
            mapped = np.lib.format.open_memmap(path, mode='w+', dtype='float32', shape=dims)
            for i, array in enumerate(arrays):
                mapped[i, :, :] = array
        else:
            dims = arrays.shape
            dtype = 'uint8' if factor in ('reference_spatial_mask', 'global_domain') else 'float32'
            mapped = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=dims)
            mapped[:] = arrays
        mapped.flush()
        del mapped
        paths[factor] = str(path)
    return paths


def _run_tasks_parallel(executor, tasks, max_pending, consume):
    iterator = iter(tasks)
    pending = set()
    def enqueue():
        try:
            task = next(iterator)
        except StopIteration:
            return False
        pending.add(executor.submit(_worker_tile, task))
        return True
    for _ in range(max_pending):
        if not enqueue():
            break
    while pending:
        finished, pending = wait(pending, return_when=FIRST_COMPLETED)
        for fut in finished:
            consume(fut.result())
            enqueue()


def input_path(section, year):
    return Path(section['directory']) / section['filename_template'].format(year=year)


def read_strict_grid(path, ref, year, title, extra_nodata=(-9999.,)):
    if not path.is_file():
        raise FileNotFoundError(f'{year}: missing {title}: {path}')
    with rasterio.open(path) as ds:
        if (ds.crs != ref.crs or ds.width != ref.width or ds.height != ref.height
                or not ds.transform.almost_equals(ref.transform, precision=1e-10)):
            raise ValueError(f'{year}: {title} grid/CRS differs from source reference: {path}')
        a = ds.read(1, masked=True)
        data = np.asarray(a.data, dtype=np.float32)
        valid = (~np.ma.getmaskarray(a)) & np.isfinite(data)
        for nd in extra_nodata:
            valid &= (data != float(nd))
    return data, valid


def baseline_method5(nue, nup, ha, valid):
    mask = (valid & np.isfinite(nue) & (nue > 0) & np.isfinite(nup)
            & (nup >= 0) & np.isfinite(ha) & (ha > 0))
    x = nue[mask].astype(np.float64)
    u = nup[mask].astype(np.float64)
    area = ha[mask].astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        n = u * area
        d = (u / x) * area
    ok = np.isfinite(n) & np.isfinite(d)
    n_sum = float(np.sum(n[ok], dtype=np.float64))
    d_sum = float(np.sum(d[ok], dtype=np.float64))
    return (n_sum / d_sum if d_sum > 0 else np.nan,
            int(np.count_nonzero(ok)), n_sum, d_sum, mask)


def save_excel(path, year_rows, plot_rows, source_rows, cfg):
    """Write the compact annual-only workbook with publicly pip-installable openpyxl.

    No per-realization (26 x 5000) records are exported. The four existing sheet
    names and column definitions stay unchanged.
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError(
            'Excel export requires openpyxl. Install it with: python -m pip install openpyxl'
        ) from exc

    wb = Workbook()
    wb.remove(wb.active)
    ga = cfg['global_aggregation']
    meta_rows = [
        {'Item': 'Crop', 'Value': cfg['crop']},
        {'Item': 'Years', 'Value': f"{cfg['years'][0]}–{cfg['years'][1]}"},
        {'Item': 'MC simulations per year', 'Value': int(cfg['monte_carlo']['n_simulations'])},
        {'Item': 'Global NUE aggregation', 'Value': 'Method 5 = SUM(Nup*HA) / SUM((Nup/NUE)*HA)'},
        {'Item': 'Baseline', 'Value': 'Processed M2 NUE, paired with baseline M2 Nup and harvested area'},
        {'Item': 'MC mean/median/SD/CV/percentiles', 'Value': 'Calculated from per-realization global Method 5 NUE (NOT pixelwise percentile rasters)'},
        {'Item': 'Black line for scheme B', 'Value': 'M2_baseline_percent'},
        {'Item': 'Shaded band for scheme B', 'Value': 'MC_p2_5_percent to MC_p97_5_percent'},
        {'Item': 'Interval interpretation', 'Value': '95% MC simulation interval, NOT a formal 95% confidence interval'},
        {'Item': 'Annual spatial domain', 'Value': 'Processed annual NUE mask, paired with baseline Nup/HA validity'},
        {'Item': 'Processed NUE folder', 'Value': str(ga['reference_nue_mask']['directory'])},
        {'Item': 'Harvested area folder', 'Value': str(ga['harvested_area']['directory'])},
        {'Item': 'Baseline Nup folder', 'Value': str(ga['baseline_nup']['directory'])},
        {'Item': 'NUE values', 'Value': 'fraction = 0–1; percent columns = 100 × fraction'},
        {'Item': 'MC CV', 'Value': 'MC global SD / MC global mean (unitless; not percent)'},
        {'Item': 'GeoTIFF outputs', 'Value': '8 annual NUE MC rasters: median, mean, sd, cv, p2_5, p97_5, p95_diff, n'},
        {'Item': 'Source-mixture model', 'Value': 'Independent pixelwise perturbations; product source selection shared per global MC realization'},
        {'Item': 'Detail simulation rows', 'Value': 'Not exported (only annual summary statistics are retained)'},
    ]
    datasets = [('Annual_Summary', year_rows), ('Plot_Scheme_B', plot_rows),
                ('Source_Weights', source_rows), ('Readme', meta_rows)]
    header_fill = PatternFill(fill_type='solid', fgColor='18364B')
    header_font = Font(bold=True, color='FFFFFF')
    for name, records in datasets:
        ws = wb.create_sheet(name)
        ws.freeze_panes = 'A2'
        if not records:
            continue
        fields = list(records[0].keys())
        ws.append(fields)
        for rec in records:
            ws.append([_excel_cell(rec.get(field)) for field in fields])
        ws.row_dimensions[1].height = 29
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(wrap_text=True, vertical='center')
        for col_idx in range(1, len(fields) + 1):
            width = (22 if col_idx == 1 else 62) if name == 'Readme' else (17 if col_idx == 1 else 21)
            ws.column_dimensions[get_column_letter(col_idx)].width = width
        if name != 'Readme':
            ws.auto_filter.ref = ws.dimensions
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))


def _excel_cell(value):
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (int, float, str, bool, np.integer, np.floating)):
        return value.item() if hasattr(value, 'item') else value
    return str(value)


def run(cfg):
    indexes = index_files(cfg)
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    mc = cfg['monte_carlo']
    nsim = int(mc['n_simulations'])
    batch = int(mc['batch_size'])
    bh, bw = int(cfg['block_height']), int(cfg['block_width'])
    nodata = float(cfg.get('out_nodata', -9999))
    eq = cfg['nup_equation']
    pcfg = cfg.get('parallel', {})
    ga = cfg['global_aggregation']
    parallel_enabled = bool(pcfg.get('enabled', True))
    workers = int(pcfg.get('max_workers', 16))
    pending_limit = int(pcfg.get('max_pending_tasks', workers * 2))
    if not parallel_enabled:
        workers = 1
    records = []
    summary_rows, plot_rows = [], []
    for year in range(int(cfg['years'][0]), int(cfg['years'][1]) + 1):
        available = {v: [(s, by_year[year]) for s, by_year in indexes[v] if year in by_year] for v in FACTORS}
        if any(not x for x in available.values()):
            missing_factors = [v for v in FACTORS if not available[v]]
            LOG.warning('%s %d skipped: missing factor(s) %s', cfg['crop'], year, missing_factors)
            continue
        with rasterio.open(available['yield'][0][1]) as ref:
            if ref.crs is None:
                raise ValueError(f'{year}: Yield CRS missing')
            if cfg.get('require_equal_yield_grids', True):
                for _source, ypath in available['yield'][1:]:
                    with rasterio.open(ypath) as chk:
                        if not (chk.crs == ref.crs and chk.width == ref.width and chk.height == ref.height
                                and chk.transform == ref.transform):
                            raise ValueError(f'{year}: Yield grids differ: {ypath}')
            extras = ga.get('additional_nodata', [-9999.0])
            mask_path = input_path(ga['reference_nue_mask'], year)
            ha_path = input_path(ga['harvested_area'], year)
            nup_path = input_path(ga['baseline_nup'], year)
            mask_val, mask_ok = read_strict_grid(mask_path, ref, year, 'processed NUE mask', extras)
            area_val, area_ok = read_strict_grid(ha_path, ref, year, 'harvested area', extras)
            nup_val, nup_ok = read_strict_grid(nup_path, ref, year, 'baseline Nup', extras)
            area_ok &= area_val > float(ga.get('area_threshold', 0.0))
            mask_n = mask_ok & (mask_val > float(ga.get('nue_min_value', 0.0)))
            base_valid = mask_n & nup_ok & (nup_val >= float(ga.get('nup_min_value', 0.0))) & area_ok
            baseline_ratio, baseline_count, baseline_num, baseline_den, baseline_mask = baseline_method5(
                mask_val, nup_val, area_val, base_valid)
            if not math.isfinite(baseline_ratio):
                raise ValueError(f'{year}: baseline Method5 undefined; inspect processed NUE/Nup/HA')
            domain = (baseline_mask if ga.get('restrict_to_baseline_valid', True)
                      else (mask_ok & area_ok))
            ref_pixels = int(mask_ok.sum())
            domain_pixels = int(domain.sum())
            if not domain_pixels:
                raise ValueError(f'{year}: annual reference domain is empty')
            LOG.info('%s %d reference=%d, global domain=%d, M2 baseline=%.4f%%',
                     cfg['crop'], year, ref_pixels, domain_pixels, 100*baseline_ratio)
            profile = ref.profile.copy()
            profile.update(driver='GTiff', count=1, dtype='float32', nodata=nodata,
                           compress='LZW', predictor=3, BIGTIFF='IF_SAFER')
            for key in ('photometric', 'alpha', 'colorinterp', 'blockxsize', 'blockysize', 'tiled'):
                profile.pop(key, None)
            probs, bounds, full = {}, {}, {}
            for factor in FACTORS:
                entries = available[factor]
                w = np.array([s['probability'] for s, _ in entries], dtype=np.float64)
                if not np.isfinite(w).all() or w.sum() <= 0:
                    raise ValueError(f'{year}: invalid source weights for {factor}')
                probs[factor] = w / w.sum()
                bounds[factor] = (
                    np.array([s['mc_relative_lower'] for s, _ in entries], dtype=np.float32),
                    np.array([s['mc_relative_upper'] for s, _ in entries], dtype=np.float32))
                full[factor] = []
                for i, (source, path) in enumerate(entries):
                    raster = read_on_grid(path, ref)
                    if factor == 'F':
                        raster = fill_f_nearest(raster)
                    elif factor in ('M', 'D'):
                        raster = np.nan_to_num(raster, nan=0.0)
                    full[factor].append(raster)
                    records.append(dict(year=year, factor=factor, source=source['id'],
                                        source_weight=float(probs[factor][i]),
                                        mc_relative_lower=source['mc_relative_lower'],
                                        mc_relative_upper=source['mc_relative_upper']))
            full['reference_spatial_mask'] = mask_ok.astype('uint8')
            full['global_domain'] = domain.astype('uint8')
            full['harvested_area'] = area_val
            selection_rng = np.random.default_rng(np.random.SeedSequence(
                [int(mc['random_seed']), year, 987654]))
            choices = {v: selection_rng.choice(len(probs[v]), size=nsim, p=probs[v]) for v in FACTORS}
            tasks = [(ti, int(win.row_off), int(win.col_off), int(win.height), int(win.width))
                     for ti, win in enumerate(windows(ref.height, ref.width, bh, bw))]
            global_num = np.zeros(nsim, dtype=np.float64)
            global_den = np.zeros(nsim, dtype=np.float64)
            global_valid_count = np.zeros(nsim, dtype=np.int64)
            scratch_setting = pcfg.get('scratch_dir')
            with tempfile.TemporaryDirectory(prefix=f'nue_mc_{year}_', dir=scratch_setting or None) as scratch:
                array_paths = _make_memmap_files(full, Path(scratch))
                del full
                args = (array_paths, choices, bounds, eq, float(cfg['SN']), float(cfg['BNF']),
                        cfg['constant_uncertainty'], nsim, batch, int(mc['random_seed']), year)
                progress = [0]
                with contextlib.ExitStack() as outputs:
                    targets = {name: outputs.enter_context(rasterio.open(
                        output_dir / f'NUE_M2_MC_Independent_{name}_{year}.tif', 'w', **profile)) for name in STATS}
                    def consume(result):
                        _, row, col, height, width, stats, n, d, ct = result
                        win = Window(col, row, width, height)
                        global_num[:] += n
                        global_den[:] += d
                        global_valid_count[:] += ct
                        for name, dst in targets.items():
                            a = np.where(np.isfinite(stats[name]), stats[name], nodata).astype(np.float32)
                            if name != 'n' and cfg.get('output_decimals') is not None:
                                a = np.round(a, int(cfg['output_decimals']))
                            dst.write(a, 1, window=win)
                        progress[0] += 1
                        if progress[0] % max(1, len(tasks)//10) == 0 or progress[0] == len(tasks):
                            LOG.info('%s %d: %d/%d tiles', cfg['crop'], year, progress[0], len(tasks))
                    if parallel_enabled and workers > 1:
                        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                                                 initializer=_worker_init, initargs=args) as pool:
                            _run_tasks_parallel(pool, tasks, pending_limit, consume)
                    else:
                        _worker_init(*args)
                        for task in tasks:
                            consume(_worker_tile(task))
            ratio = np.full(nsim, np.nan, dtype=np.float64)
            ok = (global_den > 0) & np.isfinite(global_den) & np.isfinite(global_num)
            np.divide(global_num, global_den, out=ratio, where=ok)
            if not np.any(ok):
                raise ValueError(f'{year}: no valid MC global realization')
            rgood = ratio[np.isfinite(ratio)]
            mc_mean = float(np.mean(rgood)); mc_median = float(np.median(rgood))
            low, high = [float(v) for v in np.percentile(rgood, [2.5, 97.5])]
            mc_std = float(np.std(rgood, ddof=0))
            row = dict(crop=cfg['crop'], year=year, baseline_nue_fraction=baseline_ratio,
                       baseline_nue_percent=baseline_ratio*100,
                       mc_mean_fraction=mc_mean, mc_mean_percent=mc_mean*100,
                       mc_median_fraction=mc_median, mc_median_percent=mc_median*100,
                       mc_p2_5_fraction=low, mc_p2_5_percent=low*100,
                       mc_p97_5_fraction=high, mc_p97_5_percent=high*100,
                       mc_sd_fraction=mc_std, mc_cv=(mc_std/mc_mean if mc_mean else np.nan),
                       n_valid_realizations=len(rgood), n_simulations=nsim,
                       reference_mask_pixels=ref_pixels, baseline_valid_pixels=baseline_count,
                       domain_pixels=domain_pixels, baseline_sum_nup_ha=baseline_num,
                       baseline_sum_ninput_ha=baseline_den)
            summary_rows.append(row)
            plot_rows.append(dict(year=year, M2_baseline_percent=baseline_ratio*100,
                                  MC_mean_percent=mc_mean*100, MC_median_percent=mc_median*100,
                                  MC_p2_5_percent=low*100, MC_p97_5_percent=high*100))
            # Only annual MC summary rows are retained, no nsim detail records or plot files.
            LOG.info('%s %d MC global mean %.4f%%, P2.5–P97.5 [%.4f%%, %.4f%%]',
                     cfg['crop'], year, mc_mean*100, low*100, high*100)
    if not summary_rows:
        raise RuntimeError('No MC years processed')
    # Existing 8 annual GeoTIFFs remain unchanged. One small Excel workbook is the
    # only tabular export; the 5,000 per-year samples live in memory only until
    # global MC statistics have been computed.
    save_excel(output_dir / f'MC_Global_Method5_{cfg["crop"]}_Annual.xlsx',
               summary_rows, plot_rows, records, cfg)
    LOG.info('All done: %s', output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='YAML configuration filename')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(load_config(args.config))


if __name__ == '__main__':
    main()
