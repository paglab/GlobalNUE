#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-at-a-Time sensitivity of the EXISTING deterministic M2 weighted-median NUE.

Six optional factors: Yield, synthetic fertilizer F, manure M, deposition D,
seed nitrogen SN, biological fixation BNF. Each OAT scenario perturbs only one
factor (all of its source products, where applicable) by a uniform relative
fraction; source weights and the other factors remain unchanged.

M2 input pre-processing, Nup regression/floor/manure switch, annual source-weight
normalization and pixel-level weighted median replicate the existing M2 code.
No Monte Carlo is performed. This is LOCAL sensitivity, not an uncertainty CI.

Primary Tornado response: change in GLOBAL NUE according to Method 5:
  sum_i(Nup_selected_i * HA_i) / sum_i((Nup_selected_i / NUE_M2_median_i) * HA_i).
Nup is taken from the *same source combination selected closest to the M2 NUE
weighted median*, exactly as in the deterministic M2 Nup export. Both terms
and all 13 OAT scenarios share the identical valid-pixel mask for each year.
A required YEAR-SPECIFIC processed NUE raster defines the permitted spatial
domain for all model outputs and the global summary. It supplies only a valid
pixel MASK, not numerical NUE values. Method 5 may further exclude cells where
NUE/Nup/HA or the scenario terms are invalid.
The existing pixel_mean and harvested_area_weighted modes are retained only
for backward compatibility; Method 5 is the intended analysis here.
Annual Method-5 ratios are averaged with EQUAL YEAR weights for the period plot.

Use from Windows CMD:
    python GlobalNUE_M2_OAT_Sensitivity_Method5.py --config config_GlobalNUE_OAT_Method5_Wheat.yaml
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt

LOG = logging.getLogger('M2_OAT')
FACTORS = ('yield', 'F', 'M', 'D', 'SN', 'BNF')
DATA_FACTORS = FACTORS[:4]


def read_config(config_path):
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('YAML must contain a mapping')
    if cfg.get('scientific_settings_confirmed') is not True:
        raise ValueError('scientific_settings_confirmed must be true after reviewing M2 source weights')
    for key in ('years', 'output_dir', 'sources', 'nup_equation', 'SN', 'BNF'):
        if key not in cfg:
            raise ValueError(f'Missing config key {key}')
    eq = cfg['nup_equation']
    for key in ('yield_coefficient', 'fertilizer_manure_coefficient', 'intercept'):
        if eq.get(key) is None or not math.isfinite(float(eq[key])):
            raise ValueError(f'nup_equation.{key} must be finite')
        eq[key] = float(eq[key])
    eq['min_nup'] = float(eq.get('min_nup', 3.17))
    if not math.isfinite(eq['min_nup']) or eq['min_nup'] < 0:
        raise ValueError('nup_equation.min_nup must be >=0 and finite')
    eq['include_manure'] = eq.get('include_manure', True)
    if not isinstance(eq['include_manure'], bool):
        raise ValueError('nup_equation.include_manure must be boolean')
    for name in ('SN', 'BNF'):
        if float(cfg[name]) < 0 or not math.isfinite(float(cfg[name])):
            raise ValueError(f'{name} must be finite and nonnegative')
    for factor in DATA_FACTORS:
        sources = cfg['sources'].get(factor)
        if not isinstance(sources, list) or not sources:
            raise ValueError(f'Sources missing for {factor}')
        group_weights, ids = {}, set()
        for src in sources:
            for key in ('id', 'directory', 'group', 'group_weight', 'within_group_weight'):
                if key not in src:
                    raise ValueError(f'{factor}: source missing {key}')
            if src['id'] in ids:
                raise ValueError(f'{factor}: duplicate ID {src["id"]}')
            ids.add(src['id'])
            gw, sw = float(src['group_weight']), float(src['within_group_weight'])
            if not (math.isfinite(gw) and math.isfinite(sw) and gw >= 0 and sw >= 0):
                raise ValueError(f'{factor}/{src["id"]}: invalid source weights')
            if src['group'] in group_weights and not np.isclose(group_weights[src['group']], gw):
                raise ValueError(f'{factor}: inconsistent group weight for {src["group"]}')
            group_weights[src['group']] = gw
            src['probability'] = gw * sw
        if not np.isclose(sum(group_weights.values()), 1, atol=1e-6):
            raise ValueError(f'{factor}: group weights must sum to 1')
        for group in group_weights:
            within = sum(float(s['within_group_weight']) for s in sources if s['group'] == group)
            if not np.isclose(within, 1, atol=1e-6):
                raise ValueError(f'{factor}/{group}: within-group weights must sum to 1')
    if len(cfg['years']) != 2 or int(cfg['years'][0]) > int(cfg['years'][1]):
        raise ValueError('years must be [first,last]')
    if int(cfg.get('block_height', 64)) < 1 or int(cfg.get('block_width', 64)) < 1:
        raise ValueError('Block size must be positive')
    decimals = cfg.get('output_decimals', 6)
    if decimals is not None and not (0 <= int(decimals) <= 8):
        raise ValueError('output_decimals must be 0..8 or null')
    oat = cfg.get('oat', {})
    change = float(oat.get('relative_change', 0.15))
    if not (math.isfinite(change) and 0 < change < 1):
        raise ValueError('oat.relative_change must be >0 and <1')
    chosen = oat.get('factors', list(FACTORS))
    if not chosen or len(set(chosen)) != len(chosen) or any(v not in FACTORS for v in chosen):
        raise ValueError(f'oat.factors must be a unique nonempty subset of {FACTORS}')
    aggregation = cfg.get('aggregation', {})
    mode = aggregation.get('mode', 'pixel_mean')
    if mode not in ('pixel_mean', 'harvested_area_weighted', 'nup_ratio_of_sums'):
        raise ValueError('aggregation.mode must be pixel_mean, harvested_area_weighted or nup_ratio_of_sums')
    if mode in ('harvested_area_weighted', 'nup_ratio_of_sums'):
        ha = aggregation.get('harvested_area', {})
        if not ha.get('directory') or not ha.get('filename_template'):
            raise ValueError(f'{mode} requires harvested_area.directory and harvested_area.filename_template')
        if '{year}' not in ha['filename_template']:
            raise ValueError('harvested_area.filename_template must contain {year}')
    for k in ('area_threshold', 'nue_min_value', 'nup_min_value'):
        value = float(aggregation.get(k, 0.0))
        if not math.isfinite(value):
            raise ValueError(f'aggregation.{k} must be finite')
        aggregation[k] = value
    if not isinstance(aggregation.get('use_written_raster_precision', True), bool):
        raise ValueError('aggregation.use_written_raster_precision must be true/false')
    # User-approved, processed NUE product: hard spatial mask for each year.
    mask_spec = cfg.get('reference_nue_mask')
    if not isinstance(mask_spec, dict):
        raise ValueError('reference_nue_mask must specify directory and filename_template')
    if not mask_spec.get('directory') or not mask_spec.get('filename_template'):
        raise ValueError('reference_nue_mask.directory and filename_template are required')
    if '{year}' not in mask_spec['filename_template']:
        raise ValueError('reference_nue_mask.filename_template must include {year}')
    if not isinstance(mask_spec.get('additional_nodata', [-9999.0]), list):
        raise ValueError('reference_nue_mask.additional_nodata must be a list')
    for value in mask_spec.get('additional_nodata', [-9999.0]):
        if not math.isfinite(float(value)):
            raise ValueError('reference_nue_mask.additional_nodata values must be finite')
    parallel = cfg.get('parallel', {})
    if not isinstance(parallel.get('enabled', False), bool):
        raise ValueError('parallel.enabled must be true/false')
    if int(parallel.get('max_workers', 4)) < 1:
        raise ValueError('parallel.max_workers must be >=1')
    cfg.setdefault('crop', 'crop')
    return cfg


def index_sources(cfg):
    regex = re.compile(r'(\d{4})$')
    indexes = {}
    for var in DATA_FACTORS:
        indexes[var] = {}
        for src in cfg['sources'][var]:
            folder = Path(src['directory'])
            if not folder.is_dir():
                raise FileNotFoundError(f'Input source directory not found: {folder}')
            years = {}
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() not in ('.tif', '.tiff'):
                    continue
                m = regex.search(p.stem)
                if m:
                    year = int(m.group(1))
                    if year in years:
                        raise ValueError(f'Duplicate {year}: {var}/{src["id"]}')
                    years[year] = p
            indexes[var][src['id']] = years
    return indexes


def read_on_grid(path, ref):
    # Identical to M2's nearest-neighbor harmonization for F/M/D.
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs=ref.crs, transform=ref.transform,
                       width=ref.width, height=ref.height,
                       resampling=Resampling.nearest,
                       nodata=np.nan, dtype='float32') as vrt:
            a = vrt.read(1, masked=True).astype('float32')
    return np.ma.filled(a, np.nan)


def fill_f_nearest(f):
    valid = np.isfinite(f)
    if not valid.any():
        return np.zeros_like(f, dtype=np.float32)
    if valid.all():
        return f
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return f[tuple(idx)].astype('float32', copy=False)


def iter_windows(h, w, bh, bw):
    for row in range(0, h, bh):
        for col in range(0, w, bw):
            yield Window(col, row, min(bw, w-col), min(bh, h-row))


def weighted_median_like_M2(stack, weights):
    """Exact median logic from original M2 weighted_stats(), excluding unrelated stats."""
    n, h, w = stack.shape
    a = stack.reshape(n, -1).astype('float64', copy=False)
    valid = np.isfinite(a)
    ww = np.where(valid, np.asarray(weights, dtype='float64')[:, None], 0)
    total = ww.sum(axis=0)
    safe = np.where(valid, a, np.inf)
    order = np.argsort(safe, axis=0)
    sorted_a = np.take_along_axis(safe, order, axis=0)
    sorted_w = np.take_along_axis(ww, order, axis=0)
    cum = np.cumsum(sorted_w, axis=0)
    threshold = total * 0.5
    j = np.argmax(cum >= threshold[None, :], axis=0)
    cols = np.arange(a.shape[1])
    med = sorted_a[j, cols].copy()
    exact_half = ((total > 0) & np.isclose(cum[j, cols], threshold, atol=1e-12, rtol=1e-12)
                  & (j+1 < n))
    if np.any(exact_half):
        for col in np.where(exact_half)[0]:
            for k in range(j[col]+1, n):
                if sorted_w[k, col] > 0:
                    med[col] = (med[col] + sorted_a[k, col])*0.5
                    break
    med[total == 0] = np.nan
    return med.reshape(h, w)


def make_scenarios(oat):
    relative = float(oat.get('relative_change', 0.15))
    scenarios = [('baseline', None, 0.0)]
    for factor in oat.get('factors', list(FACTORS)):
        scenarios.append((f'{factor}_minus', factor, -relative))
        scenarios.append((f'{factor}_plus', factor, +relative))
    return scenarios


def nearest_median_combo_indices(stack, med, weights):
    """Same deterministic choice rule used in M2 PFPN/Nup/Yield/F exports:
    closest finite combo NUE to weighted median; ties: greater combo weight,
    then lower source-combination index.
    """
    choice = np.full(med.shape, -1, dtype=np.int32)
    best_distance = np.full(med.shape, np.inf, dtype=np.float64)
    best_weight = np.full(med.shape, -np.inf, dtype=np.float64)
    med_valid = np.isfinite(med)
    for k, wk in enumerate(weights):
        if wk <= 0:
            continue
        a = stack[k]
        valid = np.isfinite(a) & med_valid
        if not np.any(valid):
            continue
        dist = np.abs(a.astype(np.float64, copy=False) - med)
        tie = valid & np.isclose(dist, best_distance, rtol=1e-12, atol=1e-12)
        better = valid & (((dist < best_distance) & ~tie) |
                          (tie & (wk > best_weight)))
        best_distance[better] = dist[better]
        best_weight[better] = wk
        choice[better] = k
    return choice


def tile_scenario(inputs, combinations, combo_weights, eq, sn, bnf, factor, perturb):
    """Return (M2 weighted-median NUE, median-selected-combination Nup).

    NUE formula, 32-bit combination arrays, and median remain unchanged from
    original M2; extra Nup is paired with the same selected source combination
    used by M2's Nup_M2_median_YYYY.tif, rather than an Nup weighted median.
    """
    factor_multiplier = 1.0 + perturb
    coef_y = eq['yield_coefficient']
    coef_n = eq['fertilizer_manure_coefficient']
    intercept = eq['intercept']
    min_nup = eq['min_nup']
    manure_in_nup = eq['include_manure']
    sn_s = sn * factor_multiplier if factor == 'SN' else sn
    bnf_s = bnf * factor_multiplier if factor == 'BNF' else bnf
    members = []
    nup_members = []
    for iy, iF, iM, iD in combinations:
        Y = inputs['yield'][iy]
        F = inputs['F'][iF]
        M = inputs['M'][iM]
        D = inputs['D'][iD]
        if factor == 'yield':
            Y = (Y * factor_multiplier).astype('float32')
        elif factor == 'F':
            F = (F * factor_multiplier).astype('float32')
        elif factor == 'M':
            M = (M * factor_multiplier).astype('float32')
        elif factor == 'D':
            D = (D * factor_multiplier).astype('float32')
        nup_n_input = F + M if manure_in_nup else F
        nup = coef_y * Y + coef_n * nup_n_input + intercept
        nup = np.where(np.isfinite(nup), np.maximum(nup, min_nup), np.nan)
        denominator = F + M + D + sn_s + bnf_s
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            nue = np.where((Y > 0) & np.isfinite(Y) & np.isfinite(nup)
                           & np.isfinite(denominator) & (denominator != 0),
                           nup / denominator, np.nan)
        members.append(np.asarray(nue, dtype=np.float32))
        nup_members.append(np.asarray(nup, dtype=np.float32))

    nue_stack = np.stack(members, axis=0)
    median = weighted_median_like_M2(nue_stack, combo_weights)
    selected = nearest_median_combo_indices(nue_stack, median, combo_weights)
    selected_nup = np.full(median.shape, np.nan, dtype=np.float32)
    rows, cols = np.where(selected >= 0)
    if rows.size:
        nup_stack = np.stack(nup_members, axis=0)
        selected_nup[rows, cols] = nup_stack[selected[rows, cols], rows, cols]
    return median, selected_nup


def tile_median(inputs, combinations, combo_weights, eq, sn, bnf, factor, perturb):
    """Backward-compatible helper returning NUE only."""
    return tile_scenario(inputs, combinations, combo_weights, eq, sn, bnf,
                         factor, perturb)[0]


def load_year_arrays(cfg, available, ref):
    arrays = {}
    for variable in DATA_FACTORS:
        arrays[variable] = []
        for src, path in available[variable]:
            if variable == 'yield':
                with rasterio.open(path) as ds:
                    if (ds.width != ref.width or ds.height != ref.height or
                        ds.crs != ref.crs or ds.transform != ref.transform):
                        raise ValueError(f'Yield source grid mismatch: {path}. Align it before OAT.')
                    arr = ds.read(1, masked=True).astype('float32').filled(np.nan)
            else:
                arr = read_on_grid(path, ref)
                if variable == 'F':
                    arr = fill_f_nearest(arr)
                else:
                    arr = np.nan_to_num(arr, nan=0.0)
            arrays[variable].append(arr)
        LOG.info('  %s: %d source(s) loaded', variable, len(arrays[variable]))
    return arrays


def load_ha(cfg, ref, year):
    """Method 5: load HA only on EXACT same grid as NUE; no reprojection/resampling.

    Supports filename in the configured directory or its subfolders, like
    user's original Method 5 script which used recursive year discovery.
    """
    agg = cfg.get('aggregation', {})
    if agg.get('mode', 'pixel_mean') == 'pixel_mean':
        return None
    spec = agg['harvested_area']
    folder = Path(spec['directory'])
    if not folder.is_dir():
        raise FileNotFoundError(f'Harvested-area directory missing: {folder}')
    filename = spec['filename_template'].format(year=year)
    path = folder / filename
    if not path.is_file():
        matches = sorted(p for p in folder.rglob('*') if p.is_file() and
                         p.name.casefold() == filename.casefold())
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected exactly 1 HA for {year}: {filename} '
                                    f'in {folder}; matches={matches}')
        path = matches[0]
    with rasterio.open(path) as ds:
        if (ds.crs is None or ref.crs is None or ds.crs != ref.crs or
            ds.width != ref.width or ds.height != ref.height or
            not ds.transform.almost_equals(ref.transform, precision=1e-10)):
            raise ValueError(f'{year}: NUE and harvested area grid mismatch; '
                             f'Method 5 forbids automatic resampling: {path}')
        arr = ds.read(1, masked=True).astype('float32')
        values = np.ma.filled(arr, np.nan)
    values = np.asarray(values, dtype=np.float32)
    for invalid in agg.get('additional_nodata', [-9999.0]):
        values[values == float(invalid)] = np.nan
    values[~np.isfinite(values)] = np.nan
    LOG.info('%s: matched harvested area %s (original aligned grid)', year, path)
    return values


def reference_nue_mask_path(cfg, year):
    '''Exact, mandatory annual reference raster path; do not silently skip years.'''
    spec = cfg['reference_nue_mask']
    folder = Path(spec['directory'])
    if not folder.is_dir():
        raise FileNotFoundError(f'Reference NUE mask directory not found: {folder}')
    filename = spec['filename_template'].format(year=year)
    path = folder / filename
    if not path.is_file():
        raise FileNotFoundError(f'{year}: required processed NUE mask missing: {path}')
    return path


def validate_reference_nue_mask_grid(ds, ref, year):
    '''Do not reproject/resample the reference mask: grid must be identical.'''
    if (ds.count != 1 or ds.crs is None or ref.crs is None or ds.crs != ref.crs or
        ds.width != ref.width or ds.height != ref.height or
        not ds.transform.almost_equals(ref.transform, precision=1e-10)):
        raise ValueError(f'{year}: processed NUE mask grid mismatch (CRS, size, '
                         f'transform or band count); resampling is forbidden: {ds.name}')


def reference_mask_window(mask_ds, win, extra_nodata):
    '''Use ONLY valid/NoData footprint; reference NUE numbers are never used.'''
    data = mask_ds.read(1, window=win, masked=True)
    arr = np.asarray(data.data)
    ok = (~np.ma.getmaskarray(data)) & np.isfinite(arr)
    for nd in extra_nodata:
        ok &= arr != float(nd)
    return ok


def output_profile(ref, nodata):
    p = ref.profile.copy()
    p.update(driver='GTiff', count=1, dtype='float32', nodata=nodata,
             compress='LZW', BIGTIFF='IF_SAFER')
    return p


def write_tile(ds, data, window, nodata, decimals):
    arr = np.where(np.isfinite(data), data, nodata)
    if decimals is not None:
        arr = np.where(arr == nodata, arr, np.round(arr, int(decimals)))
    ds.write(np.asarray(arr, dtype='float32'), 1, window=window)


def run(cfg):
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    index = index_sources(cfg)
    years = range(int(cfg['years'][0]), int(cfg['years'][1]) + 1)
    oat = cfg.get('oat', {})
    scenarios = make_scenarios(oat)
    nodata = float(cfg.get('out_nodata', -9999))
    bh, bw = int(cfg.get('block_height', 64)), int(cfg.get('block_width', 64))
    decimals = cfg.get('output_decimals', 6)
    output_options = cfg.get('raster_outputs', {})
    parallel = cfg.get('parallel', {})
    aggregation = cfg.get('aggregation', {})
    mode = aggregation.get('mode', 'pixel_mean')
    use_threads = parallel.get('enabled', False) and int(parallel.get('max_workers', 4)) > 1
    annual = []
    executed_years = []
    for year in years:
        available = {v: [(s, index[v][s['id']][year]) for s in cfg['sources'][v]
                         if year in index[v][s['id']]] for v in DATA_FACTORS}
        missing = [var for var, entries in available.items() if not entries]
        if missing:
            LOG.warning('%s skipped: no data for %s', year, ', '.join(missing))
            continue
        mask_path = reference_nue_mask_path(cfg, year)
        with rasterio.open(available['yield'][0][1]) as ref, rasterio.open(mask_path) as mask_ds:
            validate_reference_nue_mask_grid(mask_ds, ref, year)
            if ref.crs is None:
                raise ValueError(f'Yield raster lacks CRS: {available["yield"][0][1]}')
            arrays = load_year_arrays(cfg, available, ref)
            ha = load_ha(cfg, ref, year)
            norm = {v: sum(s['probability'] for s, _ in available[v]) for v in DATA_FACTORS}
            if any(v <= 0 for v in norm.values()):
                raise ValueError(f'{year}: source weights sum to zero for some factor')
            import itertools
            combo_indices = list(itertools.product(*(range(len(arrays[v])) for v in DATA_FACTORS)))
            weights = [math.prod(available[v][ix][0]['probability'] / norm[v]
                                 for v, ix in zip(DATA_FACTORS, indices)) for indices in combo_indices]
            LOG.info('%s: %d combos, %d scenarios, aggregation=%s',
                     year, len(combo_indices), len(scenarios), mode)
            means_numerators = {name: 0.0 for name, _, _ in scenarios}
            # Method5's denominator sum is the total input N, not total HA.
            ratio_denominators = {name: 0.0 for name, _, _ in scenarios}
            sum_weight = 0.0
            n_cells = 0
            reference_mask_cells = 0
            reference_and_ha_cells = 0
            n_tiles = 0
            profile = output_profile(ref, nodata)
            with contextlib.ExitStack() as outputs:
                executor = (outputs.enter_context(ThreadPoolExecutor(
                    max_workers=int(parallel.get('max_workers', 4)))) if use_threads else None)
                base_writer = (outputs.enter_context(rasterio.open(
                    out / f'NUE_M2_OAT_baseline_{year}.tif', 'w', **profile))
                    if output_options.get('baseline_median', True) else None)
                nup_base_writer = (outputs.enter_context(rasterio.open(
                    out / f'Nup_M2_OAT_baseline_{year}.tif', 'w', **profile))
                    if output_options.get('baseline_nup', mode == 'nup_ratio_of_sums') else None)
                delta_writers = {}
                if output_options.get('delta_pp', True):
                    for name, factor, _ in scenarios[1:]:
                        delta_writers[name] = outputs.enter_context(rasterio.open(
                            out / f'NUE_M2_OAT_delta_pp_{name}_{year}.tif', 'w', **profile))
                sensitivity_writers = {}
                if output_options.get('sensitivity_coefficient', True):
                    for factor in oat.get('factors', list(FACTORS)):
                        sensitivity_writers[factor] = outputs.enter_context(rasterio.open(
                            out / f'NUE_M2_OAT_sensitivity_{factor}_{year}.tif', 'w', **profile))
                for win in iter_windows(ref.height, ref.width, bh, bw):
                    row, col = int(win.row_off), int(win.col_off)
                    hh, ww = int(win.height), int(win.width)
                    mask_valid = reference_mask_window(
                        mask_ds, win,
                        cfg['reference_nue_mask'].get('additional_nodata', [-9999.0]))
                    reference_mask_cells += int(np.count_nonzero(mask_valid))
                    blocks = {v: [a[row:row+hh, col:col+ww] for a in arrays[v]] for v in DATA_FACTORS}
                    # All scenario calculations are restricted to processed NUE's
                    # reference footprint. No filling or extrapolation outside it.
                    blocks['yield'] = [np.where(mask_valid, a, np.nan)
                                       for a in blocks['yield']]
                    if executor is None:
                        computed = {name: tile_scenario(blocks, combo_indices, weights,
                                            cfg['nup_equation'], float(cfg['SN']),
                                            float(cfg['BNF']), factor, perturb)
                                    for name, factor, perturb in scenarios}
                    else:
                        futures = {name: executor.submit(
                            tile_scenario, blocks, combo_indices, weights,
                            cfg['nup_equation'], float(cfg['SN']), float(cfg['BNF']),
                            factor, perturb) for name, factor, perturb in scenarios}
                        computed = {name: fut.result() for name, fut in futures.items()}
                    values = {name: value[0] for name, value in computed.items()}
                    nups = {name: value[1] for name, value in computed.items()}
                    base = values['baseline']
                    # The processed NUE raster is the hard, year-specific region.
                    # A scenario cannot contribute an outside pixel even if a
                    # source gap-filling step produced a numerical value there.
                    common = mask_valid & np.logical_and.reduce(
                        [np.isfinite(a) for a in values.values()])
                    if ha is not None:
                        hblock = ha[row:row+hh, col:col+ww]
                        common &= np.isfinite(hblock) & (
                            hblock > float(aggregation.get('area_threshold', 0.0)))
                    else:
                        hblock = None
                    if hblock is not None:
                        reference_and_ha_cells += int(np.count_nonzero(
                            mask_valid & np.isfinite(hblock) &
                            (hblock > float(aggregation.get('area_threshold', 0.0)))))
                    else:
                        reference_and_ha_cells += int(np.count_nonzero(mask_valid))
                    if mode == 'nup_ratio_of_sums':
                        # Strict Method 5: core NUE > 0, Nup >= 0, HA > 0;
                        # identical common pixels in both sums AND all scenarios.
                        nue_min = float(aggregation.get('nue_min_value', 0.0))
                        nup_min = float(aggregation.get('nup_min_value', 0.0))
                        # Match the existing Method 5 script reading M2 GeoTIFFs:
                        # round to configured on-disk precision before summation.
                        if aggregation.get('use_written_raster_precision', True) and decimals is not None:
                            agg_values = {name: np.round(arr, int(decimals)).astype(np.float32)
                                          for name, arr in values.items()}
                            agg_nups = {name: np.round(arr, int(decimals)).astype(np.float32)
                                        for name, arr in nups.items()}
                        else:
                            agg_values, agg_nups = values, nups
                        for name in values:
                            common &= (agg_values[name] > nue_min) & np.isfinite(agg_nups[name]) & (
                                agg_nups[name] >= nup_min)
                        # Skip any pixel with non-finite numerator/denominator term
                        # in even one scenario; prevent overflow and division issues.
                        terms = {}
                        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                            for name in values:
                                nu = agg_nups[name].astype(np.float64, copy=False)
                                ne = agg_values[name].astype(np.float64, copy=False)
                                a = hblock.astype(np.float64, copy=False)
                                numerator = nu * a
                                denominator = (nu / ne) * a
                                terms[name] = (numerator, denominator)
                                common &= np.isfinite(numerator) & np.isfinite(denominator)
                        if np.any(common):
                            n_cells += int(common.sum())
                            sum_weight += float(np.sum(hblock[common], dtype=np.float64))
                            for name, (numerator, denominator) in terms.items():
                                means_numerators[name] += float(np.sum(numerator[common], dtype=np.float64))
                                ratio_denominators[name] += float(np.sum(denominator[common], dtype=np.float64))
                    else:
                        if np.any(common):
                            wt = (hblock[common].astype('float64', copy=False)
                                  if hblock is not None else None)
                            weight_here = float(np.sum(wt)) if wt is not None else int(common.sum())
                            sum_weight += weight_here
                            n_cells += int(common.sum())
                            for name in means_numerators:
                                vals = values[name][common]
                                means_numerators[name] += (float(np.sum(vals * wt, dtype=np.float64))
                                                            if wt is not None else float(np.sum(vals, dtype=np.float64)))
                    if base_writer is not None:
                        write_tile(base_writer, np.where(mask_valid, base, np.nan),
                                   win, nodata, decimals)
                    if nup_base_writer is not None:
                        write_tile(nup_base_writer,
                                   np.where(mask_valid, nups['baseline'], np.nan),
                                   win, nodata, decimals)
                    for name, writer in delta_writers.items():
                        diff = np.where(np.isfinite(base) & np.isfinite(values[name]),
                                        (values[name] - base) * 100.0, np.nan)
                        write_tile(writer, np.where(mask_valid, diff, np.nan), win, nodata, decimals)
                    for factor, writer in sensitivity_writers.items():
                        low, high = values[f'{factor}_minus'], values[f'{factor}_plus']
                        change = float(oat.get('relative_change', 0.15))
                        with np.errstate(divide='ignore', invalid='ignore'):
                            sensitivity = np.where(np.isfinite(base) & (base != 0) &
                                                   np.isfinite(low) & np.isfinite(high),
                                                   (high - low) / (2.0 * change * base), np.nan)
                        write_tile(writer, np.where(mask_valid, sensitivity, np.nan), win, nodata, decimals)
                    n_tiles += 1
                    if n_tiles % 100 == 0:
                        LOG.info('%s: %d tiles processed', year, n_tiles)
            if reference_mask_cells == 0:
                raise ValueError(f'{year}: processed NUE reference raster has no valid mask pixels: {mask_path}')
            dropped = reference_and_ha_cells - n_cells
            if dropped:
                LOG.warning('%s: %d/%d reference-mask-and-HA cells excluded from '
                            'Method5/global comparison because NUE/Nup/scenario validity '
                            'requirements were not met', year, dropped, reference_and_ha_cells)
            if n_cells == 0 or (mode == 'nup_ratio_of_sums' and
                                any(ratio_denominators[name] <= 0 for name in ratio_denominators)):
                LOG.warning('%s: no valid common pixels or nonpositive N-input denominator', year)
                continue
            executed_years.append(year)
            if mode == 'nup_ratio_of_sums':
                baseline_mean = means_numerators['baseline'] / ratio_denominators['baseline']
            else:
                baseline_mean = means_numerators['baseline'] / sum_weight
            for name, factor, perturb in scenarios:
                scenario_mean = (means_numerators[name] / ratio_denominators[name]
                                 if mode == 'nup_ratio_of_sums' else means_numerators[name] / sum_weight)
                annual.append(dict(crop=cfg['crop'], year=year,
                                   scenario=name, factor=factor or 'baseline',
                                   direction='baseline' if factor is None else ('minus' if perturb < 0 else 'plus'),
                                   relative_input_change=perturb,
                                   baseline_global_NUE_fraction=baseline_mean,
                                   scenario_global_NUE_fraction=scenario_mean,
                                   baseline_global_NUE_percent=100*baseline_mean,
                                   scenario_global_NUE_percent=100*scenario_mean,
                                   delta_pp=100*(scenario_mean-baseline_mean),
                                   delta_relative_percent=(100*(scenario_mean-baseline_mean)/baseline_mean
                                                           if baseline_mean != 0 else np.nan),
                                   valid_common_pixels=n_cells,
                                   reference_mask_pixels=reference_mask_cells,
                                   reference_and_ha_pixels=reference_and_ha_cells,
                                   excluded_after_reference_and_ha=reference_and_ha_cells-n_cells,
                                   reference_mask_file=str(mask_path),
                                   sum_spatial_weights=sum_weight,
                                   sum_nup_ha=(means_numerators[name] if mode == 'nup_ratio_of_sums' else np.nan),
                                   sum_ninput_ha=(ratio_denominators[name] if mode == 'nup_ratio_of_sums' else np.nan),
                                   aggregation=mode))
            LOG.info('%s: Method=%s, baseline global NUE=%.5f%%; common cells=%d',
                     year, mode, baseline_mean*100, n_cells)
    if not annual:
        raise RuntimeError('No valid years. Check source folders/year suffix and harvested-area rasters.')
    annual_df = pd.DataFrame(annual)
    annual_df.to_csv(out / 'OAT_global_annual_scenarios.csv', index=False, float_format='%.12g')
    factors = oat.get('factors', list(FACTORS))
    tornado_annual, tornado_period = [], []
    for year in executed_years:
        sample = annual_df[annual_df.year == year]
        bas = sample.loc[sample.scenario=='baseline'].iloc[0]
        for factor in factors:
            minus = sample.loc[sample.scenario == factor+'_minus'].iloc[0]
            plus = sample.loc[sample.scenario == factor+'_plus'].iloc[0]
            tornado_annual.append(dict(crop=cfg['crop'], year=year, factor=factor,
                                       baseline_NUE_percent=bas.baseline_global_NUE_percent,
                                       minus_NUE_percent=minus.scenario_global_NUE_percent,
                                       plus_NUE_percent=plus.scenario_global_NUE_percent,
                                       minus_delta_pp=minus.delta_pp, plus_delta_pp=plus.delta_pp,
                                       span_pp=abs(plus.delta_pp-minus.delta_pp),
                                       aggregation=bas.aggregation))
    yr_tornado = pd.DataFrame(tornado_annual)
    yr_tornado.to_csv(out/'OAT_tornado_annual.csv', index=False, float_format='%.12g')
    # Arithmetic mean of each year's global mean; each year has equal influence.
    for factor in factors:
        r = yr_tornado[yr_tornado.factor == factor]
        tornado_period.append(dict(crop=cfg['crop'], factor=factor,
                                   years_first=min(executed_years), years_last=max(executed_years),
                                   n_years=len(executed_years),
                                   input_change_percent=100*float(oat.get('relative_change', 0.15)),
                                   baseline_NUE_percent=r.baseline_NUE_percent.mean(),
                                   minus_NUE_percent=r.minus_NUE_percent.mean(),
                                   plus_NUE_percent=r.plus_NUE_percent.mean(),
                                   minus_delta_pp=r.minus_delta_pp.mean(),
                                   plus_delta_pp=r.plus_delta_pp.mean(),
                                   span_pp=abs(r.plus_delta_pp.mean()-r.minus_delta_pp.mean()),
                                   aggregation=mode,
                                   temporal_aggregation='equal_year_mean'))
    summary = pd.DataFrame(tornado_period).sort_values('span_pp', ascending=False)
    summary.insert(0, 'sensitivity_rank', np.arange(1, len(summary)+1))
    summary.to_csv(out/'OAT_tornado_global.csv', index=False, float_format='%.12g')
    metadata = {
        'method': 'deterministic source-weighted M2 median re-evaluated for each OAT scenario',
        'crop': cfg['crop'], 'years_requested': cfg['years'], 'years_completed': executed_years,
        'oat': oat, 'nup_equation': cfg['nup_equation'],
        'SN': cfg['SN'], 'BNF': cfg['BNF'],
        'aggregation': aggregation,
        'parallel': parallel,
        'spatial_comparison': ('hard annual processed-NUE valid-pixel mask AND intersection of all scenario '
                               'validity conditions, including Nup and HA for Method5'),
        'reference_nue_mask': cfg['reference_nue_mask'],
        'reference_mask_meaning': ('valid pixels only (finite, non-NoData); processed NUE '
                                   'numeric values are NOT used in the calculation'),
        'temporal_comparison': 'equal weight across included annual global ratios of sums',
        'input_source_groups': cfg['sources'],
        'units': {'NUE': 'ratio (normally 0..1)', 'delta_pp':'percentage points of NUE, NOT percent change',
                  'sensitivity_coefficient': '(NUE_plus - NUE_minus)/(2*relative_change*NUE_baseline)'},
        'nup_selection': 'closest valid source-combination NUE to M2 weighted median; ties by higher weight then lower index',
        'method5_formula': 'sum(Nup_selected*HA) / sum((Nup_selected/NUE_M2_weighted_median)*HA)',
        'important': 'Local OAT sensitivity, not MC confidence interval. Method 5 is NOT the HA-weighted mean. '
                     'Method 5 can round NUE/Nup to output TIFF precision before spatial summation. ' 'All outputs must lie within the annual processed-NUE reference mask.'}
    with open(out/'OAT_run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    if cfg.get('plot', {}).get('create_tornado', True):
        try:
            plot_tornado(out/'OAT_tornado_global.csv', out,
                         dpi=int(cfg.get('plot', {}).get('dpi', 300)))
        except ImportError:
            LOG.warning('matplotlib not installed: CSV produced but no plot. Install matplotlib to plot.')
    LOG.info('Finished OAT: %s', out)
    return summary


def plot_tornado(csv_path, output_directory=None, dpi=300):
    """Publication-ready global tornado with separate +/- bars and 0-pp axis."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df = pd.read_csv(csv_path).sort_values('span_pp', ascending=True)
    if df.empty:
        raise ValueError('Empty tornado table')
    factor_names = {'yield':'Yield', 'F':'Synthetic fertilizer N', 'M':'Manure N',
                    'D':'Atmospheric deposition N', 'SN':'Seed N', 'BNF':'Biological N fixation'}
    labs = [factor_names.get(x, x) for x in df.factor]
    n = len(df)
    fig, ax = plt.subplots(figsize=(9, max(4.2, 0.68*n+1.7)), layout='constrained')
    y = np.arange(n)
    change = float(df.input_change_percent.iloc[0])
    ax.barh(y-0.18, df.minus_delta_pp, height=0.34, color='#3274A1',
            label=f'Input −{change:g}%')
    ax.barh(y+0.18, df.plus_delta_pp, height=0.34, color='#D27D35',
            label=f'Input +{change:g}%')
    ax.axvline(0, color='black', linewidth=0.9)
    ax.set_yticks(y, labs)
    ax.set_xlabel('Change in global NUE (percentage points)')
    years = f'{int(df.years_first.iloc[0])}–{int(df.years_last.iloc[0])}'
    label = df.crop.iloc[0].capitalize()
    ax.set_title(f'{label} | Global NUE OAT sensitivity ({years})', pad=12)
    ax.grid(axis='x', color='0.87', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='lower right')
    ax.text(0.02, -0.14,
            'Annual global values averaged equally across years; method: '
            + str(df.aggregation.iloc[0]).replace('_',' '),
            transform=ax.transAxes, va='top', fontsize=9, color='0.4')
    out = Path(output_directory) if output_directory is not None else Path(csv_path).parent
    out.mkdir(parents=True, exist_ok=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'OAT_global_tornado.{ext}', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out/'OAT_global_tornado.png'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(read_config(args.config))


if __name__ == '__main__':
    main()
