#!/usr/bin/env python3
"""Global wheat NUE M2: source-group-balanced weighted Cartesian ensemble.

M2 HERE IS AN EXPLICIT OPERATIONAL ASSUMPTION:
  * Calculate Nup with the configured linear model for every Yield x F x M x D combination.
  * Assign probabilities hierarchically (group -> source within group).
  * Combination probability is the product of the four source probabilities.
  * Pixel M2 = weighted median across valid combinations, renormalizing weights
    over available/valid combinations on that pixel.

Additional weighted ensemble percentiles: P2.5, P97.5, and their difference.
Optional manure term and minimum Nup; annual PFPN, Nup, Yield and F for the median-selected combo.
No Monte Carlo is performed by this script. Scientific source grouping and
weights MUST be reviewed in YAML before production use.
"""
import argparse
import contextlib
import itertools
import json
import logging
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as transform_coords
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
import yaml

LOG = logging.getLogger('NUE_M2')


def read_config(filename):
    with open(filename, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError('YAML root must be a mapping')
    if not cfg.get('scientific_settings_confirmed', False):
        raise ValueError('Set scientific_settings_confirmed: true in YAML ONLY after confirming groups, weights and M2 definition.')
    equation = cfg.get('nup_equation')
    if not isinstance(equation, dict):
        raise ValueError('YAML must contain nup_equation with all three coefficients')
    for key in ('yield_coefficient', 'fertilizer_manure_coefficient', 'intercept'):
        if equation.get(key) is None:
            raise ValueError(f'nup_equation.{key} is required; provide the fitted coefficient')
        try:
            value = float(equation[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f'nup_equation.{key} must be numeric') from exc
        if not math.isfinite(value):
            raise ValueError(f'nup_equation.{key} must be finite')
        equation[key] = value
    # Optional Nup settings. Existing defaults retain manure in Nup.
    min_nup = equation.get('min_nup', 3.17)
    try:
        min_nup = float(min_nup)
    except (TypeError, ValueError) as exc:
        raise ValueError('nup_equation.min_nup must be numeric') from exc
    if not math.isfinite(min_nup) or min_nup < 0:
        raise ValueError('nup_equation.min_nup must be finite and >= 0')
    equation['min_nup'] = min_nup
    include_manure = equation.get('include_manure', True)
    if not isinstance(include_manure, bool):
        raise ValueError('nup_equation.include_manure must be true or false')
    equation['include_manure'] = include_manure
    for var in ('yield', 'F', 'M', 'D'):
        if var not in cfg['sources'] or not cfg['sources'][var]:
            raise ValueError(f'No sources for {var}')
        names = set()
        group_weights = {}
        for s in cfg['sources'][var]:
            for k in ('id', 'directory', 'group', 'group_weight', 'within_group_weight'):
                if k not in s:
                    raise ValueError(f'{var}: source missing {k}')
            if s['id'] in names:
                raise ValueError(f'{var}: duplicate source ID {s["id"]}')
            names.add(s['id'])
            gw = float(s['group_weight'])
            sw = float(s['within_group_weight'])
            if not (math.isfinite(gw) and math.isfinite(sw) and gw >= 0 and sw >= 0):
                raise ValueError(f'{var}/{s["id"]}: invalid weight')
            if s['group'] in group_weights and not np.isclose(group_weights[s['group']], gw):
                raise ValueError(f'{var}: inconsistent group_weight for group {s["group"]}')
            group_weights[s['group']] = gw
        if not np.isclose(sum(group_weights.values()), 1, atol=1e-6):
            raise ValueError(f'{var}: group weights must sum to 1')
        for group in group_weights:
            w = sum(float(s['within_group_weight']) for s in cfg['sources'][var] if s['group'] == group)
            if not np.isclose(w, 1, atol=1e-6):
                raise ValueError(f'{var}/{group}: within-group weights must sum to 1')
        for s in cfg['sources'][var]:
            s['probability'] = float(s['group_weight']) * float(s['within_group_weight'])
    y0, y1 = cfg['years']
    if int(y0) > int(y1):
        raise ValueError('Invalid years')
    return cfg


def index_sources(cfg):
    regex = re.compile(r'(\d{4})$')
    data = {}
    for variable, sources in cfg['sources'].items():
        data[variable] = {}
        for s in sources:
            directory = Path(s['directory'])
            if not directory.is_dir():
                raise FileNotFoundError(f'Missing input directory: {directory}')
            by_year = {}
            for p in sorted(directory.iterdir()):
                if p.suffix.lower() not in ('.tif', '.tiff'):
                    continue
                m = regex.search(p.stem)
                if m:
                    year = int(m.group(1))
                    if year in by_year:
                        raise ValueError(f'Duplicate {year} for {variable}/{s["id"]}')
                    by_year[year] = p
            data[variable][s['id']] = by_year
    return data


def read_on_grid(path, ref):
    """Read one full raster on the chosen reference grid using nearest-neighbor."""
    with rasterio.open(path) as src:
        with WarpedVRT(src, crs=ref.crs, transform=ref.transform, width=ref.width,
                       height=ref.height, resampling=Resampling.nearest,
                       nodata=np.nan, dtype='float32') as vrt:
            a = vrt.read(1, masked=True).astype('float32')
    return np.ma.filled(a, np.nan)


def fill_f_nearest(f):
    valid = np.isfinite(f)
    if not valid.any():
        # Preserve M0's zero fallback if NO fertilizer source pixel is available.
        return np.zeros_like(f, dtype=np.float32)
    if valid.all():
        return f
    idx = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return f[tuple(idx)].astype('float32', copy=False)


def iter_windows(h, w, bh, bw):
    for r in range(0, h, bh):
        for c in range(0, w, bw):
            yield Window(c, r, min(bw, w-c), min(bh, h-r))


def weighted_percentile_sorted(values, weights, total, q):
    """Weighted percentile, linear interpolation of weighted plotting positions.

    values/weights: (n, pixels), already sorted by ascending values;
    invalid values must have zero weight. Positive-weight sample midpoints
    are rescaled from [first midpoint, last midpoint] to [0, 1]. This
    agrees with NumPy's default linear percentile for uniform weights.
    This is a descriptive ensemble percentile, not a probabilistic CI.
    """
    n, size = values.shape
    positive = weights > 0
    idx = np.arange(n, dtype=np.int32)[:, None]
    cum = np.cumsum(weights, axis=0)
    centers = cum - weights / 2
    first_idx = np.argmax(positive, axis=0)
    last_idx = n - 1 - np.argmax(positive[::-1], axis=0)
    cols = np.arange(size)
    first = centers[first_idx, cols]
    last = centers[last_idx, cols]
    target = first + q * (last - first)
    # Find first positive-weight observation at or beyond the target.
    at_or_above = positive & (centers >= target[None, :])
    upper = np.argmax(at_or_above, axis=0)
    upper = np.where(np.any(at_or_above, axis=0), upper, last_idx)
    previous_positive = np.maximum.accumulate(np.where(positive, idx, -1), axis=0)
    lower = previous_positive[np.maximum(upper - 1, 0), cols]
    lower = np.where(lower < 0, upper, lower)
    v0 = np.where(total > 0, values[lower, cols], 0.0)
    v1 = np.where(total > 0, values[upper, cols], 0.0)
    x0 = centers[lower, cols]
    x1 = centers[upper, cols]
    fraction = np.zeros(size, dtype=np.float64)
    np.divide(target-x0, x1-x0, out=fraction, where=(x1 > x0))
    result = v0 + np.clip(fraction, 0.0, 1.0) * (v1-v0)
    result[total <= 0] = np.nan
    return result


def weighted_stats(stack, weights):
    """stack (n,h,w). Valid-cell probability is renormalized pixel by pixel.

    Weighted median uses mid-point of adjacent values when cumulative weight
    reaches exactly 0.5 (consistent with np.median for 2 equal weights).
    """
    n, h, w = stack.shape
    a = stack.reshape(n, -1).astype('float64', copy=False)
    valid = np.isfinite(a)
    counts = valid.sum(axis=0)
    ww = np.where(valid, np.asarray(weights, dtype='float64')[:, None], 0)
    total = ww.sum(axis=0)
    avg = np.full(a.shape[1], np.nan)
    np.divide(np.sum(np.where(valid, a, 0) * ww, axis=0), total,
              out=avg, where=total>0)
    safe = np.where(valid, a, np.inf)
    order = np.argsort(safe, axis=0)
    sorted_a = np.take_along_axis(safe, order, axis=0)
    sorted_w = np.take_along_axis(ww, order, axis=0)
    # Zero-weight members must not affect a weighted percentile.
    sorted_a_for_quantiles = np.where(sorted_w > 0, sorted_a, np.inf)
    p2_5 = weighted_percentile_sorted(sorted_a_for_quantiles, sorted_w, total, 0.025)
    p97_5 = weighted_percentile_sorted(sorted_a_for_quantiles, sorted_w, total, 0.975)
    cum = np.cumsum(sorted_w, axis=0)
    threshold = total * 0.5
    j = np.argmax(cum >= threshold[None, :], axis=0)
    cols = np.arange(a.shape[1])
    med = sorted_a[j, cols].copy()
    exact_half = (total > 0) & np.isclose(cum[j, cols], threshold, atol=1e-12, rtol=1e-12) & (j+1 < n)
    # Ignore zero-weight and invalid entries when searching the next positive weight.
    if np.any(exact_half):
        for col in np.where(exact_half)[0]:
            for k in range(j[col]+1, n):
                if sorted_w[k, col] > 0:
                    med[col] = (med[col] + sorted_a[k, col])*0.5
                    break
    med[total == 0] = np.nan
    maxv = np.max(np.where(valid, a, -np.inf), axis=0)
    minv = np.min(safe, axis=0)
    maxv[counts == 0] = np.nan
    minv[counts == 0] = np.nan
    return {k: v.reshape(h, w) for k,v in {
        'median': med, 'mean': avg, 'min': minv, 'max': maxv,
        'dif': maxv-minv, 'n': counts.astype('float64'),
        'weight_sum': total, 'p2_5': p2_5, 'p97_5': p97_5,
        'p95_diff': p97_5-p2_5}.items()}


def nearest_median_combo_indices(stack, med, weights):
    """For each pixel, choose the positive-weight valid NUE combination closest
    to the M2 weighted median. If multiple distances tie, prefer the larger
    combination weight, and then the earliest combination index.

    This does NOT change the original weighted median itself. A weighted median
    that is an average of two combination NUE values has no unique source,
    so this documented deterministic rule is used for PFPN provenance.
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
        # Treat differences within float64 round-off as ties.
        tie = valid & np.isclose(dist, best_distance, rtol=1e-12, atol=1e-12)
        better = valid & ((dist < best_distance) & ~tie | (tie & (wk > best_weight)))
        best_distance[better] = dist[better]
        best_weight[better] = wk
        choice[better] = k
    return choice


def pfpn_from_selected_combo(selected, combo_source_keys, yields, fertilizers,
                             window, nodata, decimals=None):
    """PFPN = Yield / fertilizer N for the selected M2 median combination.
    Requires strictly positive fertilizer N. No alternative combination is
    substituted if the selected combination has missing/zero fertilizer.
    """
    h, w = selected.shape
    r, c = int(window.row_off), int(window.col_off)
    out = np.full((h, w), nodata, dtype=np.float32)
    selected_ids = np.unique(selected[selected >= 0])
    for k in selected_ids:
        y_key, f_key = combo_source_keys[int(k)]
        y = yields[y_key][r:r+h, c:c+w]
        f = fertilizers[f_key][r:r+h, c:c+w]
        use = (selected == k) & np.isfinite(y) & (y > 0) & np.isfinite(f) & (f > 0)
        if np.any(use):
            out[use] = (y[use] / f[use]).astype(np.float32)
    if decimals is not None:
        out = np.where(out == nodata, out, np.round(out, int(decimals))).astype(np.float32)
    return out


def nup_from_selected_combo(selected, combo_source_keys, combo_manure_keys,
                             yields, fertilizers, manures, equation,
                             window, nodata, decimals=None):
    """Return the Nup of the SAME combination used for median-selected PFPN.

    Reuse the original, preprocessed Yield/F/M arrays, Nup regression and
    minimum Nup threshold. This does not compute the weighted median of Nup.
    Pixels without a selected valid NUE combination remain NoData.
    """
    h, w = selected.shape
    r, c = int(window.row_off), int(window.col_off)
    out = np.full((h, w), nodata, dtype=np.float32)
    for k in np.unique(selected[selected >= 0]):
        k = int(k)
        y_key, f_key = combo_source_keys[k]
        m_key = combo_manure_keys[k]
        y = yields[y_key][r:r+h, c:c+w]
        f = fertilizers[f_key][r:r+h, c:c+w]
        m = manures[m_key][r:r+h, c:c+w]

        # Match the original NUE combination calculation exactly.
        nup_n_input = f + m if equation['include_manure'] else f
        nup = (equation['yield_coefficient'] * y
               + equation['fertilizer_manure_coefficient'] * nup_n_input
               + equation['intercept'])
        nup = np.where(np.isfinite(nup),
                       np.maximum(nup, equation['min_nup']), np.nan)
        use = (selected == k) & np.isfinite(nup)
        if np.any(use):
            out[use] = nup[use].astype(np.float32)
    if decimals is not None:
        out = np.where(out == nodata, out,
                       np.round(out, int(decimals))).astype(np.float32)
    return out


def yield_fertilizer_from_selected_combo(selected, combo_source_keys, yields,
                                         fertilizers, window, nodata, decimals=None):
    """Export Yield and synthetic fertilizer N from the same NUE-median-selected
    combination already used for PFPN/Nup.

    Yield must be positive. F must be finite, but zero/negative F values are
    retained here to reflect the selected source input exactly. PFPN's existing
    strict F > 0 condition is unchanged. Invalid or unselected cells are NoData.
    """
    h, w = selected.shape
    r, c = int(window.row_off), int(window.col_off)
    out_y = np.full((h, w), nodata, dtype=np.float32)
    out_f = np.full((h, w), nodata, dtype=np.float32)
    for k in np.unique(selected[selected >= 0]):
        y_key, f_key = combo_source_keys[int(k)]
        y = yields[y_key][r:r+h, c:c+w]
        f = fertilizers[f_key][r:r+h, c:c+w]
        use = ((selected == k) & np.isfinite(y) & (y > 0)
               & np.isfinite(f))
        if np.any(use):
            out_y[use] = y[use]
            out_f[use] = f[use]
    if decimals is not None:
        out_y = np.where(out_y == nodata, out_y,
                         np.round(out_y, int(decimals))).astype(np.float32)
        out_f = np.where(out_f == nodata, out_f,
                         np.round(out_f, int(decimals))).astype(np.float32)
    return out_y, out_f


def load_points(path):
    if not path:
        return []
    raw = pd.read_excel(path, header=None, engine='openpyxl')
    pts = []
    for row in raw.itertuples(index=False, name=None):
        try:
            seq, lon, lat, year = row[:4]
            pts.append(dict(seq=seq, lon=float(lon), lat=float(lat), year=int(year)))
        except (ValueError, TypeError, IndexError):
            continue
    return pts


def run(cfg):
    outdir = Path(cfg['output_dir'])
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_setting = cfg.get('tmp_dir')
    temp_created = tmp_setting is None
    tmp = Path(tempfile.mkdtemp(prefix='nue_m2_')) if temp_created else Path(tmp_setting)
    tmp.mkdir(parents=True, exist_ok=True)
    indexes = index_sources(cfg)
    points = load_points(cfg.get('points_xlsx'))
    point_rows, provenance = [], []
    nodata = float(cfg.get('out_nodata', -9999))
    decimals = cfg.get('output_decimals', 6)
    if decimals is not None and not (0 <= int(decimals) <= 8):
        raise ValueError('output_decimals must be 0..8 or null')
    bh = int(cfg.get('block_height', 512))
    bw = int(cfg.get('block_width', 512))
    eq = cfg['nup_equation']
    try:
        for year in range(int(cfg['years'][0]), int(cfg['years'][1])+1):
            avail = {var: [(s, indexes[var][s['id']][year])
                           for s in cfg['sources'][var]
                           if year in indexes[var][s['id']]]
                     for var in ('yield', 'F', 'M', 'D')}
            if not all(avail.values()):
                LOG.warning('%d skipped: missing all input datasets for at least one factor', year)
                continue
            combinations = list(itertools.product(*(avail[var] for var in ('yield','F','M','D'))))
            # Normalize within the *available* sources for each factor and year.
            # If missing years occur, weights redistribute only across available sources.
            factor_norm = {v: sum(s['probability'] for s, _ in avail[v]) for v in avail}
            weights = []
            paths = []
            # Preserve the original PFPN inputs; additionally retain the manure
            # sources needed to reconstruct selected-combination Nup by Tile.
            pf_yields = {}
            pf_fertilizers = {}
            nup_manures = {}
            combo_source_keys = []
            combo_manure_keys = []
            LOG.info('%d: %d combinations', year, len(combinations))
            for k, combo in enumerate(combinations):
                sy, sf, sm, sd = (item[0] for item in combo)
                weight = math.prod(s['probability']/factor_norm[var]
                                   for var, s in zip(('yield','F','M','D'), (sy,sf,sm,sd)))
                dy_path, f_path, m_path, d_path = (item[1] for item in combo)
                with rasterio.open(dy_path) as ref:
                    if ref.crs is None:
                        raise ValueError(f'Yield lacks CRS: {dy_path}')
                    y = ref.read(1, masked=True).astype('float32').filled(np.nan)
                    profile = ref.profile.copy()
                    f = fill_f_nearest(read_on_grid(f_path, ref))
                    m = np.nan_to_num(read_on_grid(m_path, ref), nan=0.0)
                    d = np.nan_to_num(read_on_grid(d_path, ref), nan=0.0)
                    # Nup's manure term is configurable; manure ALWAYS remains
                    # in the NUE denominator (unchanged original logic).
                    nup_n_input = f + m if eq['include_manure'] else f
                    nup = (eq['yield_coefficient'] * y
                           + eq['fertilizer_manure_coefficient'] * nup_n_input
                           + eq['intercept'])
                    # Apply the requested floor only for finite Nup values.
                    nup = np.where(np.isfinite(nup),
                                   np.maximum(nup, eq['min_nup']), np.nan)
                    denominator = f + m + d + float(cfg['SN']) + float(cfg['BNF'])
                    # Keep the exact input arrays required to recover PFPN for
                    # the NUE-median-selected combination later, window by window.
                    y_key, f_key = str(dy_path), str(f_path)
                    m_key = str(m_path)
                    if y_key not in pf_yields:
                        pf_yields[y_key] = y
                    if f_key not in pf_fertilizers:
                        pf_fertilizers[f_key] = f
                    if m_key not in nup_manures:
                        nup_manures[m_key] = m
                    with np.errstate(divide='ignore', invalid='ignore'):
                        nue = np.where((y > 0) & np.isfinite(y) & np.isfinite(nup)
                                       & np.isfinite(denominator) & (denominator != 0),
                                       nup / denominator, np.nan)
                    # NUE is a dimensionless fraction (multiply by 100 for percentage).
                    path = tmp / f'M2_{year}_combo_{k:04d}.tif'
                    p = profile.copy()
                    p.update(driver='GTiff', count=1, dtype='float32', nodata=nodata,
                             compress='LZW', BIGTIFF='IF_SAFER')
                    with rasterio.open(path, 'w', **p) as dst:
                        dst.write(np.where(np.isfinite(nue), nue, nodata).astype('float32'), 1)
                paths.append(path)
                weights.append(weight)
                combo_source_keys.append((y_key, f_key))
                combo_manure_keys.append(m_key)
                provenance.append({'year':year,'combo':k,'yield':sy['id'],'F':sf['id'],
                                   'M':sm['id'],'D':sd['id'],'weight':weight})
            if not paths:
                continue
            # Check all combinations share an identical grid, otherwise window
            # aggregation would silently mix unrelated pixels.
            with contextlib.ExitStack() as stack:
                srcs = [stack.enter_context(rasterio.open(p)) for p in paths]
                ref = srcs[0]
                for s in srcs[1:]:
                    if s.width != ref.width or s.height != ref.height or s.crs != ref.crs or s.transform != ref.transform:
                        raise ValueError(f'{year}: yield source grids differ; standardize grids before running M2')
                out_profile = ref.profile.copy()
                names = ('median','mean','min','max','dif','n','weight_sum',
                         'p2_5','p97_5','p95_diff')
                with contextlib.ExitStack() as outputs:
                    targets = {name: outputs.enter_context(rasterio.open(outdir / f'NUE_M2_{name}_{year}.tif', 'w', **out_profile))
                               for name in names}
                    # One additional output; existing NUE statistics remain unchanged.
                    pfpn_target = outputs.enter_context(
                        rasterio.open(outdir / f'PFPN_M2_median_{year}.tif', 'w', **out_profile))
                    # NEW: Nup corresponding to the same median-selected NUE combination.
                    nup_target = outputs.enter_context(
                        rasterio.open(outdir / f'Nup_M2_median_{year}.tif', 'w', **out_profile))
                    # Additional source-consistent Yield and fertilizer N maps.
                    yield_target = outputs.enter_context(
                        rasterio.open(outdir / f'Yield_M2_median_{year}.tif', 'w', **out_profile))
                    fertilizer_target = outputs.enter_context(
                        rasterio.open(outdir / f'Fertilizer_M2_median_{year}.tif', 'w', **out_profile))
                    for win in iter_windows(ref.height, ref.width, bh, bw):
                        a = np.stack([np.where((x := s.read(1, window=win)) == nodata, np.nan, x)
                                      for s in srcs], axis=0)
                        stats = weighted_stats(a, weights)
                        # The weighted median can lie between combo NUE values.
                        # Choose the closest available NUE; deterministic ties:
                        # highest source-combination weight, then first index.
                        selected = nearest_median_combo_indices(a, stats['median'], weights)
                        pfpn = pfpn_from_selected_combo(
                            selected, combo_source_keys, pf_yields, pf_fertilizers,
                            win, nodata, decimals)
                        pfpn_target.write(pfpn, 1, window=win)
                        # Reconstruct Nup only for the selected combinations.
                        nup_selected = nup_from_selected_combo(
                            selected, combo_source_keys, combo_manure_keys,
                            pf_yields, pf_fertilizers, nup_manures,
                            eq, win, nodata, decimals)
                        nup_target.write(nup_selected, 1, window=win)
                        # Read the exact corresponding source inputs, with no
                        # independent Yield or fertilizer median aggregation.
                        selected_yield, selected_fertilizer = yield_fertilizer_from_selected_combo(
                            selected, combo_source_keys, pf_yields, pf_fertilizers,
                            win, nodata, decimals)
                        yield_target.write(selected_yield, 1, window=win)
                        fertilizer_target.write(selected_fertilizer, 1, window=win)
                        for name, ds in targets.items():
                            value = np.where(np.isfinite(stats[name]), stats[name], nodata)
                            if decimals is not None and name not in ('n',):
                                value = np.round(value, int(decimals))
                            value = value.astype('float32')
                            ds.write(value, 1, window=win)
                for pt in (q for q in points if q['year'] == year):
                    # Excel locations are longitude/latitude in EPSG:4326.
                    px, py = transform_coords('EPSG:4326', ref.crs, [pt['lon']], [pt['lat']])
                    values = []
                    for s in srcs:
                        try:
                            val = next(s.sample([(px[0], py[0])], masked=True))[0]
                            values.append(np.nan if np.ma.is_masked(val) or val == nodata else float(val))
                        except (ValueError, IndexError):
                            values.append(np.nan)
                    valid = np.isfinite(values)
                    wv = np.asarray(weights)[valid]
                    point_rows.append({**pt, 'M2_weighted_median': float(weighted_stats(np.array(values, dtype='float32')[:,None,None], weights)['median'][0,0]),
                                       'n_valid':int(np.sum(valid)), 'valid_weight':float(np.sum(wv)),
                                       **{f'combo_{i:04d}': val for i, val in enumerate(values)}})
            for p in paths:
                p.unlink(missing_ok=True)
            LOG.info('%d completed', year)
        if provenance:
            pd.DataFrame(provenance).to_csv(outdir/'M2_combination_weights.csv', index=False)
        if point_rows:
            pd.DataFrame(point_rows).to_excel(outdir/'M2_points.xlsx', index=False)
        with open(outdir/'M2_run_metadata.json', 'w', encoding='utf-8') as f:
            json.dump({'mode': 'M2_weighted_median', 'years':cfg['years'], 'sources':cfg['sources'],
                       'SN':cfg['SN'],'BNF':cfg['BNF'], 'nup_equation':cfg['nup_equation'],
                       'pfpn': {'file_pattern':'PFPN_M2_median_{year}.tif',
                                'definition':'yield divided by synthetic fertilizer N from selected combination',
                                'selection':'closest finite NUE to weighted median; ties: higher combo weight then lowest combo index',
                                'zero_or_missing_fertilizer':'nodata'},
                       'nup_median_selected': {
                           'file_pattern':'Nup_M2_median_{year}.tif',
                           'unit':'kg N/ha',
                           'definition':'Nup of the same combination selected for PFPN',
                           'selection':'closest finite NUE to weighted median; ties: higher combo weight then lowest combo index'},
                       'selected_source_inputs': {
                           'yield_file_pattern': 'Yield_M2_median_{year}.tif',
                           'fertilizer_file_pattern': 'Fertilizer_M2_median_{year}.tif',
                           'yield_unit': 'kg/ha',
                           'fertilizer_unit': 'kg N/ha',
                           'selection': 'same NUE-weighted-median-selected combo as PFPN and Nup',
                           'fertilizer_zero_value': 'retained as 0 in F GeoTIFF; PFPN remains nodata'},
                       'percentiles': {'method':'weighted_linear_plotting_positions',
                                       'lower': 0.025, 'upper': 0.975}}, f, indent=2, ensure_ascii=False)
    finally:
        if temp_created and cfg.get('delete_tmp', True):
            shutil.rmtree(tmp, ignore_errors=True)
    LOG.info('All done. Output: %s', outdir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Path to YAML config')
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    run(read_config(args.config))


if __name__ == '__main__':
    main()
