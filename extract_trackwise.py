#!/usr/bin/env python3
"""Track-wise extraction, detached-run version of
geos5data_0.0625deg_30mn_trackwise.ipynb.

Block-structured so memory is FLAT: ground tracks, nadir points and slabs are
built and discarded one 200-field block at a time. Resume = skip blocks whose
parquet part already exists. A part is written only when every task in the
block succeeded, so a resume always refetches an incomplete block instead of
freezing its gaps in place.  Launch:  nohup python3 extract_trackwise.py &
"""
import warnings; warnings.filterwarnings("ignore")
import os, sys, gc, time as _time
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd, shapely
from joblib import Parallel, delayed

# ---- configuration (mirrors the notebook) --------------------------------
# ---- WEATHER WINDOW ------------------------------------------------------
# 8 months starting at the wettest point of the G5NR record. June-July 2005 is
# the wettest consecutive 2-month window (area-weighted global mean prectot from
# tavg01mo_2d_met3_Cx: 3.204 mm/day, highest of all 24 months), so a June start
# opens on peak monsoon/ITCZ and then sweeps through NH autumn into winter --
# giving real seasonal spread rather than one regime.
# G5NR 30-min axis starts 2005-05-15 21:30. The window must start at the same
# UTC time-of-day as the orbit (startdate 15:04 UTC): decision i reads weather
# step FIELD_START + i//30, so any time-of-day offset here shifts the local
# solar time of every weather sample. Starting at 00:00 (step 773) put the
# ~14:00 descending pass on ~22:40 weather and the ~02:00 pass on ~10:35.
FIELD_START = 803       # == 2005-06-01 15:00, 4 min before orbit time-of-day
N_FIELDS    = 11760     # 245 days -> 2006-02-01 15:00 (exclusive)

DECISION  = timedelta(minutes=1)
SUB_STEP  = timedelta(seconds=1)
G5_STEP   = timedelta(minutes=30)
# Sustained rates measured on this box with a WARM pool, 40 tasks each, all
# succeeding (retries absorb the 504s / connection resets NCCS throws):
#     6 workers -> 10.07 s/task   12 -> 6.73   18 -> 5.94
# A warm pool matters: 18 cold workers all opening DAP handles at once looks
# like an attack and stalls. joblib reuses the pool across a block, so only the
# first task of a block pays the open cost.
# 18 is ~12% faster than 12; drop to 12 if the log shows blocks failing outright
# (as opposed to individual retries, which are normal).
DEC_PER_FIELD, GROUP, WORKERS, CHECKPOINT = 30, 3, 18, 200
# A task whose 4 in-task retries all fail is re-queued at block level after a
# pause (60 s, 120 s, ... capped at 15 min; ~45 min total), which rides out NCCS
# outages and laptop sleeps. The 2005-06 run had no such pass and lost 1,161
# decisions in 49 whole-task gaps (one gap of 807 decisions).
RETRY_ROUNDS, RETRY_PAUSE = 6, 60
CACHE = "cache_8mo_jun2005_lst"
G5NR_URL = ("https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/"
            "0.0625_deg/inst/inst30mn_2d_met1_Nx")
MSI = ["tautot","tauhgh","taulow","cldtot","cldhgh","cldmid","cldlow","cldtmp","lwtup"]
LAB = ["prectot","preccon","iwp","lwp"]
VARS = MSI + LAB
startdate = datetime(2025,7,19,15,4,tzinfo=timezone.utc)
PER_FIELD = int(G5_STEP/DECISION)
TLE = ["1 59908U 24101A   25200.34125573  .00010433  00000+0  14571-3 0  9999",
       "2 59908  97.0168 326.4971 0001222 108.6708 251.4681 15.57041891 64775"]
os.makedirs(CACHE, exist_ok=True)

def log(m): print(f"[{datetime.now():%m-%d %H:%M:%S}] {m}", flush=True)

# libnetcdf blocks forever on a stalled DAP transfer unless HTTP.TIMEOUT is set.
# Set it here rather than relying on a file some other notebook created. Kept
# short: after a laptop sleep every handle is dead, and a long timeout means each
# doomed task burns 4 x TIMEOUT before the retry loop gives up.
_dodsrc = os.path.expanduser("~/.dodsrc")
_cur = open(_dodsrc).read() if os.path.exists(_dodsrc) else ""
if "HTTP.TIMEOUT=120" not in _cur:
    with open(_dodsrc, "w") as f:
        f.write("HTTP.TIMEOUT=120\nHTTP.CONNECTTIMEOUT=30\n")
    log("set ~/.dodsrc HTTP.TIMEOUT=120, CONNECTTIMEOUT=30")

# ---- satellite / orbit ----------------------------------------------------
from tatc import utils
from tatc.schemas import PointedInstrument, Satellite, TwoLineElements
from tatc.analysis import compute_ground_track
from tatc.constants import timescale
from skyfield.api import EarthSatellite, wgs84
SAT = Satellite(name="EarthCare", orbit=TwoLineElements(tle=TLE),
    instruments=[PointedInstrument(name="MSI",
        field_of_regard=utils.swath_width_to_field_of_regard(394e3,150e3)+2*5.760868,
        cross_track_field_of_view=utils.swath_width_to_field_of_view(394e3,150e3,5.760868),
        along_track_field_of_view=utils.swath_width_to_field_of_view(394e3,10e3,5.760868),
        roll_angle=5.760868, is_rectangular=True)])
SF = EarthSatellite(TLE[0], TLE[1], "EarthCare", timescale)
SUB_PER_DEC = int(DECISION/SUB_STEP)

# ---- worker-side DAP handle ----------------------------------------------
_WDS = None
def _wds(force=False):
    """One DAP handle per worker, reused; `force` discards a dead one.

    The reset MUST go through this function's own `global _WDS`. An earlier
    version did `globals()["_WDS"] = None` from inside process_field; under
    cloudpickle that resolves to a different namespace, so the reset silently
    did nothing — a worker whose handle died (laptop sleep) kept returning that
    dead handle forever and every later task timed out. 18 workers spent 10 h
    producing zero rows against a perfectly healthy server.
    """
    global _WDS
    if force and _WDS is not None:
        try:
            _WDS.close()
        except Exception:
            pass
        _WDS = None
    if _WDS is None:
        import xarray as xr
        _WDS = xr.open_dataset(G5NR_URL, decode_times=True)
    return _WDS

# np.isfinite() alone lets through a handful of G5NR cells that return a
# large-but-technically-finite fill value (up to ~3.4e38, float32's max) for
# missing data -- caught 640 rows in the 8mo dataset, e.g. a "cloud fraction"
# of 1.3e35. No MSI/label variable here is ever physically close to 1e6.
def _sane(a):
    return a[np.isfinite(a) & (np.abs(a) < 1e6)]
def _nanmax(a):
    a = _sane(a); return float(a.max()) if a.size else float("nan")
def _nanmean(a):
    a = _sane(a); return float(a.mean()) if a.size else float("nan")

def reduce_decision(geom, sub, nlat, nlon):
    la = sub["lat"].values; lo = sub["lon"].values
    if la.size == 0 or lo.size == 0: return None
    LO, LA = np.meshgrid(lo, la)
    inside = shapely.contains_xy(geom, LO, LA)
    if not inside.any():
        c = geom.centroid; inside = np.zeros_like(LO, dtype=bool)
        inside[np.abs(la-c.y).argmin(), np.abs(lo-c.x).argmin()] = True
    keep = ((nlat>=la.min())&(nlat<=la.max())&(nlon>=lo.min())&(nlon<=lo.max()))
    if not keep.any(): keep = np.ones_like(nlat, dtype=bool)
    ni = np.abs(la[:,None]-nlat[keep][None,:]).argmin(axis=0)
    nj = np.abs(lo[:,None]-nlon[keep][None,:]).argmin(axis=0)
    row = {f"{v}_mean": _nanmean(sub[v].values[inside]) for v in MSI}
    row["tautot_max"] = _nanmax(sub["tautot"].values[inside])
    for v in LAB:
        arr = sub[v].values
        row[f"{v}_swath"] = _nanmax(arr[inside]); row[f"{v}_nadir"] = _nanmax(arr[ni,nj])
    row["n_swath_cells"] = int(inside.sum()); row["n_nadir_cells"] = len(set(zip(ni,nj)))
    return row

def process_task(fi, geoms, decs, times, nlats, nlons):
    """One small bbox (<=GROUP decisions). `fi` is the RELATIVE field index;
    the absolute G5NR time index is FIELD_START + fi. Returns None when the
    download failed (retryable), a list of rows otherwise."""
    g5i = FIELD_START + fi
    import xarray as xr
    pad = 0.13
    bs = [g.bounds for g in geoms]
    lat0,lat1 = min(b[1] for b in bs), max(b[3] for b in bs)
    lon0,lon1 = min(b[0] for b in bs), max(b[2] for b in bs)
    sub = None
    for a in range(4):
        try:
            ds = _wds(force=(a > 0))    # retries always get a FRESH handle
            sel = dict(lat=slice(lat0-pad, lat1+pad))
            if lon1-lon0 > 180:
                w = ds[VARS].isel(time=g5i).sel(lon=slice(120,180), **sel).load()
                e = ds[VARS].isel(time=g5i).sel(lon=slice(-180,-120), **sel).load()
                sub = xr.concat([w,e], dim="lon")
            else:
                sub = ds[VARS].isel(time=g5i).sel(lon=slice(lon0-pad, lon1+pad), **sel).load()
            break
        except Exception:
            _time.sleep(min(4*2**a, 45))
    if sub is None: return None
    out = []
    for g,d,t,nla,nlo in zip(geoms, decs, times, nlats, nlons):
        r = reduce_decision(g, sub, nla, nlo)
        if r is None: continue
        r.update(decision=int(d), field=int(fi), g5_step=int(g5i), time=t,
                 lat=g.centroid.y, lon=g.centroid.x)
        out.append(r)
    return out

def block_tracks(f0, f1):
    """Ground-track footprints + nadir points for one block of fields."""
    decs = np.array([f*PER_FIELD+k for f in range(f0,f1)
                     for k in range(min(DEC_PER_FIELD, PER_FIELD))])
    gt = pd.concat(Parallel(n_jobs=-1)(
        delayed(compute_ground_track)(
            SAT, pd.date_range(startdate+int(i)*DECISION,
                               startdate+int(i)*DECISION+DECISION,
                               freq=SUB_STEP, inclusive="left"), crs="spice")
        for i in decs), ignore_index=True)
    gt["decision"] = decs; gt["field"] = decs // PER_FIELD
    off = np.arange(SUB_PER_DEC)*SUB_STEP.total_seconds()
    secs = (decs[:,None]*DECISION.total_seconds() + off[None,:]).ravel()
    ts = timescale.from_datetimes(pd.to_datetime(startdate)+pd.to_timedelta(secs,unit="s"))
    sp = wgs84.subpoint(SF.at(ts))
    nla = sp.latitude.degrees.astype("float32").reshape(len(decs), SUB_PER_DEC)
    nlo = sp.longitude.degrees.astype("float32").reshape(len(decs), SUB_PER_DEC)
    del sp, ts, secs
    return gt, nla, nlo, {int(d): i for i,d in enumerate(decs)}

def main():
    blocks = [(b, min(b+CHECKPOINT, N_FIELDS)) for b in range(0, N_FIELDS, CHECKPOINT)]
    todo = [(a,b) for a,b in blocks
            if not os.path.exists(f"{CACHE}/rows_{a//CHECKPOINT:05d}.parquet")]
    log(f"window: G5NR steps {FIELD_START}..{FIELD_START+N_FIELDS-1} "
        f"(2005-06-01 15:00 -> 2006-02-01 15:00, 8 months)")
    log(f"{N_FIELDS:,} fields x {DEC_PER_FIELD} decisions = "
        f"{N_FIELDS*DEC_PER_FIELD:,} rows | {N_FIELDS//CHECKPOINT+1} blocks | "
        f"{len(blocks)-len(todo)} cached | {len(todo)} to do | {WORKERS} workers")
    _rate = {6: 10.07, 12: 6.73, 18: 5.94}.get(WORKERS, 10.07)
    log(f"estimate: {N_FIELDS*DEC_PER_FIELD//GROUP:,} tasks x ~{_rate:.1f}s = "
        f"{N_FIELDS*DEC_PER_FIELD/GROUP*_rate/3600:.0f} h "
        f"({N_FIELDS*DEC_PER_FIELD/GROUP*_rate/3600/24:.1f} days)")
    t0 = _time.time(); prev_el = 0.0; first_bt = None
    for n,(f0,f1) in enumerate(todo, 1):
        try:
            gt, nla, nlo, row_of = block_tracks(f0, f1)
            args = []
            for fi, seg in gt.groupby("field"):
                seg = seg.sort_values("decision")
                for s in range(0, len(seg), GROUP):
                    ch = seg.iloc[s:s+GROUP]
                    ri = [row_of[int(d)] for d in ch.decision]
                    args.append((int(fi), list(ch.geometry), list(ch.decision),
                                 list(ch.time), [nla[i] for i in ri], [nlo[i] for i in ri]))
            res = Parallel(n_jobs=WORKERS, backend="loky")(
                delayed(process_task)(*x) for x in args)
            flat = [r for rs in res if rs for r in rs]
            failed = [x for x, rs in zip(args, res) if rs is None]
            for rnd in range(1, RETRY_ROUNDS+1):
                if not failed: break
                pause = min(RETRY_PAUSE*2**(rnd-1), 900)
                log(f"block {f0//CHECKPOINT}: {len(failed)} tasks failed, "
                    f"retry round {rnd}/{RETRY_ROUNDS} in {pause} s")
                _time.sleep(pause)
                res = Parallel(n_jobs=min(WORKERS, len(failed)), backend="loky")(
                    delayed(process_task)(*x) for x in failed)
                flat += [r for rs in res if rs for r in rs]
                failed = [x for x, rs in zip(failed, res) if rs is None]
            want = sum(len(x[2]) for x in args)
            if failed:
                log(f"block {f0//CHECKPOINT} INCOMPLETE: {len(failed)} tasks "
                    f"({sum(len(x[2]) for x in failed)} decisions) still failing "
                    f"-> part NOT saved, rerun to resume")
            else:
                if len(flat) != want:
                    log(f"block {f0//CHECKPOINT}: {want-len(flat)} decisions had "
                        f"an empty slab (not a download failure)")
                pd.DataFrame(flat).to_parquet(
                    f"{CACHE}/rows_{f0//CHECKPOINT:05d}.parquet", index=False)
            el = _time.time()-t0
            bt = el - prev_el; prev_el = el
            if first_bt is None: first_bt = bt
            warn = "  <-- SLOW: stale handles?" if bt > 3*first_bt else ""
            log(f"block {f0//CHECKPOINT} ({n}/{len(todo)}): {len(flat):,}/{want:,} rows | "
                f"{len(args):,} req | block {bt/60:.0f} min | eta "
                f"{el/n*(len(todo)-n)/3600:.1f} h{warn}")
            del gt, nla, nlo, args, res, flat, failed
        except Exception as e:
            log(f"block {f0//CHECKPOINT} FAILED: {type(e).__name__}: {e}")
        gc.collect()
    left = [a for a,b in blocks
            if not os.path.exists(f"{CACHE}/rows_{a//CHECKPOINT:05d}.parquet")]
    log(f"DONE in {(_time.time()-t0)/3600:.1f} h | "
        + (f"{len(left)} blocks incomplete {[a//CHECKPOINT for a in left]} "
           f"-> rerun to resume" if left else f"all {len(blocks)} blocks complete"))

if __name__ == "__main__":
    main()
