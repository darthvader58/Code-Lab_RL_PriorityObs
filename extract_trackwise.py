#!/usr/bin/env python3
"""Track-wise extraction for one mission (settings in trackwise_missions.py).

Block-structured so memory is FLAT: ground tracks, nadir points and slabs are
built and discarded one 200-field block at a time. Resume = skip blocks whose
parquet part already exists. A part is written only when every task in the
block succeeded, so a resume always refetches an incomplete block instead of
freezing its gaps in place.
Launch (the cache folder is created in the working directory):
    nohup python3 -u extract_trackwise.py --mission trmm > trmm.log &
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, os, sys, gc, time as _time
from functools import lru_cache
from datetime import datetime, timezone, timedelta
import numpy as np, pandas as pd, shapely
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trackwise_missions  # noqa: E402

# ---- configuration (mirrors the notebook) --------------------------------
# ---- WEATHER WINDOW ------------------------------------------------------
# 8 months starting at the wettest point of the G5NR record. June-July 2005 is
# the wettest consecutive 2-month window (area-weighted global mean prectot from
# tavg01mo_2d_met3_Cx: 3.204 mm/day, highest of all 24 months), so a June start
# opens on peak monsoon/ITCZ and then sweeps through NH autumn into winter --
# giving real seasonal spread rather than one regime.
# G5NR 30-min axis starts 2005-05-15 21:30. Decision i reads weather step
# FIELD_START + i//30, so FIELD_START must sit at the SAME UTC time-of-day as
# startdate (15:00 UTC, below): then every decision reads the latest snapshot
# at or before the satellite's time (offset 0-29 min). Starting at 00:00 (step
# 773) put the ~14:00 pass on ~22:40 weather; a 15:04 orbit start against a
# 15:00 snapshot left 4 of every 30 decisions on a needlessly old snapshot.
FIELD_START = 803       # == 2005-06-01 15:00, same time-of-day as startdate
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
# 18 is ~12% faster than 12, but on this 10-core machine the footprint step
# (one worker per core) shrank the shared loky pool from 18 to 10 every block;
# workers shut down with an open DAP handle sometimes segfault in libnetcdf,
# which failed the next call. Every Parallel call now uses the same WORKERS
# count, so the pool is never resized. (A fixed pool of 10 avoided the crashes
# but took ~64 min per block versus 15-33 min with 18.)
DEC_PER_FIELD, GROUP, WORKERS, CHECKPOINT = 30, 3, 18, 200
# A task whose 4 in-task retries all fail is re-queued at block level after a
# pause (60 s, 120 s, ... capped at 15 min; ~45 min total), which rides out NCCS
# outages and laptop sleeps. The 2005-06 run had no such pass and lost 1,161
# decisions in 49 whole-task gaps (one gap of 807 decisions).
RETRY_ROUNDS, RETRY_PAUSE = 6, 60
# A task normally takes ~15 s; in-task retries can stretch that to ~10 min.
# Past TASK_TIMEOUT the pool is killed and the whole call counts as failed:
# after a network change (VPN drop) libcurl sits on a dead socket forever and
# ignores HTTP.TIMEOUT, which once froze a block for over an hour.
TASK_TIMEOUT = 900
# loky workers occasionally die between calls (no crash report); the next call
# on the reused pool then raises TerminatedWorkerError at once, which skipped 6
# of the first 30 blocks. block_tracks is retried on that, and a block that is
# still not saved is re-queued at the end of the run, up to BLOCK_PASSES times.
TRACK_ATTEMPTS, BLOCK_PASSES = 3, 10
G5NR_URL = ("https://opendap.nccs.nasa.gov/dods/OSSE/G5NR/Ganymed/7km/"
            "0.0625_deg/inst/inst30mn_2d_met1_Nx")
# The orbit window must match the G5NR window in UTC time of day AND day of
# year (season): each mission's startdate (trackwise_missions.py) sits at 15:00
# UTC on 1 June, paired with G5NR 2005-06-01 15:00 (step 803).
PER_FIELD = int(G5_STEP/DECISION)
# Each mission's orbit comes from historical element sets spanning the window
# (missions/<mission>/tle/sat000<NORAD>.txt, CelesTrak special data request). A
# set of TLEs (not one) captures the orbit maintenance flown during the window.
# TATC and the nadir propagation below both use the TLE whose epoch is closest
# to each time.

class GeometryError(RuntimeError):
    """A footprint or nadir track that does not fit its weather slab: a bug in the
    geometry, never a download problem, so it stops the run instead of being retried."""

def _is_geometry_error(e):
    # by name: an exception raised in a loky worker comes back re-pickled, and a class
    # defined in the __main__ script need not be the same object afterwards
    return type(e).__name__ == "GeometryError"

def load_tles(path, norad):
    """Element sets for satellite `norad` in `path` -> chronological flat
    [line1, line2, ...] list. Name lines and duplicate epochs are dropped."""
    from sgp4.api import Satrec
    from sgp4.conveniences import sat_epoch_datetime
    lines = [ln.rstrip() for ln in open(path) if ln.strip()]
    sets = {}
    for a, b in zip(lines, lines[1:]):
        if a.startswith("1 ") and b.startswith("2 ") and a[2:7] == b[2:7] == norad:
            sets[sat_epoch_datetime(Satrec.twoline2rv(a, b))] = (a, b)
    return [ln for ep in sorted(sets) for ln in sets[ep]]

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
from tatc import config, utils
# Footprints (TATC) and nadir points (Skyfield) must come from the same
# propagation. TATC reuses a detected repeat cycle by default; this TLE has none
# (checked: identical to Skyfield), but pin it off so a future TLE cannot split
# the two trajectories silently.
config.rc.repeat_cycle_for_orbit_track = False
from tatc.schemas import PointedInstrument, Satellite, TwoLineElements
from tatc.analysis import compute_ground_track
from tatc.constants import timescale
from skyfield.api import EarthSatellite, wgs84

def _instrument(inst, altitude_m):
    """TATC instrument for a rectangular cross-track swath `inst.swath_m` wide,
    centred `inst.roll_deg` off nadir (along-track extent 10 km per sample)."""
    return PointedInstrument(name=inst.name,
        field_of_regard=utils.swath_width_to_field_of_regard(altitude_m, inst.swath_m)
                        + 2*inst.roll_deg,
        cross_track_field_of_view=utils.swath_width_to_field_of_view(
            altitude_m, inst.swath_m, inst.roll_deg),
        along_track_field_of_view=utils.swath_width_to_field_of_view(
            altitude_m, 10e3, inst.roll_deg),
        roll_angle=inst.roll_deg, is_rectangular=True)

class Orbit:
    """One mission's TLE set, TATC satellite (instrument 0 = passive sensor,
    instrument 1 = scanning radar if any) and Skyfield propagators."""
    def __init__(self, mission):
        self.mission = mission
        self.tle = load_tles(mission.tle_file, mission.norad)
        inst = [_instrument(mission.passive, mission.altitude_m)]
        if mission.radar:
            inst.append(_instrument(mission.radar, mission.altitude_m))
        self.sat = Satellite(name=mission.sat_name, orbit=TwoLineElements(tle=self.tle),
                             instruments=inst)
        self.sfs = [EarthSatellite(self.tle[2*k], self.tle[2*k+1], mission.sat_name, timescale)
                    for k in range(len(self.tle)//2)]

    def check_tle_coverage(self, end):
        """Fail early unless the TLE epochs bracket [startdate, end] with no gap
        longer than the mission's max_tle_gap."""
        start, limit = self.mission.startdate, self.mission.max_tle_gap
        ep = [sf.epoch.utc_datetime() for sf in self.sfs]
        gaps = np.diff(ep) if len(ep) > 1 else np.array([timedelta(0)])
        assert ep[0] <= start and ep[-1] >= end, \
            f"TLEs span {ep[0]:%Y-%m-%d} .. {ep[-1]:%Y-%m-%d}, need {start:%Y-%m-%d} .. {end:%Y-%m-%d}"
        assert max(gaps) <= limit, f"largest TLE gap {max(gaps)} > {limit}"
        return len(ep), ep[0], ep[-1], max(gaps)

    def nadir(self, times):
        """Sub-satellite lat/lon at `times` (list of datetimes), each time propagated
        from the same closest-epoch TLE that TATC picks for the footprints."""
        idx = np.asarray(self.sat.orbit.get_closest_tle_index(list(times)))
        lat = np.empty(len(times)); lon = np.empty(len(times))
        times = np.asarray(times, dtype=object)
        for k in np.unique(idx):
            m = idx == k
            sp = wgs84.subpoint(self.sfs[k].at(timescale.from_datetimes(list(times[m]))))
            lat[m] = sp.latitude.degrees; lon[m] = sp.longitude.degrees
        return lat, lon

@lru_cache(maxsize=None)
def orbit(name):
    """The Orbit of mission `name` (built once per process)."""
    return Orbit(trackwise_missions.get(name))
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

def _cells_inside(geom, LO, LA, what):
    inside = shapely.contains_xy(geom, LO, LA)
    if not inside.any():
        raise GeometryError(f"no grid-cell centre inside the {what} footprint "
                            f"(centroid {geom.centroid.y:.2f}, {geom.centroid.x:.2f})")
    return inside

def reduce_decision(geom, rgeom, sub, nlat, nlon, features, labels):
    """One decision -> one row. Passive features are means over the passive
    footprint `geom`; each label variable is a maximum over that footprint
    (`_swath`), over the cells under the 60 one-second nadir points (`_nadir`)
    and, for a scanning radar (`rgeom`), over the radar swath (`_radar`, with
    its mean as `_radar_mean`)."""
    la = sub["lat"].values; lo = sub["lon"].values
    LO, LA = np.meshgrid(lo, la)
    inside = _cells_inside(geom, LO, LA, "passive")
    keep = ((nlat>=la.min())&(nlat<=la.max())&(nlon>=lo.min())&(nlon<=lo.max()))
    if not keep.all():
        raise GeometryError(f"{int((~keep).sum())} nadir points outside the weather slab")
    ni = np.abs(la[:,None]-nlat[None,:]).argmin(axis=0)
    nj = np.abs(lo[:,None]-nlon[None,:]).argmin(axis=0)
    row = {f"{v}_mean": _nanmean(sub[v].values[inside]) for v in features}
    row["tautot_max"] = _nanmax(sub["tautot"].values[inside])
    for v in labels:
        arr = sub[v].values
        row[f"{v}_swath"] = _nanmax(arr[inside]); row[f"{v}_nadir"] = _nanmax(arr[ni,nj])
    row["n_swath_cells"] = int(inside.sum()); row["n_nadir_cells"] = len(set(zip(ni,nj)))
    if rgeom is not None:
        rin = _cells_inside(rgeom, LO, LA, "radar")
        for v in labels:
            arr = sub[v].values
            row[f"{v}_radar"] = _nanmax(arr[rin]); row[f"{v}_radar_mean"] = _nanmean(arr[rin])
        row["n_radar_cells"] = int(rin.sum())
    return row

def _lon_windows(geoms, pad):
    """Longitude slices covering `geoms`. A task that crosses +/-180 deg gets two
    tight slices, one each side of the dateline, from its actual extents."""
    bs = [g.bounds for g in geoms]
    lon0, lon1 = min(b[0] for b in bs), max(b[2] for b in bs)
    if lon1 - lon0 <= 180:
        return [slice(lon0-pad, lon1+pad)]
    lons = np.concatenate([np.asarray(p.exterior.coords)[:, 0] for g in geoms
                           for p in getattr(g, "geoms", [g])])
    west, east = lons[lons > 0].min(), lons[lons < 0].max()
    return [slice(west-pad, 180), slice(-180, east+pad)]

def process_task(fi, geoms, rgeoms, decs, times, nlats, nlons, features, labels):
    """One small bbox (<=GROUP decisions). `fi` is the RELATIVE field index;
    the absolute G5NR time index is FIELD_START + fi. `rgeoms` are the radar
    footprints (None entries for a nadir-only radar). Returns None when the
    download failed (retryable), a list of rows otherwise."""
    g5i = FIELD_START + fi
    import xarray as xr
    pad = 0.13
    shapes = list(geoms) + [g for g in rgeoms if g is not None]
    bs = [g.bounds for g in shapes]
    lat0,lat1 = min(b[1] for b in bs), max(b[3] for b in bs)
    windows = _lon_windows(shapes, pad)
    variables = list(features) + list(labels)
    sub = None
    for a in range(4):
        try:
            ds = _wds(force=(a > 0))    # retries always get a FRESH handle
            sel = dict(lat=slice(lat0-pad, lat1+pad))
            parts = [ds[variables].isel(time=g5i).sel(lon=w, **sel).load() for w in windows]
            sub = parts[0] if len(parts) == 1 else xr.concat(parts, dim="lon")
            # NCCS occasionally answers with a zero-size slab instead of an
            # error; the padded bbox always spans grid cells, so treat it as a
            # failed download (it once left a 3-decision gap).
            if 0 in (sub.sizes["lat"], sub.sizes["lon"]):
                raise ValueError("empty slab")
            break
        except Exception:
            sub = None
            _time.sleep(min(4*2**a, 45))
    if sub is None: return None
    out = []
    for g,rg,d,t,nla,nlo in zip(geoms, rgeoms, decs, times, nlats, nlons):
        r = reduce_decision(g, rg, sub, nla, nlo, features, labels)
        r.update(decision=int(d), field=int(fi), g5_step=int(g5i), time=t,
                 lat=g.centroid.y, lon=g.centroid.x)
        out.append(r)
    return out

def _footprints(sat, decs, start, instrument_index):
    """TATC footprint of each one-minute decision, sampled every second."""
    return pd.concat(Parallel(n_jobs=WORKERS, backend="loky")(
        delayed(compute_ground_track)(
            sat, pd.date_range(start+int(i)*DECISION, start+int(i)*DECISION+DECISION,
                               freq=SUB_STEP, inclusive="left"),
            instrument_index=instrument_index, crs="spice")
        for i in decs), ignore_index=True)

def block_tracks(orb, f0, f1):
    """Ground-track footprints (passive and, if any, radar) + nadir points for
    one block of fields."""
    start = orb.mission.startdate
    decs = np.array([f*PER_FIELD+k for f in range(f0,f1)
                     for k in range(min(DEC_PER_FIELD, PER_FIELD))])
    gt = _footprints(orb.sat, decs, start, 0)
    gt["radar_geometry"] = (list(_footprints(orb.sat, decs, start, 1).geometry)
                            if orb.mission.radar else [None]*len(gt))
    gt["decision"] = decs; gt["field"] = decs // PER_FIELD
    off = np.arange(SUB_PER_DEC)*SUB_STEP.total_seconds()
    secs = (decs[:,None]*DECISION.total_seconds() + off[None,:]).ravel()
    times = (pd.to_datetime(start)+pd.to_timedelta(secs,unit="s")).to_pydatetime()
    lat, lon = orb.nadir(times)
    nla = lat.astype("float32").reshape(len(decs), SUB_PER_DEC)
    nlo = lon.astype("float32").reshape(len(decs), SUB_PER_DEC)
    del lat, lon, times, secs
    return gt, nla, nlo, {int(d): i for i,d in enumerate(decs)}

def run_tasks(tasks):
    """process_task over `tasks` -> results aligned with `tasks`, None = failed.

    Results are collected as they arrive, so if the pool aborts (a stalled task
    hits TASK_TIMEOUT, or a worker segfaults) the tasks already finished are
    kept and only the rest count as failed."""
    out = [None]*len(tasks)
    try:
        gen = Parallel(n_jobs=WORKERS, backend="loky", timeout=TASK_TIMEOUT,
                       return_as="generator")(delayed(process_task)(*x) for x in tasks)
        for i, r in enumerate(gen):
            out[i] = r
    except Exception as e:      # stalled task (TimeoutError) or a killed worker
        if _is_geometry_error(e):
            raise
        kept = sum(r is not None for r in out)
        log(f"pool aborted ({type(e).__name__}): kept {kept}, "
            f"{len(tasks)-kept} tasks count as failed")
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mission", required=True, choices=sorted(trackwise_missions.MISSIONS))
    orb = orbit(ap.parse_args().mission)
    m = orb.mission
    CACHE = m.cache
    os.makedirs(CACHE, exist_ok=True)
    blocks = [(b, min(b+CHECKPOINT, N_FIELDS)) for b in range(0, N_FIELDS, CHECKPOINT)]
    n_tle, ep0, ep1, gap = orb.check_tle_coverage(m.startdate + N_FIELDS*G5_STEP)
    log(f"mission: {m.title} (NORAD {m.norad}) | passive {m.passive.name} "
        f"{m.passive.swath_m/1e3:.0f} km | radar "
        + (f"{m.radar.name} {m.radar.swath_m/1e3:.0f} km swath" if m.radar else "nadir only")
        + f" | {len(m.variables)} G5NR variables | cache {os.path.abspath(CACHE)}")
    log(f"orbit: {n_tle} TLEs, epochs {ep0:%Y-%m-%d %H:%M} .. {ep1:%Y-%m-%d %H:%M}, "
        f"largest gap {gap}")
    todo = [(a,b) for a,b in blocks
            if not os.path.exists(f"{CACHE}/rows_{a//CHECKPOINT:05d}.parquet")]
    log(f"window: G5NR steps {FIELD_START}..{FIELD_START+N_FIELDS-1} "
        f"(2005-06-01 15:00 -> 2006-02-01 15:00, 8 months); orbit from "
        f"{m.startdate:%Y-%m-%d %H:%M} UTC")
    log(f"{N_FIELDS:,} fields x {DEC_PER_FIELD} decisions = "
        f"{N_FIELDS*DEC_PER_FIELD:,} rows | {N_FIELDS//CHECKPOINT+1} blocks | "
        f"{len(blocks)-len(todo)} cached | {len(todo)} to do | {WORKERS} workers")
    _rate = {6: 10.07, 12: 6.73, 18: 5.94}.get(WORKERS, 10.07)
    log(f"estimate: {N_FIELDS*DEC_PER_FIELD//GROUP:,} tasks x ~{_rate:.1f}s = "
        f"{N_FIELDS*DEC_PER_FIELD/GROUP*_rate/3600:.0f} h "
        f"({N_FIELDS*DEC_PER_FIELD/GROUP*_rate/3600/24:.1f} days)")
    t0 = _time.time(); prev_el = 0.0; first_bt = None; tries = {}
    for n,(f0,f1) in enumerate(todo, 1):     # todo grows when a block is re-queued
        tries[f0] = tries.get(f0, 0) + 1
        try:
            for att in range(1, TRACK_ATTEMPTS+1):
                try:
                    gt, nla, nlo, row_of = block_tracks(orb, f0, f1); break
                except Exception as e:
                    if att == TRACK_ATTEMPTS: raise
                    log(f"block {f0//CHECKPOINT}: footprints failed ({type(e).__name__}), "
                        f"attempt {att}/{TRACK_ATTEMPTS}, retrying in 10 s")
                    _time.sleep(10)
            args = []
            for fi, seg in gt.groupby("field"):
                seg = seg.sort_values("decision")
                for s in range(0, len(seg), GROUP):
                    ch = seg.iloc[s:s+GROUP]
                    ri = [row_of[int(d)] for d in ch.decision]
                    args.append((int(fi), list(ch.geometry), list(ch.radar_geometry),
                                 list(ch.decision), list(ch.time),
                                 [nla[i] for i in ri], [nlo[i] for i in ri],
                                 m.features, m.labels))
            res = run_tasks(args)
            flat = [r for rs in res if rs for r in rs]
            failed = [x for x, rs in zip(args, res) if rs is None]
            for rnd in range(1, RETRY_ROUNDS+1):
                if not failed: break
                pause = min(RETRY_PAUSE*2**(rnd-1), 900)
                log(f"block {f0//CHECKPOINT}: {len(failed)} tasks failed, "
                    f"retry round {rnd}/{RETRY_ROUNDS} in {pause} s")
                _time.sleep(pause)
                res = run_tasks(failed)
                flat += [r for rs in res if rs for r in rs]
                failed = [x for x, rs in zip(failed, res) if rs is None]
            want = sum(len(x[3]) for x in args)
            if failed:
                log(f"block {f0//CHECKPOINT} INCOMPLETE: {len(failed)} tasks "
                    f"({sum(len(x[3]) for x in failed)} decisions) still failing "
                    f"-> part NOT saved, rerun to resume")
            else:
                assert len(flat) == want, f"{len(flat)} rows for {want} decisions"
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
            if _is_geometry_error(e):
                raise
            log(f"block {f0//CHECKPOINT} FAILED: {type(e).__name__}: {e}")
        if (not os.path.exists(f"{CACHE}/rows_{f0//CHECKPOINT:05d}.parquet")
                and tries[f0] < BLOCK_PASSES):
            todo.append((f0, f1))
            log(f"block {f0//CHECKPOINT}: re-queued (pass {tries[f0]+1}/{BLOCK_PASSES})")
        gc.collect()
    left = [a for a,b in blocks
            if not os.path.exists(f"{CACHE}/rows_{a//CHECKPOINT:05d}.parquet")]
    log(f"DONE in {(_time.time()-t0)/3600:.1f} h | "
        + (f"{len(left)} blocks incomplete {[a//CHECKPOINT for a in left]} "
           f"-> rerun to resume" if left else f"all {len(blocks)} blocks complete"))

if __name__ == "__main__":
    main()
