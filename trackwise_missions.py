"""Per-mission settings for the track-wise extraction, enrichment and exports.

Each mission is one benchmark with its own orbit, instruments, G5NR variables and outputs; the
method (extract_trackwise.py, task_outputs/enrich_trackwise.py) is shared. Benchmark settings
(labels, feature tiers, split) live in baselines/missions.py.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Instrument:
    name: str
    swath_m: float                 # cross-track swath width on the ground
    roll_deg: float = 0.0          # off-nadir pointing of the swath centre


@dataclass(frozen=True)
class TrackwiseMission:
    name: str                      # CLI name: earthcare, trmm, fy3g
    title: str                     # display name: EarthCARE, TRMM, FY-3G
    file_prefix: str               # EarthCare, TRMM, FY3G (data and notebook file names)
    norad: str
    tle_file: Path
    startdate: datetime            # orbit time of decision 0; same UTC time of day and day of
                                   # year as G5NR step FIELD_START (2005-06-01 15:00)
    max_tle_gap: timedelta         # largest allowed gap between TLE epochs in the window
    altitude_m: float              # used only to convert swath widths into instrument angles
    passive: Instrument            # passive sensor: features are means over its swath
    radar: Instrument | None       # scanning radar swath; None = nadir-only radar
    radar_name: str                # the active sensor being gated (CPR, PR, PMR)
    features: tuple                # G5NR variables averaged over the passive swath -> *_mean
    labels: tuple                  # G5NR variables taken as maxima -> *_swath, *_nadir, *_radar
    cache: str                     # extraction cache folder (relative to the working directory)
    sat_name: str

    @property
    def variables(self):
        return self.features + self.labels

    @property
    def folder(self):
        return ROOT / "missions" / self.name

    @property
    def registry(self):
        return self.folder / "column_registry.yaml"

    @property
    def data_path(self):
        return self.folder / "data" / f"{self.file_prefix}_trackwise_decisions.parquet"

    @property
    def geometries(self):
        """Label geometries, primary (the radar's own footprint) first."""
        return ("radar", "swath", "nadir") if self.radar else ("nadir", "swath")


EARTHCARE = TrackwiseMission(
    name="earthcare", title="EarthCARE", file_prefix="EarthCare", norad="59908",
    tle_file=ROOT / "missions" / "earthcare" / "tle" / "sat000059908.txt",
    # orbit 2025, weather 2005: same day of year and UTC time; neither span has a 29 February
    startdate=datetime(2025, 6, 1, 15, 0, tzinfo=timezone.utc),
    max_tle_gap=timedelta(days=3),
    altitude_m=394e3,
    passive=Instrument("MSI", 150e3, roll_deg=5.760868),
    radar=None,                    # CPR: nadir only
    radar_name="CPR",
    features=("tautot", "tauhgh", "taulow", "cldtot", "cldhgh", "cldmid", "cldlow", "cldtmp",
              "lwtup"),
    labels=("prectot", "preccon", "iwp", "lwp"),
    cache="cache_8mo_jun2005_season",
    sat_name="EarthCare",
)

TRMM = TrackwiseMission(
    name="trmm", title="TRMM", file_prefix="TRMM", norad="25063",
    tle_file=ROOT / "missions" / "trmm" / "tle" / "sat000025063.txt",
    # TRMM flew during G5NR's own year, so the orbit and the weather share the calendar date
    startdate=datetime(2005, 6, 1, 15, 0, tzinfo=timezone.utc),
    # the set has one 3 d 0 h 23 min gap (2005-10-07 -> 10); G5NR is a free-running simulation,
    # so a few km of orbit error there shifts which simulated weather is sampled, not its statistics
    max_tle_gap=timedelta(days=3, hours=12),
    altitude_m=400e3,              # TLE set: 398.5-400.6 km over the window
    # post-boost (Aug 2001 onward) swaths, NASA GES DISC: VIRS 833 km, PR 247 km, both centred
    passive=Instrument("VIRS", 833e3),
    radar=Instrument("PR", 247e3),
    radar_name="PR",
    # no cloud-layer diagnostics (PR does not reveal cloud layers) and no iwp/lwp (13.8 GHz PR does
    # not measure cloud water); lwtup stands in for the VIRS thermal-IR window (no CERES after 2000)
    features=("tautot", "cldtot", "cldtmp", "lwtup"),
    labels=("prectot", "preccon"),
    cache="cache_trmm_8mo_jun2005",
    sat_name="TRMM",
)

FY3G = TrackwiseMission(
    name="fy3g", title="FY-3G", file_prefix="FY3G", norad="56232",
    tle_file=ROOT / "missions" / "fy3g" / "tle" / "sat000056232.txt",
    # orbit 2025, weather 2005, aligned as for EarthCARE: same day of year and UTC time
    startdate=datetime(2025, 6, 1, 15, 0, tzinfo=timezone.utc),
    # the set has a 7 d 23 h 37 min gap (2025-09-16 -> 24) and a 3 d 13 h one (2025-08-05 -> 08),
    # each spanning an orbit raise (403 -> 410 km), so the closest-TLE switch at the gap midpoint
    # moves the track along-orbit; as for TRMM's gap, this shifts which simulated weather is
    # sampled, not its statistics (inclination and local-time drift are unchanged)
    max_tle_gap=timedelta(days=8),
    altitude_m=407e3,              # nominal; TLE set: 403-411 km over the window
    # Yang et al. (2025, AMT): PMR swath 303 km; CMA (IPWG-11, 2024): MERSI-RM swath 1000 km
    passive=Instrument("MERSI-RM", 1000e3),
    radar=Instrument("PMR", 303e3),
    radar_name="PMR",
    # same variables as TRMM: MERSI-RM is a VIS/IR imager like VIRS (no cloud-layer diagnostics,
    # no iwp/lwp), and FY-3G carries no broadband radiometer, so lwtup stands in for the 10.8/12 um
    # window; the MWRI-RM microwave imager is left out, as TMI was for TRMM
    features=("tautot", "cldtot", "cldtmp", "lwtup"),
    labels=("prectot", "preccon"),
    cache="cache_fy3g_8mo_jun2005_season",
    sat_name="FY-3G",
)

MISSIONS = {m.name: m for m in (EARTHCARE, TRMM, FY3G)}


def get(name):
    if name not in MISSIONS:
        raise ValueError(f"unknown mission {name!r}; choose from {sorted(MISSIONS)}")
    return MISSIONS[name]
