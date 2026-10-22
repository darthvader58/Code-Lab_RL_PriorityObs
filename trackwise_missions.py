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
    name: str                      # CLI name: earthcare, trmm
    title: str                     # display name: EarthCARE, TRMM
    file_prefix: str               # EarthCare, TRMM (data and notebook file names)
    norad: str
    tle_file: Path
    startdate: datetime            # orbit time of decision 0; same UTC time of day and day of
                                   # year as G5NR step FIELD_START (2005-06-01 15:00)
    max_tle_gap: timedelta         # largest allowed gap between TLE epochs in the window
    altitude_m: float              # used only to convert swath widths into instrument angles
    passive: Instrument            # passive sensor: features are means over its swath
    radar: Instrument | None       # scanning radar swath; None = nadir-only radar
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
    # no cloud-layer diagnostics (PR does not reveal cloud layers) and no iwp/lwp (13.8 GHz PR does
    # not measure cloud water); lwtup stands in for the VIRS thermal-IR window (no CERES after 2000)
    features=("tautot", "cldtot", "cldtmp", "lwtup"),
    labels=("prectot", "preccon"),
    cache="cache_trmm_8mo_jun2005",
    sat_name="TRMM",
)

MISSIONS = {m.name: m for m in (EARTHCARE, TRMM)}


def get(name):
    if name == "gpm":
        raise NotImplementedError("gpm: decisions D2 (passive data source) and D3 (added-value "
                                  "label) come first; see docs/baseline_paper/14_submission_plan.md")
    if name not in MISSIONS:
        raise ValueError(f"unknown mission {name!r}; choose from {sorted(MISSIONS)}")
    return MISSIONS[name]
