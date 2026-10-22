"""Per-mission benchmark configuration (docs/baseline_paper/03_dataset.md, 05_baselines.md).

Each mission is a separate benchmark: its own label column, comparison columns, feature tiers,
strata and blocked split. EarthCARE and TRMM are defined; GPM fails loudly until decisions D2 and
D3 are made.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MM_PER_HR = 1 / 3600          # 1 mm/hr in kg m-2 s-1 (G5NR precipitation units)

GEOMETRY = ["lat", "sin_sh", "cos_sh", "sin_lon", "cos_lon", "land_frac_nadir", "ocean_frac_nadir"]


@dataclass(frozen=True)
class Mission:
    name: str
    data_path: Path                # missions/<name>/data/<Prefix>_trackwise_decisions.parquet
    results_dir: Path              # missions/<name>/results/
    label_col: str                 # precipitation in the radar's own footprint
    swath_col: str                 # precipitation over the passive swath (label-geometry comparison)
    convective_col: str            # convective precipitation in the radar footprint
    mass_col: str                  # weight for mass-capture (rain the radar collects)
    # E4 label geometries per variable: (radar footprint, passive swath, *other comparisons)
    label_geometries: dict = field(default_factory=dict)
    # E6 latitude bands: (name, lower |lat|, upper |lat|), upper bound exclusive
    lat_strata: tuple = ()
    tiers: dict = field(default_factory=dict)
    # time-blocked split by G5NR weather time (03_dataset.md §3.3, 07_experiments.md E1)
    train_months: tuple = ("2005-06", "2005-07", "2005-08", "2005-09")
    val_months: tuple = ("2005-10",)
    test_months: tuple = ("2005-11", "2005-12", "2006-01", "2006-02")
    threshold: float = MM_PER_HR


# `solar_hour` is cyclic, so every tier uses its sin/cos encoding (sin_sh, cos_sh) instead.
EARTHCARE = Mission(
    name="earthcare",
    data_path=ROOT / "missions" / "earthcare" / "data" / "EarthCare_trackwise_decisions.parquet",
    results_dir=ROOT / "missions" / "earthcare" / "results",
    label_col="prectot_nadir",
    swath_col="prectot_swath",
    convective_col="preccon_nadir",
    mass_col="prectot_nadir",
    label_geometries={"prectot": ("prectot_nadir", "prectot_swath"),
                      "preccon": ("preccon_nadir", "preccon_swath")},
    lat_strata=(("tropics |lat|<23.5", 0, 23.5), ("subtropics 23.5-35", 23.5, 35),
                ("mid-latitudes 35-60", 35, 60), ("polar >60", 60, float("inf"))),
    tiers={
        "A": ["cldtot_mean", "cldtmp_mean", "tautot_mean", "tautot_max", "lwtup_mean",
              "lwtup_fore"] + GEOMETRY,
        "A_notau": ["cldtot_mean", "cldtmp_mean", "lwtup_mean", "lwtup_fore"] + GEOMETRY,
        "B": ["cldtmp_mean", "lwtup_mean", "lwtup_fore"] + GEOMETRY,
        "C": GEOMETRY,
    },
)

# TRMM: the label is precipitation anywhere in the 247 km PR swath; mass-capture weights by the
# PR-swath mean (rain area x rate). No forward-view feature: VIRS is a cross-track scanner and
# TRMM had no working broadband radiometer in 2005, so lwtup_mean stands in for the VIRS window.
TRMM = Mission(
    name="trmm",
    data_path=ROOT / "missions" / "trmm" / "data" / "TRMM_trackwise_decisions.parquet",
    results_dir=ROOT / "missions" / "trmm" / "results",
    label_col="prectot_radar",
    swath_col="prectot_swath",
    convective_col="preccon_radar",
    mass_col="prectot_radar_mean",
    label_geometries={"prectot": ("prectot_radar", "prectot_swath", "prectot_nadir"),
                      "preccon": ("preccon_radar", "preccon_swath", "preccon_nadir")},
    lat_strata=(("deep tropics |lat|<10", 0, 10), ("outer tropics 10-23.5", 10, 23.5),
                ("subtropics >=23.5", 23.5, float("inf"))),
    tiers={
        "A": ["cldtot_mean", "cldtmp_mean", "tautot_mean", "tautot_max", "lwtup_mean"] + GEOMETRY,
        "A_notau": ["cldtot_mean", "cldtmp_mean", "lwtup_mean"] + GEOMETRY,
        "B": ["cldtmp_mean", "lwtup_mean"] + GEOMETRY,
        "C": GEOMETRY,
    },
)


def get_mission(name):
    if name == "earthcare":
        return EARTHCARE
    if name == "trmm":
        return TRMM
    if name == "gpm":
        raise NotImplementedError(
            "gpm: no dataset yet. Decisions D2 (data source) and D3 (added-value label) come "
            "first; then define its label columns and feature tiers here.")
    raise ValueError(f"unknown mission {name!r}")
