"""Per-mission benchmark configuration (docs/baseline_paper/03_dataset.md, 05_baselines.md).

Each mission is a separate benchmark: its own label column, comparison column, feature tiers and
blocked split. Only EarthCARE has a dataset so far; TRMM and GPM fail loudly until theirs exist.
"""
from dataclasses import dataclass, field

MM_PER_HR = 1 / 3600          # 1 mm/hr in kg m-2 s-1 (G5NR precipitation units)

GEOMETRY = ["lat", "sin_sh", "cos_sh", "sin_lon", "cos_lon", "land_frac_nadir", "ocean_frac_nadir"]


@dataclass(frozen=True)
class Mission:
    name: str
    label_col: str                 # precipitation in the radar's own footprint
    swath_col: str                 # precipitation over the passive swath (label-geometry comparison)
    convective_col: str
    tiers: dict = field(default_factory=dict)
    # time-blocked split by G5NR weather time (03_dataset.md §3.3, 07_experiments.md E1)
    train_months: tuple = ("2005-06", "2005-07", "2005-08", "2005-09")
    val_months: tuple = ("2005-10",)
    test_months: tuple = ("2005-11", "2005-12", "2006-01", "2006-02")
    threshold: float = MM_PER_HR


# `solar_hour` is cyclic, so every tier uses its sin/cos encoding (sin_sh, cos_sh) instead.
EARTHCARE = Mission(
    name="earthcare",
    label_col="prectot_nadir",
    swath_col="prectot_swath",
    convective_col="preccon_nadir",
    tiers={
        "A": ["cldtot_mean", "cldtmp_mean", "tautot_mean", "tautot_max", "lwtup_mean",
              "lwtup_fore"] + GEOMETRY,
        "A_notau": ["cldtot_mean", "cldtmp_mean", "lwtup_mean", "lwtup_fore"] + GEOMETRY,
        "B": ["cldtmp_mean", "lwtup_mean", "lwtup_fore"] + GEOMETRY,
        "C": GEOMETRY,
    },
)


def get_mission(name):
    if name == "earthcare":
        return EARTHCARE
    if name in ("trmm", "gpm"):
        raise NotImplementedError(
            f"{name}: no dataset yet. Define its label column (radar-swath maximum), swath column "
            "(passive swath), and feature tiers here once the extraction exists"
            + ("; GPM also needs decisions D2 (data source) and D3 (added-value label)."
               if name == "gpm" else "."))
    raise ValueError(f"unknown mission {name!r}")
