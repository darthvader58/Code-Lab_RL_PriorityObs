"""Isolated seasonal runner, dispatched by run_baselines --protocol seasonal."""
import json
import os
import platform
import time
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .core import SEED, load, seasonal_folds
from .missions import get_mission
from .run_baselines import sha256
from .seasonal import PROTOCOL, THRESHOLD_SOURCE, aggregate, run_fold


def run(args):
    os.environ["LOKY_MAX_CPU_COUNT"] = str(args.threads)
    m = get_mission(args.mission)
    path = Path(args.data or m.data_path)
    out = Path(args.out or m.results_dir/"seasonal_lomo")
    if out.resolve() == m.results_dir.resolve():
        raise ValueError("seasonal results must not overwrite chronological results")
    if args.smoke and not args.out:
        raise ValueError("smoke tests require an explicit --out outside production results")
    if args.n_boot < 2 or args.threads < 1:
        raise ValueError("n_boot must be >=2 and threads >=1")
    log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {m.name}: {msg}",flush=True)
    df = load(path,m)
    full_rows = len(df)
    if args.smoke:
        # Include all temporal extrema so boundary definitions survive subsampling.
        boundary = df.groupby('g5_month').g5_step.transform('min').eq(df.g5_step) | df.groupby('g5_month').g5_step.transform('max').eq(df.g5_step)
        samples = df.groupby('g5_month',group_keys=False).sample(n=300,random_state=SEED)
        df = df.loc[df.index.isin(samples.index) | boundary].copy()
    folds = list(seasonal_folds(df))
    meta = {"protocol": PROTOCOL,"mission":m.name,"dataset_hash":sha256(path),"data":str(path),
            "seed":SEED,"rows":len(df),"full_dataset_rows":full_rows,"smoke":args.smoke,
            "threshold_source":THRESHOLD_SOURCE,"n_boot":args.n_boot,"threads":args.threads,
            "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
            "sklearn":sklearn.__version__,"lightgbm":lightgbm.__version__,
            "code_hashes":{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')
                           if p.name in ('core.py','experiments.py','seasonal.py','run_seasonal.py','missions.py')},
            "label":m.label_col,"label_threshold":m.threshold,"mass_column":m.mass_col,
            "models":"unchanged P1-P7; P8 oracle reference only",
            "preprocessing":"fit only on the seven training months, including MLP internal validation",
            "e1":"preserved chronological leakage comparison: ../e1_split.csv",
            "e9":"preserved architecture cost measurements: ../e9_policy_cost.csv",
            "folds":[{"held_out_month":f.held_out_month,"train_months":list(f.train_months),
                      "n_train_before_label_filter":len(f.train_index),"n_test_before_label_filter":len(f.test_index),
                      "n_buffer":len(f.dropped_index),"buffer_steps":list(f.buffer_steps)} for f in folds]}
    if (out/'run_meta.json').exists():
        old=json.loads((out/'run_meta.json').read_text())
        if not args.resume or any(old.get(k)!=v for k,v in meta.items()):
            raise ValueError(f"existing run at {out}; use a new --out or --resume with identical provenance")
    elif out.exists() and any(out.iterdir()):
        raise ValueError(f"refusing to mix with existing files in {out}")
    out.mkdir(parents=True,exist_ok=True)
    (out/'run_meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    with threadpool_limits(limits=args.threads):
        for fold in folds:
            folder=out/'folds'/fold.held_out_month
            if (folder/'complete.json').exists():
                hashes=json.loads((folder/'complete.json').read_text())
                if any(not (folder/f).exists() or sha256(folder/f)!=h for f,h in hashes.items()):
                    raise ValueError(f"completed fold files changed: {folder}")
                log(f"verified existing fold {fold.held_out_month}")
                continue
            folder.mkdir(parents=True,exist_ok=True)
            tables=run_fold(df,m,fold,meta['dataset_hash'],include_mlp=True,n_boot=args.n_boot,
                            importance_rows=500 if args.smoke else 50_000,
                            importance_repeats=2 if args.smoke else 5,log=log)
            for name,table in tables.items():
                table.to_csv(folder/f'{name}.csv',index=False)
            (folder/'complete.json').write_text(json.dumps({p.name:sha256(p) for p in folder.glob('*.csv')},indent=2)+'\n')
            log(f"completed {fold.held_out_month}")
    aggregate(out)
    (out/'complete.json').write_text(json.dumps({"folds":len(folds),"dataset_hash":meta['dataset_hash'],
                                                "smoke":args.smoke},indent=2)+'\n')
    log(f"all {len(folds)} folds complete: {out}")
