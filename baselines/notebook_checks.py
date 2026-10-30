"""Read-only notebook provenance checks in the environment used for baseline computation.

Some local notebooks use a TATC kernel without LightGBM. BASELINE_PYTHON can point to the
baseline environment; the existing local /opt/anaconda3 environment is the fallback here.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def check_in_baseline_environment(mission):
    root=Path(__file__).resolve().parents[1]
    python=os.environ.get('BASELINE_PYTHON')
    if python is None:
        python=sys.executable if importlib.util.find_spec('lightgbm') else '/opt/anaconda3/bin/python'
    if not Path(python).is_file():
        raise RuntimeError('Set BASELINE_PYTHON to the interpreter used to run the baseline suite.')
    return json.loads(subprocess.check_output([python,'-m','baselines.notebook_checks',mission],cwd=root,text=True))


def verify(name):
    from .core import load, seasonal_folds
    from .missions import get_mission, ROOT
    from .run_baselines import sha256
    if name=='comparison':
        meta=json.loads((ROOT/'missions/comparison/results/seasonal_lomo/run_meta.json').read_text())
        assert not meta['smoke']
        for mission,digest in meta['dataset_hashes'].items():
            m=get_mission(mission)
            assert sha256(m.data_path)==digest
            assert [f.held_out_month for f in seasonal_folds(load(m.data_path,m))]==meta['folds']
        return meta
    m=get_mission(name); res=m.results_dir/'seasonal_lomo'
    meta=json.loads((res/'run_meta.json').read_text())
    assert not meta['smoke'] and (res/'complete.json').exists()
    assert sha256(m.data_path)==meta['dataset_hash']
    folds=list(seasonal_folds(load(m.data_path,m)))
    assert [f.held_out_month for f in folds]==[f['held_out_month'] for f in meta['folds']]
    for fold,saved in zip(folds,meta['folds']):
        assert list(fold.train_months)==saved['train_months']
        assert list(fold.buffer_steps)==saved['buffer_steps']
        assert len(fold.train_index)==saved['n_train_before_label_filter']
        assert len(fold.test_index)==saved['n_test_before_label_filter']
    return meta


if __name__=='__main__':
    print(json.dumps(verify(sys.argv[1])))
