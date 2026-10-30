"""Audit generated seasonal evidence and preservation of the original local files.

Run after both full suites and E12: python -m baselines.validate_seasonal
The preservation snapshot was taken before implementation, without modifying any dataset.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import load, seasonal_folds
from .missions import EARTHCARE, TRMM, ROOT
from .run_baselines import sha256
from .seasonal import THRESHOLD_SOURCE


def main():
    evidence=ROOT/'task_outputs/seasonal_validation'
    original=json.loads((evidence/'original_file_hashes.json').read_text())
    changed=[name for name,digest in original.items() if not (ROOT/name).exists() or sha256(ROOT/name)!=digest]
    assert not changed, f'original files changed: {changed}'
    report={'original_files_unchanged':len(original),'missions':{},'checks':[
        'eight weather-month holdouts; February merged into January',
        'train/test weather months and indices disjoint; both sides of boundary purged',
        'preprocessing isolation and threshold label-invariance tested by tests/test_seasonal.py',
        'every policy reports nominal and realized held-out duty',
        'global fold thresholds preserved for E6 strata',
        'original E2 replay and generalized P5-P7 scores compared to saved E2',
        'P8 marked as a truth-using reference, including bootstrap metadata',
        '1000 whole-day bootstrap draws within each held-out month',
        'dataset hashes and recorded code hashes match current source',
    ]}
    for m in (EARTHCARE,TRMM):
        res=m.results_dir/'seasonal_lomo'
        assert json.loads((res/'complete.json').read_text())['folds']==8
        meta=json.loads((res/'run_meta.json').read_text())
        assert not meta['smoke'] and meta['dataset_hash']==sha256(m.data_path)
        assert all(sha256(ROOT/'baselines'/name)==h for name,h in meta['code_hashes'].items())
        data=load(m.data_path,m);folds=list(seasonal_folds(data))
        months=[f.held_out_month for f in folds]
        assert len(months)==8 and '2006-02' not in months
        assert '2006-02' in set(data.loc[folds[-1].test_index,'g5_month'])
        for f in folds:
            tr,te=f.frames(data,[m.label_col,m.mass_col])
            train_m=set(tr.g5_month.replace({'2006-02':'2006-01'}))
            test_m=set(te.g5_month.replace({'2006-02':'2006-01'}))
            assert train_m==set(f.train_months) and test_m=={f.held_out_month}
            assert not tr.index.intersection(te.index).size
            assert not set(tr.g5_step)&set(te.g5_step)
            assert not set(tr.g5_step)&set(f.buffer_steps)
            assert not set(te.g5_step)&set(f.buffer_steps)
            lo,hi=te.g5_step.min(),te.g5_step.max()
            before=tr.loc[tr.g5_step<lo,'g5_step'];after=tr.loc[tr.g5_step>hi,'g5_step']
            if len(before): assert lo-before.max()>=3
            if len(after): assert after.min()-hi>=3
            folder=res/'folds'/f.held_out_month
            for name,h in json.loads((folder/'complete.json').read_text()).items():
                assert sha256(folder/name)==h
        monthly=pd.read_csv(res/'e2_monthly.csv')
        required=['mission','policy','held_out_month','train_months','n_train','n_test','base_rate',
                  'nominal_budget','realized_duty','threshold_source','auc','average_precision',
                  'capture','mass_capture','precision','seed','dataset_hash']
        assert not set(required)-set(monthly.columns)
        assert monthly.groupby(['held_out_month','budget_rule']).policy.nunique().eq(8).all()
        assert monthly.realized_duty.between(0,1).all()
        assert monthly.nominal_budget.between(0,1).all()
        assert monthly.dataset_hash.eq(meta['dataset_hash']).all()
        assert monthly.loc[monthly.policy!='P8 oracle','threshold_source'].eq(THRESHOLD_SOURCE).all()
        assert monthly.loc[monthly.policy=='P8 oracle','threshold_source'].eq('oracle_held_out_truth_reference').all()
        for f in folds:
            tr,te=f.frames(data,[m.label_col,m.mass_col])
            rows=monthly[monthly.held_out_month==f.held_out_month]
            assert rows.n_train.eq(len(tr)).all() and rows.n_test.eq(len(te)).all()
            assert np.allclose(rows.base_rate,(te[m.label_col]>=m.threshold).mean())
        strata=pd.read_csv(res/'e6_strata.csv')
        assert strata.groupby(['held_out_month','nominal_budget']).score_threshold.nunique().eq(1).all()
        global_p6=monthly[(monthly.policy=='P6 gradient boosting') & (monthly.budget_rule=='fixed')]
        join=strata.merge(global_p6[['held_out_month','nominal_budget','score_threshold']],on=['held_out_month','nominal_budget'],suffixes=('_stratum','_global'))
        assert np.allclose(join.score_threshold_stratum,join.score_threshold_global)
        boot=pd.read_csv(res/'e10_bootstrap.csv')
        assert set(boot.held_out_month)==set(months) and boot.n_boot.eq(1000).all()
        assert boot.loc[boot.policy=='P8 oracle','threshold_source'].eq('oracle_held_out_truth_reference').all()
        assert boot.n_days.between(28,33).all()
        old=pd.read_csv(m.results_dir/'e2_leave_one_month_out.csv').set_index(['policy','held_out'])
        replay=pd.read_csv(evidence/f'e2_original_replay_{m.name}.csv').set_index(['policy','held_out'])
        replay_error=float((replay-old).abs().max().max())
        assert replay_error<1e-12
        got=monthly[(monthly.budget_rule=='fixed') & np.isclose(monthly.nominal_budget,.13)]
        got=got[got.policy.str.startswith(('P5','P6','P7'))].set_index(['policy','held_out_month'])
        got.index.names=old.index.names
        got=got.loc[old.index]
        errors={new:float((got[new]-old[previous]).abs().max()) for new,previous in
                [('auc','auc'),('average_precision','ap'),('capture','capture@13%'),('n_test','n_test')]}
        assert max(errors.values())<1e-5, errors
        base=monthly[(monthly.budget_rule=='training_base_rate') & monthly.policy.str.startswith(('P5','P6','P7'))].set_index(['policy','held_out_month'])
        base.index.names=old.index.names
        base_error=float((base.loc[old.index,'capture']-old['capture@base_rate']).abs().max())
        assert base_error<1e-5
        report['missions'][m.name]={'dataset_hash':meta['dataset_hash'],'folds':months,
            'monthly_rows':len(monthly),'policies':8,'bootstrap_draws_per_month':1000,
            'original_e2_replay_max_error':replay_error,'generalized_e2_max_errors':errors,
            'base_rate_budget_capture_max_error':base_error}
    comp=ROOT/'missions/comparison/results/seasonal_lomo'
    cm=json.loads((comp/'run_meta.json').read_text())
    assert not cm['smoke'] and cm['folds']==months
    assert cm['latitude_limit']==35.2
    c=pd.read_csv(comp/'e12_comparison.csv')
    assert c.groupby('mission').held_out_month.nunique().eq(8).all()
    assert c.latitude_limit.eq(35.2).all()
    assert all(sha256(m.data_path)==cm['dataset_hashes'][m.name] for m in (EARTHCARE,TRMM))
    transfer=pd.read_csv(comp/'e12_transfer.csv')
    assert transfer.threshold_source.eq('target_other_seven_months_unlabeled_scores').all()
    report['comparison']={'matched_months':months,'latitude_limit':35.2,'rows':len(c),'transfer_rows':len(transfer)}
    (evidence/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
