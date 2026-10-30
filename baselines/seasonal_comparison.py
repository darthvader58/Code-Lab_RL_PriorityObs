"""E12 under matched weather-month holdouts and a common latitude domain."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .core import LearnedPolicy, fit_seasonal_policy, load, seasonal_folds
from .experiments import label_fn
from .missions import EARTHCARE, TRMM, GEOMETRY
from .run_baselines import sha256
from .seasonal import CURVE, PROTOCOL, metadata, scored_rows, summarize


def run(out, *, smoke=False):
    os.environ["LOKY_MAX_CPU_COUNT"] = "2"
    out=Path(out)
    if out.resolve() == (EARTHCARE.results_dir.parents[1]/'comparison'/'results').resolve():
        raise ValueError('use a separate seasonal comparison directory')
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'output already exists: {out}')
    missions={m.name:m for m in (EARTHCARE,TRMM)}
    data={n:load(m.data_path,m) for n,m in missions.items()}
    hashes={n:sha256(m.data_path) for n,m in missions.items()}
    # Constant geographic comparison region; no held-out labels used to define it.
    latitude_limit=35.2
    folds={n:list(seasonal_folds(df)) for n,df in data.items()}
    assert [f.held_out_month for f in folds['earthcare']] == [f.held_out_month for f in folds['trmm']]
    own,curves,transfer,local=[],[],[],[]
    with threadpool_limits(limits=2):
        for i in range(8):
            month=folds['earthcare'][i].held_out_month
            print(f'E12 seasonal {month}',flush=True)
            band={n:df[df.lat.abs()<=latitude_limit] for n,df in data.items()}
            if smoke:
                band={n:df.groupby('g5_month',group_keys=False).sample(n=150,random_state=0) for n,df in band.items()}
            for n,m in missions.items():
                f=folds[n][i]
                tr,te=f.frames(band[n],[m.label_col,m.mass_col,m.swath_col])
                lab=label_fn(m.label_col,m.threshold)
                common={**metadata(m,f,tr,te,hashes[n]),'latitude_limit':latitude_limit}
                for pname,kind,tier in [('P5 logistic regression','lr','A'),('P6 gradient boosting','lgbm','A'),
                                        ('P7 MLP','mlp','A'),('P6 Tier B','lgbm','B')]:
                    _,st,se=fit_seasonal_policy(LearnedPolicy(kind,m.tiers[tier]),tr,te,lab)
                    res=[{**common,'policy':pname,'tier':tier,'swath_base_rate':float((te[m.swath_col]>=m.threshold).mean()),**r}
                         for r in scored_rows(st,se,lab(te),te[m.mass_col],CURVE)]
                    curves.extend(res)
                    own.extend(r for r in res if np.isclose(r['nominal_budget'],.20))
                    if tier=='A':
                        # Global training-fold threshold retained in every local-time group.
                        hour=te.solar_hour.to_numpy()
                        window=np.logical_or.reduce([np.abs((hour-node+12)%24-12)<1 for node in (1+56/60,13+56/60)])
                        for group,mask in [('EarthCARE windows',window),('other hours',~window)]:
                            if not mask.any(): continue
                            local.extend({**common,'policy':pname,'group':group,'n_test':int(mask.sum()),
                                          'threshold_scope':'global_training_fold',**r}
                                         for r in scored_rows(st,se[mask],lab(te)[mask],te[m.mass_col].to_numpy()[mask],(.20,)))
            for setting in ('nadir','own radar'):
                for feat_name,features in [('shared cloud + geometry',TRMM.tiers['A']),('geometry only',GEOMETRY)]:
                    for kind,pname in [('lr','P5 logistic regression'),('lgbm','P6 gradient boosting')]:
                        for src,ms in missions.items():
                            target_src='prectot_nadir' if setting=='nadir' else ms.label_col
                            fs=folds[src][i]
                            train,_=fs.frames(band[src],[target_src])
                            pol=LearnedPolicy(kind,features).fit(train,label_fn(target_src,ms.threshold)(train))
                            for dst,md in missions.items():
                                target_dst='prectot_nadir' if setting=='nadir' else md.label_col
                                mass_dst=target_dst if setting=='nadir' else md.mass_col
                                fd=folds[dst][i]
                                calibration,test=fd.frames(band[dst],[target_dst,mass_dst])
                                # Only target's OTHER seven months supply unlabeled calibration scores.
                                common=metadata(md,fd,train,test,hashes[dst])
                                for r in scored_rows(pol.score(calibration),pol.score(test),
                                                     label_fn(target_dst,md.threshold)(test),test[mass_dst],(.13,.20)):
                                    transfer.append({**common,'policy':pname,'trained_on':src,'tested_on':dst,
                                                     'source_dataset_hash':hashes[src],'label_setting':setting,
                                                     'features':feat_name,'latitude_limit':latitude_limit,
                                                     'n_calibration':len(calibration),**r,
                                                     'threshold_source':'target_other_seven_months_unlabeled_scores'})
    out.mkdir(parents=True)
    for name,rows in [('e12_comparison',own),('e12_capture_curves',curves),('e12_transfer',transfer),('e12_local_time',local)]:
        table=pd.DataFrame(rows)
        table.to_csv(out/f'{name}.csv',index=False)
        if name=='e12_comparison':
            summarize(table,['mission','policy','nominal_budget']).to_csv(out/f'{name}_summary.csv',index=False)
    (out/'run_meta.json').write_text(json.dumps({'protocol':PROTOCOL,'smoke':smoke,'dataset_hashes':hashes,
        'latitude_limit':latitude_limit,'seed':0,'folds':[f.held_out_month for f in folds['earthcare']],
        'threshold_source':'own: training scores; transfer: target other-seven-month unlabeled scores',
        'interpretation':'matched weather months; separate mission labels and shared G5NR weather, not independent climate realizations'},indent=2)+'\n')
    print(f'E12 complete: {out}',flush=True)
