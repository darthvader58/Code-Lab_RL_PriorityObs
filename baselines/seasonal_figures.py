"""Seasonal figures and generated manuscript tables; every value comes from CSV/JSON."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .missions import ROOT


def render(mission, results=None, output=None):
    res=Path(results or mission.results_dir/'seasonal_lomo')
    out=Path(output or ROOT/'docs'/'baseline_paper'/'figures'/mission.name/'seasonal_lomo')
    meta=json.loads((res/'run_meta.json').read_text())
    completion=json.loads((res/'complete.json').read_text())
    if completion['folds'] != 8:
        raise ValueError('paper figures require all eight completed folds')
    if meta['smoke']:
        raise ValueError('smoke results must not be used for paper figures')
    out.mkdir(parents=True,exist_ok=True)
    monthly=pd.read_csv(res/'e2_monthly.csv')
    fixed=monthly[(monthly.budget_rule=='fixed') & np.isclose(monthly.nominal_budget,.20)]
    curve=pd.read_csv(res/'e3_capture_curves.csv')
    summary=pd.read_csv(res/'e2_summary.csv')
    summary=summary[np.isclose(summary.nominal_budget,.20)]
    names=list(fixed.policy.unique())
    colors={name:plt.get_cmap('tab10')(i) for i,name in enumerate(names)}
    def finish(fig,name):
        fig.savefig(out/name,dpi=180,bbox_inches='tight'); plt.close(fig)
    for metric,title in [('auc','ROC-AUC'),('capture','Capture at nominal 20% budget')]:
        fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
        for name,g in fixed.groupby('policy',sort=False):
            ax.plot(g.held_out_month,g[metric],marker='o',ms=4,label=name,color=colors[name],
                    linestyle='--' if name in ('P1 random','P8 oracle') else '-')
        ax.set(xlabel='Held-out G5NR weather month (Feb tail merged into Jan)',ylabel=title,
               title=f'{mission.name.upper()} seasonal robustness — {title}',ylim=(0,1.03))
        ax.legend(bbox_to_anchor=(1.01,1),loc='upper left',fontsize=8)
        finish(fig,f'monthly_{metric}.png')
    fig,axes=plt.subplots(2,4,figsize=(15,8),sharex=True,sharey=True)
    fig.subplots_adjust(left=.06,right=.99,top=.96,bottom=.17,hspace=.18,wspace=.12)
    for ax,(month,g) in zip(axes.flat,curve.groupby('held_out_month',sort=True)):
        for name,p in g.groupby('policy',sort=False):
            ax.plot(p.nominal_budget,p.capture,label=name,color=colors[name],lw=1.4)
        ax.set(title=month,xlim=(0,1),ylim=(0,1.02))
    fig.supxlabel('Nominal radar budget (training-calibrated; realized duty may differ)',y=.09)
    fig.supylabel('Fraction of positive decisions captured')
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.005),ncols=4,fontsize=8)
    finish(fig,'seasonal_capture_curves.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for name,g in fixed.groupby('policy',sort=False):
        if name=='P1 random':
            axes[0].plot(g.held_out_month,g.base_rate,'o-',color='black')
        if name!='P8 oracle':
            axes[1].plot(g.held_out_month,g.realized_duty,'o-',label=name,color=colors[name],ms=3)
    axes[0].set(title='Monthly radar-label base rate',ylabel='Positive fraction')
    axes[1].axhline(.20,color='black',ls='--',lw=1)
    axes[1].set(title='Realized duty at nominal 20%',ylabel='Fraction of held-out decisions fired')
    for ax in axes: ax.tick_params(axis='x',rotation=35)
    axes[1].legend(fontsize=7,bbox_to_anchor=(1.01,1),loc='upper left')
    finish(fig,'monthly_base_rate_and_duty.png')
    historical=pd.read_csv(mission.results_dir/'e3_capture_table.csv').set_index('policy')
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    learned=summary[summary.policy.str.startswith(('P2','P4','P5','P6','P7'))]
    x=np.arange(len(learned))
    for ax,metric,old_col in [(axes[0],'auc','auc'),(axes[1],'capture','capture@20%')]:
        mean=learned[f'{metric}_mean'].to_numpy()
        ax.errorbar(x-.08,mean,yerr=[mean-learned[f'{metric}_min'],learned[f'{metric}_max']-mean],
                    fmt='o',capsize=3,label='Seasonal: mean and min–max of 8 months')
        ax.plot(x+.08,historical.loc[learned.policy,old_col],'s',label='Chronological: pooled later-season stress test')
        ax.set_xticks(x,[p.split()[0] for p in learned.policy])
        ax.set(ylabel=metric,title=metric.upper() if metric=='auc' else 'Capture at nominal 20%')
    axes[0].legend(fontsize=7,loc='lower left')
    fig.suptitle(f'{mission.name.upper()}: different estimands and training sizes; difference is not a causal seasonal penalty')
    finish(fig,'chronological_vs_seasonal.png')
    importance=pd.read_csv(res/'e11_permutation_importance.csv')
    piv=importance.pivot(index='feature',columns='held_out_month',values='auc_drop_mean')
    fig,ax=plt.subplots(figsize=(9,6),layout='constrained')
    im=ax.imshow(piv,aspect='auto',cmap='viridis')
    ax.set_xticks(range(len(piv.columns)),piv.columns,rotation=35)
    ax.set_yticks(range(len(piv.index)),piv.index)
    ax.set_title(f'{mission.name.upper()}: P6 permutation AUC loss by held-out month')
    fig.colorbar(im,ax=ax,label='Held-out AUC decrease on permutation')
    finish(fig,'monthly_feature_importance.png')
    return out


def markdown_table(df):
    def cell(v):
        if isinstance(v,(float,np.floating)):
            return f'{v:.4f}' if np.isfinite(v) else 'NA'
        return str(v).replace('|','/')
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(cell(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])


def write_report(missions, destination):
    lines=['# Seasonal robustness results — generated from result files',
           '', 'Generated by `python -m baselines.paper_figures --protocol seasonal --report`. Do not edit numerical tables by hand.',
           '', 'Eight leave-one-G5NR-month-out folds are the primary seasonal evaluation. February 2006 is merged into January. '
           'Means weight months equally; SD is the sample SD across eight months. Each model is fitted on the other seven months. '
           'Thresholds use training-score quantiles. Capture@20% therefore refers to a nominal budget; realized duty is reported.',
           '', 'The chronological future-season stress test retains June–September training, October calibration and November–January testing '
           '(including the partial February tail). Differences between protocols also reflect training size and calibration; they do not isolate a causal seasonal penalty.',
           '', 'These folds share training data and one G5NR weather realization. They are not independent climate replicates or evidence of year-round, multi-year generalization.']
    for m in missions:
        res=m.results_dir/'seasonal_lomo'
        meta=json.loads((res/'run_meta.json').read_text())
        completion=json.loads((res/'complete.json').read_text())
        if completion['folds'] != 8:
            raise ValueError('paper figures require all eight completed folds')
        if meta['smoke']: raise ValueError('cannot report smoke results')
        e2=pd.read_csv(res/'e2_monthly.csv')
        e2=e2[(e2.budget_rule=='fixed') & np.isclose(e2.nominal_budget,.20)]
        s=pd.read_csv(res/'e2_summary.csv'); s=s[np.isclose(s.nominal_budget,.20)]
        lines += ['',f'## {m.name.upper()}', '',f'Dataset SHA-256: `{meta["dataset_hash"]}`; seed `{meta["seed"]}`; '
                  f'threshold source `{meta["threshold_source"]}`. Source: `missions/{m.name}/results/seasonal_lomo/`.',
                  '', '### Per-month scores at nominal 20% budget', '', markdown_table(e2[['held_out_month','policy','n_test','base_rate','auc','average_precision','capture','mass_capture','precision','realized_duty']]),
                  '', '### Equal-weight mean, SD and range across months', '',markdown_table(s[['policy','auc_mean','auc_std','auc_min','auc_max','capture_mean','capture_std','capture_min','capture_max','realized_duty_min','realized_duty_max']]),
                  '', '### Monthly block-bootstrap intervals',
                  '', 'Whole-day resampling is performed separately within each month with fixed gates. Below are the P6 intervals; all policies and metrics are in `e10_bootstrap.csv`. '
                  'Across-fold summaries of interval endpoints are descriptive, not confidence intervals for a pooled all-season score.', '']
        b=pd.read_csv(res/'e10_bootstrap.csv')
        lines += [markdown_table(b[(b.policy=='P6 gradient boosting') & b.metric.isin(['auc','capture@20%'])][['held_out_month','metric','mean','ci_low','ci_high','n_boot']]),
                  '', '### Additional seasonal analyses', '']
        for title,filename,columns in [
            ('E4: geometry by training and scoring label','e4_label_geometry_summary',
             ['variable','trained_on','scored_on','auc_mean','capture_mean','capture_min','capture_max','precision_mean','base_rate_mean']),
            ('E5: tiers and illumination at nominal 20%','e5_tiers_summary',
             ['tier','regime','capture_mean','capture_std','capture_min','capture_max','realized_duty_mean']),
            ('E6: global thresholds within strata at nominal 20%','e6_strata_summary',
             ['stratum','auc_mean','capture_mean','capture_min','capture_max','realized_duty_min','realized_duty_max']),
            ('E7: label sensitivity at nominal 20%','e7_label_sensitivity_summary',
             ['policy','label','auc_mean','average_precision_mean','capture_mean','capture_min','capture_max','base_rate_mean']),
            ('E11: monthly importance mean and range','e11_permutation_importance_summary',
             ['feature','auc_drop_mean_mean','auc_drop_mean_std','auc_drop_mean_min','auc_drop_mean_max']),
        ]:
            table=pd.read_csv(res/(filename+'.csv'))
            if 'nominal_budget' in table: table=table[np.isclose(table.nominal_budget,.20)]
            lines += ['', '### '+title, '', markdown_table(table[columns]), '']
        lines += ['### Seasonal figures', '']
        for name,caption in [('monthly_auc','Per-month AUC; P8 is an oracle reference.'),
                             ('monthly_capture','Per-month capture at nominal 20%; see realized duty below.'),
                             ('seasonal_capture_curves','Independent capture curves for all held-out months and policies.'),
                             ('monthly_base_rate_and_duty','Monthly base rate and realized duty at nominal 20%.'),
                             ('chronological_vs_seasonal','Chronological stress test versus equal-weight seasonal mean and range; different estimands.'),
                             ('monthly_feature_importance','P6 fixed in advance: held-out permutation importance by month.')]:
            lines += [f'![{caption}](figures/{m.name}/seasonal_lomo/{name}.png)', '',caption,'']
    comparison=ROOT/'missions/comparison/results/seasonal_lomo'
    cm=json.loads((comparison/'run_meta.json').read_text())
    if cm['smoke']: raise ValueError('cannot report smoke comparison')
    table=pd.read_csv(comparison/'e12_comparison.csv')
    lines += ['', '## E12: matched seasonal comparison', '',
              f"Both missions hold out the same G5NR weather month and restrict fitting, calibration and testing to |latitude| <= {cm['latitude_limit']} degrees. "
              'The primary labels remain mission-specific; these scores are not a common-footprint instrument contest. The same G5NR weather limits independence.', '',
              markdown_table(table[['mission','held_out_month','policy','base_rate','auc','average_precision','capture','mass_capture','realized_duty']]), '']
    transfer=pd.read_csv(comparison/'e12_transfer.csv')
    transfer=transfer[np.isclose(transfer.nominal_budget,.20)]
    groups=['trained_on','tested_on','label_setting','features','policy']
    stats=transfer.groupby(groups)[['auc','capture','realized_duty']].agg(['mean','min','max'])
    stats.columns=['_'.join(c) for c in stats.columns]
    lines += ['### Transfer: monthly mean and range at nominal 20%', '',
              'Thresholds use target-side **other-seven-month unlabeled scores**, never held-out labels. Compare each cross-mission row with the corresponding own-mission row using the same features and label setting.',
              '', markdown_table(stats.reset_index()), '']
    dest=Path(destination); dest.write_text('\n'.join(lines)+'\n')
