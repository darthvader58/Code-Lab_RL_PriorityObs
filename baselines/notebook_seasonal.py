"""Shared seasonal notebook cells; generators and existing notebooks use this same adapter."""
import nbformat as nbf

TAG='seasonal_lomo'


def cells_for(mission):
    md=nbf.v4.new_markdown_cell
    code=nbf.v4.new_code_cell
    cells=[md('''# Primary evaluation: seasonal robustness

Seasonal leave-one-month-out asks whether the policy remains reliable across the sampled weather
regimes. The **Chronological future-season stress test** below separately asks how preparation on
June–September and October transfers into November–January, including the existing February tail.
The latter is not evidence of general all-season performance. LOMO trains on the other seven
G5NR weather months, which may include months after the held-out month. All preprocessing and
thresholds use training-side data. Quantile gates have nominal budgets, not guaranteed test duty.

This section reads generated results and verifies fold definitions with `baselines.core`; it has
no independent split logic and performs no model fitting, extraction or export. Provenance checks
run in the baseline Python environment (`BASELINE_PYTHON`, or the existing local baseline interpreter)
so the original TATC notebook kernel can still be used. The source data
cover only a limited set of G5NR model-weather months. See `docs/baseline_paper/07_experiments.md`.
'''),code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Markdown, Image
SEASON_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'baselines/core.py').exists())
sys.path.insert(0, str(SEASON_ROOT))
from baselines.missions import get_mission
from baselines.notebook_checks import check_in_baseline_environment
SEASON_MISSION = get_mission(MISSION_PLACEHOLDER)
SEASON_RES = SEASON_MISSION.results_dir / 'seasonal_lomo'
season_meta = check_in_baseline_environment(SEASON_MISSION.name)
display(Markdown(f"Dataset SHA-256: `{season_meta['dataset_hash']}`; seed `{season_meta['seed']}`; threshold source `{season_meta['threshold_source']}`."))
display(pd.DataFrame(season_meta['folds']))
season_read = lambda name: pd.read_csv(SEASON_RES / (name + '.csv'))
'''.replace('MISSION_PLACEHOLDER',repr(mission))),md('''## E2/E3: every held-out month, every policy

The table uses a nominal 20% training-score budget. Read event/mass capture alongside precision,
base rate and realized duty; P8 alone uses held-out truth as an unattainable reference.
The summary weights months equally and reports sample SD and min–max, not a pooled score.
'''),code('''season_monthly = season_read('e2_monthly')
season_20 = season_monthly[(season_monthly.budget_rule == 'fixed') & np.isclose(season_monthly.nominal_budget, .20)]
with pd.option_context('display.max_rows', None):
    display(season_20[['held_out_month','policy','n_train','n_test','base_rate','nominal_budget','realized_duty','auc','average_precision','capture','mass_capture','precision']].round(4))
season_summary = season_read('e2_summary')
for metrics in [('auc','average_precision'),('capture','mass_capture'),('precision','realized_duty','base_rate')]:
    columns = ['policy','nominal_budget'] + [f'{metric}_{stat}' for metric in metrics for stat in ['mean','std','min','max']]
    display(season_summary.loc[np.isclose(season_summary.nominal_budget, .20), columns].round(4))
SEASON_FIG = SEASON_ROOT / 'docs/baseline_paper/figures' / SEASON_MISSION.name / 'seasonal_lomo'
for filename in ['monthly_auc.png','monthly_capture.png','seasonal_capture_curves.png','monthly_base_rate_and_duty.png','chronological_vs_seasonal.png']:
    display(Image(filename=str(SEASON_FIG / filename)))
'''),md('''## E4–E8: geometry, illumination, geography, labels and availability

Every table retains the held-out weather month. E6 uses the global training-fold threshold without
stratum retuning. E7 median labels and base-rate budgets are defined on training rows. E8 records
availability before fitting, including whether each feature can receive a training-fitted missing
indicator. Full tables remain in the CSVs; the displays below expose all monthly records.
'''),code('''for experiment in ['e4_label_geometry','e5_tiers','e6_strata','e7_label_sensitivity','e8_missing_data']:
    display(Markdown('### ' + experiment))
    table = season_read(experiment)
    if 'nominal_budget' in table:
        table = table[np.isclose(table.nominal_budget, .20)]
    with pd.option_context('display.max_rows', None):
        display(table.drop(columns=['dataset_hash','train_months','buffer_steps'], errors='ignore').round(4))
'''),md('''## E10/E11: within-month uncertainty and seasonal importance

Bootstrap resamples whole days inside each held-out month with gates fixed. P6 is a prespecified
paired-comparison and permutation-importance reference. Across-fold summaries of confidence-bound
endpoints are descriptive, not a confidence interval for a pooled all-season mean. Single-feature
score direction is chosen using training labels only. E1 and E9 retain the historical leakage
comparison and architecture-cost measurements separately below.
'''),code('''for experiment in ['e10_bootstrap','e10_bootstrap_summary','e11_permutation_importance','e11_permutation_importance_summary','e11_single_feature_auc']:
    display(Markdown('### ' + experiment))
    table = season_read(experiment)
    with pd.option_context('display.max_rows', None):
        display(table.drop(columns=['dataset_hash','train_months','buffer_steps'], errors='ignore').round(4))
display(Image(filename=str(SEASON_FIG / 'monthly_feature_importance.png')))
''')]
    for i,c in enumerate(cells):
        c.metadata['tags']=[TAG]; c.id=f'seasonal-{mission}-{i}'
    return cells


def comparison_cells():
    cells=[nbf.v4.new_markdown_cell('''# Primary E12: matched seasonal folds

EarthCARE and TRMM are evaluated separately on the same omitted G5NR weather month with all fitting,
calibration and testing restricted to `|lat| <= 35.2°`. Mission labels remain distinct. Transfer
uses shared features and target-side other-seven-month unlabeled scores for threshold calibration.
Local-time strata retain global thresholds. The historical chronological E12 section below answers
a different transfer question and must not be mixed into seasonal averages.
'''),nbf.v4.new_code_cell('''from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Markdown
SEASON_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'baselines/core.py').exists())
sys.path.insert(0, str(SEASON_ROOT))
from baselines.notebook_checks import check_in_baseline_environment
SEASON_COMPARE = SEASON_ROOT / 'missions/comparison/results/seasonal_lomo'
season_comparison_meta = check_in_baseline_environment('comparison')
display(season_comparison_meta)
for experiment in ['e12_comparison','e12_comparison_summary','e12_transfer','e12_local_time']:
    display(Markdown('## ' + experiment))
    table = pd.read_csv(SEASON_COMPARE / (experiment + '.csv'))
    with pd.option_context('display.max_rows', None):
        display(table.drop(columns=['dataset_hash','source_dataset_hash','train_months','buffer_steps'],errors='ignore').round(4))
''')]
    for i,c in enumerate(cells): c.metadata['tags']=[TAG]; c.id=f'seasonal-comparison-{i}'
    return cells


REPLACEMENTS={
    '# Part 2. Baseline results (E1–E11)':'# Historical evaluation: Chronological future-season stress test (E1–E11)',
    '## E2. Does skill hold across months?':'## Legacy E2: original P5–P7 seasonal table (retained for reproduction)',
    'The main split trains on Jun–Sep only':'The chronological stress test trains on Jun–Sep only',
    'time-blocked numbers because they are the honest test of generalising to new weather.':'chronological numbers to study future-season transfer; seasonal LOMO above studies robustness across months.',
    'That is an upper bound on the seasonal penalty, since those folds also train on 7 months ':'This is not an isolated seasonal penalty: those folds also train on 7 months ',
    'time-blocked numbers are conservative.':'chronological and seasonal evaluations still answer different questions.',
    '## Summary':'## Historical chronological summary',
    '## E3. Main result:':'## Historical E3. Chronological result:',
    '**What this means for the RL work.** These are the numbers to beat:':'**Historical bar for RL.** Also evaluate seasonal consistency above; these chronological numbers answer future-season transfer only:',
}


def attach(nb,mission):
    """Preserve historical cells/outputs, relabel scope, and add a separately executable section."""
    nb.cells=[c for c in nb.cells if TAG not in c.metadata.get('tags',[]) and c.source != '# Historical E12: Chronological future-season stress test']
    for cell in nb.cells:
        for old,new in REPLACEMENTS.items():
            cell.source=cell.source.replace(old,new)
        # Remove stale inferential claims from cached narrative outputs too.
        for output in cell.get('outputs',[]):
            for key,value in output.get('data',{}).items():
                if isinstance(value,str):
                    for old,new in REPLACEMENTS.items(): value=value.replace(old,new)
                    output['data'][key]=value
    inserted=comparison_cells() if mission=='comparison' else cells_for(mission)
    if mission=='comparison':
        nb.cells=inserted+[nbf.v4.new_markdown_cell('# Historical E12: Chronological future-season stress test')]+nb.cells
    else:
        pos=next((i for i,c in enumerate(nb.cells) if '# Historical evaluation:' in c.source),len(nb.cells))
        nb.cells[pos:pos]=inserted
    return nb
