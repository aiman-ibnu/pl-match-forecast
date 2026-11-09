"""Generate the three notebooks from plain code cells, then execute them (python scripts/build_notebooks.py)."""
from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
HEAD = "import sys, pathlib\nsys.path.insert(0, str(pathlib.Path.cwd().parent))\nimport warnings; warnings.filterwarnings('ignore')\n"


def make(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(t) if k == "md" else nbf.v4.new_code_cell(t) for k, t in cells]
    ExecutePreprocessor(timeout=600, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(NB)}})
    nbf.write(nb, NB / name)


make("01_data_collection.ipynb", [
    ("md", "# 01 Data collection\nSource: openfootball/football.json (CC0). We download one JSON file per season and clean it. No scraping of restricted sites."),
    ("code", HEAD + "import config\nfrom src import data_collection as dc, data_processing as dp\nprint(config.SEASONS)"),
    ("code", "# Re-download with: python -m src.data_collection  (not done here, so results stay tied to the stored snapshot)\nprint(sorted(p.name for p in config.RAW_DIR.glob('*.json')))\nm = dp.load()\nm.head()"),
    ("md", "## Integrity checks\n20 clubs per season, no duplicate fixtures, 380 matches in complete seasons, and fixtures without a real score are NOT treated as results."),
    ("code", "info = dp.validate(m)\nimport pandas as pd\npd.DataFrame(info).T"),
    ("code", "print('fixtures', len(m), '| played', int(m['played'].sum()), '| clubs', m['home'].nunique())\nprint(m[~m['played']].groupby('season').size())"),
])

make("02_eda.ipynb", [
    ("md", "# 02 Exploratory analysis\nQuestions: how strong is home advantage, how many goals, how does Elo relate to results?"),
    ("code", HEAD + "import pandas as pd, matplotlib.pyplot as plt, config\nfrom src import data_processing as dp\nm = dp.load(); p = m[m['played']]\nfigs = config.FIGURES_DIR; figs.mkdir(parents=True, exist_ok=True)"),
    ("code", "share = p.groupby('season')['result'].value_counts(normalize=True).unstack()[['H','D','A']]\nax = share.plot(marker='o', figsize=(9,4), color=['#2563eb','#9ca3af','#dc2626']); ax.set_ylabel('share of matches'); ax.set_title('Result shares per season')\nplt.tight_layout(); plt.savefig(figs/'result_share_by_season.png', dpi=130); plt.show()\nshare.round(3)"),
    ("md", "Home wins are the most common result in every season. The 2020-21 season (empty stadiums) shows the smallest home advantage."),
    ("code", "g = (p['home_goals']+p['away_goals']).groupby(p['season']).mean()\nax = g.plot.bar(figsize=(9,3.5), color='#2563eb'); ax.set_ylabel('goals per match')\nplt.tight_layout(); plt.savefig(figs/'goals_per_match.png', dpi=130); plt.show()"),
    ("code", "f = pd.read_csv(config.FEATURES_FILE).dropna(subset=['elo_diff'])\nf['bin'] = pd.qcut(f['elo_diff'], 8)\nt = f.groupby('bin', observed=True)['result'].value_counts(normalize=True).unstack()[['H','D','A']]\nax = t.plot.bar(stacked=True, figsize=(9,4), color=['#2563eb','#9ca3af','#dc2626']); ax.set_xlabel('Elo difference (home minus away), 8 equal-size groups'); ax.set_ylabel('share')\nplt.tight_layout(); plt.savefig(figs/'result_by_elo_diff.png', dpi=130); plt.show()\nt.round(3)"),
    ("md", "The bigger the Elo advantage of the home club, the more home wins and the fewer away wins. Draws are most common between evenly matched clubs (about 27%) and rarer at the extremes (17-20%), but never more likely than the best alternative, which is why draws are hard to predict."),
])

make("03_modelling.ipynb", [
    ("md", "# 03 Modelling and evaluation\nOut-of-time split: train on 2015-16..2022-23 (2014-15 is only Elo warm-up), test on 2023-24 and 2024-25, 2025-26 is an incomplete partial check. Hyper-parameters chosen by expanding-window CV on train only."),
    ("code", HEAD + "import pandas as pd, json, matplotlib.pyplot as plt, config\nfrom src import evaluation as ev\nfigs = config.FIGURES_DIR\nmt = pd.read_csv(config.METRICS_FILE)\nmt[mt['set']=='test'][['model','accuracy','log_loss','RPS','Brier','cv_log_loss','selected_by_cv']].round(4)"),
    ("code", "print(json.dumps(json.load(open(config.REPORTS_DIR/'bootstrap_best_vs_baselines.json')), indent=1))"),
    ("md", "Reading: the selected model beats always-home clearly (interval excludes 0) but is indistinguishable from the Elo-only baseline."),
    ("code", "pd.read_csv(config.REPORTS_DIR/'ablation_features.csv').round(4)"),
    ("code", "pr = pd.read_csv(config.PREDICTIONS_FILE); te = pr[pr['set']=='test']\ncal = ev.calibration_table(te['result'], te['pH'])\nfig, ax = plt.subplots(figsize=(4.5,4.5)); ax.plot([0,1],[0,1],'--',color='#9ca3af'); ax.plot(cal['pred'], cal['actual'], 'o-', color='#2563eb')\nax.set_xlabel('predicted P(home win)'); ax.set_ylabel('actual rate'); plt.tight_layout(); plt.savefig(figs/'calibration_home.png', dpi=130); plt.show(); cal.round(3)"),
    ("code", "print(pd.crosstab(te['result'], te['pred']))"),
    ("md", "No draw is ever the most likely class, so the model never predicts one. This is a known limit of 3-class football models using only scores."),
    ("code", "imp = pd.read_csv(config.REPORTS_DIR/'permutation_importance_test.csv').sort_values('logloss_increase')\nax = imp.plot.barh(x='feature', y='logloss_increase', legend=False, figsize=(6,4), color='#2563eb'); ax.set_xlabel('log-loss increase when shuffled')\nplt.tight_layout(); plt.savefig(figs/'permutation_importance.png', dpi=130); plt.show()"),
])
print("notebooks built")
