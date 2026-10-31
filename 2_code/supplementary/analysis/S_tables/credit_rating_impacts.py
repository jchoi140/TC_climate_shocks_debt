"""SI Table T11 (`tab:supp_rating_notch_attribution_summary`): sovereign credit
rating impact of climate shocks, summarised by 2019 rating panel.

Per country: notch = E[rating | observed] - E[rating | counterfactual-combined],
with E[.] taken over the XGB classifier's predicted rating distribution
(500 seeds, main model XGB_classification_neglogloss_0_500), then averaged across
seeds. Negative = downgrade attributable to climate. The 181-country universe
is the set of ISOs in attributable_shares.csv with an XGB prediction.

Panels: Investment grade / Speculative grade by the max 2019 rating across S&P,
Moody's, Fitch (IG = score >= 12); Non-rated = no 2019 rating from any agency;
SIDS = prep.SIDS_ISOS (38 listed, 33 with a prediction).

Run:
    python3 credit_rating_impacts.py
Output:
    3_output/S_tables/credit_rating_impacts.csv
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common"))
from paths import p, read_predictions  # noqa: E402
from prep import SIDS_ISOS  # noqa: E402

PRED = p("3_output", "XGB", "predictions", "XGB_classification_neglogloss_0_500.csv")
OUT = p("3_output", "S_tables", "credit_rating_impacts.csv")


def expected_rating(probs):
    """E[class] from space-separated class-probability strings."""
    arr = np.vstack([np.fromstring(s, sep=" ") for s in probs])
    return arr @ np.arange(arr.shape[1], dtype=float)


preds = read_predictions(PRED)
attr_isos = set(pd.read_csv(p("3_output", "counterfactual_data", "attributable_shares.csv"), usecols=["ISO"])["ISO"])
preds = preds[preds["ISO"].isin(attr_isos)].copy()
preds["notch"] = expected_rating(preds["y_probs"]) - expected_rating(preds["y_probs_cf_combined"])
per_iso = preds.groupby("ISO")["notch"].mean()

# 2019 grade by the best agency rating (shipped aggregate, data_prep/build_proprietary_aggregates.py)
grade_2019 = pd.read_csv(p("1_input", "credit_ratings", "rating_grade_2019.csv")).set_index("ISO")["grade_2019"]
isos = set(per_iso.index)
ig = set(grade_2019.index[grade_2019 == "IG"]) & isos
sg = set(grade_2019.index[grade_2019 == "SG"]) & isos
panels = {
    "All countries": isos,
    "Investment grade": ig,
    "Speculative grade": sg,
    "Non-rated": isos - ig - sg,
    "SIDS": set(SIDS_ISOS) & isos,
}

rows = []
for name, members in panels.items():
    v = per_iso[sorted(members)]
    rows.append(dict(panel=name, n=len(v), mean=v.mean(), median=v.median(), min=v.min(), max=v.max()))
table = pd.DataFrame(rows)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
table.to_csv(OUT, index=False)
print(table.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
print(f"saved: {OUT}")
