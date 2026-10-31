"""
rating_ordinalscale_table.py

SI figure ``rating_ordinalscale.png``: the 41-step ordinal credit-rating scale with the
corresponding Moody's / S&P / Fitch letter grades, split into investment- and
speculative-grade blocks.

Usage (from anywhere inside the repo):
    python3 2_code/supplementary/figures/rating_ordinalscale_table.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SI_OUT_DIR = os.path.join(ROOT, "4_figures", "supplementary")
os.makedirs(SI_OUT_DIR, exist_ok=True)

# Fixed ordinal scale
ordinal_scale = list(reversed([
    1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5,
    11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5,
    20, 20.5, 21
]))

n = len(ordinal_scale)
investment_rows = list(range(0, 19))
speculative_rows = list(range(19, 41))

category = ["" for _ in range(n)]
category[len(investment_rows)//2] = "investment-grade"
category[len(investment_rows) + len(speculative_rows)//2] = "speculative-grade"

data = {
    "category": category,
    "ordinal scale": ordinal_scale,
    "Moody's": [
        "Aaa", "Aaa*/Aa1**", "Aa1", "Aa1*/Aa2**", "Aa2", "Aa2*/Aa3**", "Aa3",
        "Aa3*/A1**", "A1", "A1*/A2**", "A2", "A2*/A3**", "A3", "A3*/Baa1**", "Baa1",
        "Baa1*/Baa2**", "Baa2", "Baa2*/Baa3**", "Baa3", "Baa3*/Ba1**", "Ba1",
        "Ba1*/a2**", "Ba2", "Ba2*/Ba3**", "Ba3", "Ba3*/B1**", "B1", "B1*/B2**", "B2",
        "B2*/B3**", "B3", "B3*/Caa1**", "Caa1", "Caa1*/Caa2**", "Caa2", "Caa2*/Caa3**",
        "Caa3", "Caa3*/Ca**", "Ca", "Ca*", "C"
    ],
    "S&P": [
        "AAA", "AAA*/AA+**", "AA+", "AA+*/AA**", "AA", "AA*/AA-**", "AA-",
        "AA-*/A+**", "A+", "A+*/A**", "A", "A*/A-**", "A-", "A-*/BBB+**",
        "BBB+", "BBB+*/BBB**", "BBB", "BBB*/BBB-**", "BBB-", "BBB-*/BB+**",
        "BB+", "BB+*/BB**", "BB", "BB*/BB-**", "BB-", "BB-*/B+**", "B+",
        "B+*/B**", "B", "B*/B-**", "B-", "B-*/CCC+**", "CCC+",
        "CCC+*/CCC**", "CCC", "CCC*/CCC-**", "CCC-", "CCC-*/CC**", "CC",
        "CC*", "C"
    ],
    "Fitch": [
        "AAA", "AAA*/AA+**", "AA+", "AA+*/AA**", "AA", "AA*/AA-**", "AA-",
        "AA-*/A+**", "A+", "A+*/A**", "A", "A*/A-**", "A-", "A-*/BBB+**",
        "BBB+", "BBB+*/BBB**", "BBB", "BBB*/BBB-**", "BBB-", "BBB-*/BB+**",
        "BB+", "BB+*/BB**", "BB", "BB*/BB-**", "BB-", "BB-*/B+**", "B+",
        "B+*/B**", "B", "B*/B-**", "B-", "B-*/CCC+**", "CCC+",
        "CCC+*/CCC**", "CCC", "CCC*/CCC-**", "CCC-", "CCC-*/CC**", "CC",
        "CC*", "C"
    ]
}


df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(4, 6))
ax.axis('off')

table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

table.scale(1, 0.7)

for (row, col), cell in table.get_celld().items():
    cell.set_fontsize(12)
    cell.set_linewidth(0.5)
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor("#f2f2f2")

# Function to simulate single-border merged block
def format_merged_block(rows):
    start = rows[0] + 1
    end = rows[-1] + 1
    center = rows[len(rows)//2] + 1

    for r in range(start, end + 1):
        cell = table[r, 0]
        cell.get_text().set_text("")
        cell.visible_edges = ""

    for edge_r, edge_sides in [(start, "T"), (end, "B")]:
        table[edge_r, 0].visible_edges = edge_sides + "LR"
    for r in range(start + 1, end):
        table[r, 0].visible_edges = "LR"

    table[center, 0].get_text().set_text(df.iloc[center - 1, 0])
    table[center, 0].set_text_props(weight='bold', ha='center', va='center')

format_merged_block(investment_rows)
format_merged_block(speculative_rows)

out = os.path.join(SI_OUT_DIR, "rating_ordinalscale.png")
plt.savefig(out, bbox_inches='tight', dpi=200)
print("saved", out)
plt.close(fig)
