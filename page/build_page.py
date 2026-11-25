"""Builds page/index.html: a self-contained, offline results page for the IE624 project.

Reads the result files under IE624/runs/ and inlines them, so the page opens by double-click
(no server, no internet). Re-run after any notebook re-run:  python page/build_page.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "IE624", "runs")
OUT = os.path.join(ROOT, "page", "index.html")
TEMPLATE = os.path.join(ROOT, "page", "template.html")


def load(*parts):
    with open(os.path.join(RUNS, *parts), encoding="utf-8") as f:
        return json.load(f)


def r(x, n=3):
    return None if x is None else round(float(x), n)


r2 = load("02_map_and_shift", "reports", "results_02.json")
r3 = load("03_causal_tests", "reports", "results_03.json")
r5 = load("05_judged_grid", "reports", "results_05.json")
proj = load("02_map_and_shift", "page", "projections.json")
atlas = load("05_judged_grid", "page", "atlas.json")
grid = load("05_judged_grid", "page", "grid.json")
agree = load("03_causal_tests", "handlabel", "agreement.json")

cats = r2["categories"]
data = {
    "categories": {
        "cats": cats["cats"],
        "raw": [[r(v) for v in row] for row in cats["cos_matrix_raw_corrected"]],
        "residue": [[r(v) for v in row] for row in cats["cos_matrix_residue_corrected"]],
        "cos_to_A": [r(v) for v in cats["cos_to_A"]],
        "pc1": r(r2["pca"]["pc1_share"][r2["L_star"]]),
        "pc1_null": r(r2["pca"]["null_mean"][r2["L_star"]]),
    },
    "induce": {k: r(v["rate"]) for k, v in r3["partC"].items()},
    "proj": {k: {"x": [round(float(p), 1) for p in v["proj"]], "s": [round(float(s), 2) for s in v["refusal_score"]]}
             for k, v in proj.items()},
    "shift": {
        "cos": [r(v[0]) for v in r2["shift_curves"]["strict-neutral"]],
        "lo": [r(v[0]) if isinstance(v, list) else r(v) for v in r2["ci_lo"]],
        "hi": [r(v[0]) if isinstance(v, list) else r(v) for v in r2["ci_hi"]],
        "headline": {k: (r(v) if not isinstance(v, list) else [r(z) for z in v]) for k, v in r2["headline"].items()},
        "frac_along": [r(v, 4) for v in r3["shift"]["frac_sq_norm_along_r"]],
    },
    "partB": {k: [r(z) for z in v] for k, v in r3["partB"].items()},
    "partB_sweep": {k: [r(z) for z in v] for k, v in r3["partB_layer_sweep"].items()},
    "partA03": {k: r(v) for k, v in r3["partA"]["rates"].items()},
    "judge03": r3["judge_validation"]["overall"],
    "grid": r5["grid"],
    "effects": r5["grid_effects"],
    "harmful": r5["harmful_check"],
    "atlas": atlas,
    "explorer": grid,
    "agreement": agree,
}

with open(TEMPLATE, encoding="utf-8") as f:
    html = f.read()
payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
html = html.replace("/*__DATA__*/null", payload)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")
