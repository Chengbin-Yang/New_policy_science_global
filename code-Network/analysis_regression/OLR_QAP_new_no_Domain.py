# ============================================================================== 
# Climate Policy Similarity Analysis: Ordered Logit + QAP (Clean Version) 
# Notes: 
#   - Error bars (CI) are Wald/model-based, not QAP-adjusted. 
#   - Stars/colors are based on QAP permutation p-values. 
# ==============================================================================

import os
import warnings
from pathlib import Path
from functools import reduce

# Avoid BLAS/OMP oversubscription when using joblib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from joblib import Parallel, delayed


# ------------------------------------------------------------------------------ 
# 1) Config 
# ------------------------------------------------------------------------------ 
class Config:
    BASE_DIR = Path(r"F:/Desktop/科研项目/1.负责科研项目/Climate Policy/CAMPF_Supplementary_V2")
    PATH_X = BASE_DIR / "data/PDF_data_Visual/Long_dataframe"
    PATH_Y_INTENSITY = BASE_DIR / "data/Y_overlapping_cluster_heatmap/Long_dataframe/Intensity"
    PATH_Y_BREADTH = BASE_DIR / "data/Y_overlapping_cluster_heatmap/Long_dataframe/Breadth"
    PATH_OUT = BASE_DIR / "result/OLR_QAP_regression"

    RANDOM_SEED = 42
    N_PERM = 5000
    VIF_THRESH = 5
    N_JOBS = max(1, (os.cpu_count() or 2) - 1)

    # Optional binning if Y has too many distinct values
    Y_QCUT_BINS = None           # e.g., 5
    Y_MAX_LEVELS_BEFORE_BIN = 30

    PLOT_PARAMS = {"font.size": 14, "figure.dpi": 300, "savefig.dpi": 300}

    # Removed Flight_Int, GHG_Sim, Trade_Int
    FILES_X = {
        "Ideal_Sim": "1-1-norm_ideal_sim_2005_2023_long.csv",
        "Geo_Prox": "1-3-norm_geographical_proximity_avg_long.csv",
        "Lang_Off": "1-4-Language_Index_long.csv",
        "GDP_Sim": "1-6-1-GDP_Sim_long.csv",
        "Rent_Sim": "1-6-3-Rent_Sim_long.csv",
        "Fuel_Sim": "1-6-4-Fuel_Ex_Sim_long.csv",
        "Paper_Total": "1-7-All-Paper_collab-Total_long.csv",
    }

    FILES_Y = {
        "Intensity": [
            ("Y_All", "Y_All_long.csv"),
            ("Y_Commitment-based", "Y_Commitment-based_long.csv"),
            ("Y_Incentive-based", "Y_Incentive-based_long.csv"),
            ("Y_Regulatory", "Y_Regulatory_long.csv"),
            ("Y_R&D", "Y_Research and Development (R&D)_long.csv"),
        ],
        "Breadth": [
            ("Y_All", "Y_All_long.csv"),
            ("Y_Commitment-based", "Y_Commitment-based_long.csv"),
            ("Y_Incentive-based", "Y_Incentive-based_long.csv"),
            ("Y_Regulatory", "Y_Regulatory_long.csv"),
            ("Y_R&D", "Y_Research and Development (R&D)_long.csv"),
        ],
    }


Config.PATH_OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(Config.RANDOM_SEED)
plt.rcParams.update(Config.PLOT_PARAMS)


# ------------------------------------------------------------------------------ 
# 2) Data utilities 
# ------------------------------------------------------------------------------ 
def _find_col_case_insensitive(df: pd.DataFrame, names_lower: list[str]) -> str | None:
    mapping = {c.lower().strip(): c for c in df.columns}
    for n in names_lower:
        if n in mapping:
            return mapping[n]
    return None


def _order_pairs(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    try:
        ai = pd.to_numeric(a, errors="raise")
        bi = pd.to_numeric(b, errors="raise")
        src = np.where(ai <= bi, a, b)
        tgt = np.where(ai <= bi, b, a)
    except Exception:
        src = np.where(a <= b, a, b)
        tgt = np.where(a <= b, b, a)
    return pd.Series(src, index=a.index).astype(str), pd.Series(tgt, index=b.index).astype(str)


def load_network_data(path: Path, value_name: str) -> pd.DataFrame:
    if not path.exists():
        print(f"Missing file, skipped: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)

    src_col = "Source" if "Source" in df.columns else _find_col_case_insensitive(df, ["source"])
    tgt_col = "Target" if "Target" in df.columns else _find_col_case_insensitive(df, ["target"])
    val_col = _find_col_case_insensitive(df, ["weight", "value"])

    if not src_col or not tgt_col or not val_col:
        raise ValueError(f"Bad schema in {path.name}: need Source/Target and weight/value columns")

    df = df.rename(columns={src_col: "Source", tgt_col: "Target", val_col: value_name})
    df["Source"] = df["Source"].astype(str)
    df["Target"] = df["Target"].astype(str)

    df["Source"], df["Target"] = _order_pairs(df["Source"], df["Target"])
    df["Key"] = df["Source"] + "_" + df["Target"]

    df = df.drop_duplicates("Key", keep="first")
    return df[["Key", "Source", "Target", value_name]]


def get_master_x() -> pd.DataFrame:
    print(f"Loading X variables (n_jobs={Config.N_JOBS})...")
    dfs = []
    for var, fname in Config.FILES_X.items():
        df = load_network_data(Config.PATH_X / fname, var)
        if df.empty:
            continue
        dfs.append(df[["Key", var]])

    if not dfs:
        raise ValueError("No X variables loaded. Check paths and files.")

    df_x = reduce(lambda l, r: pd.merge(l, r, on="Key", how="inner"), dfs).dropna()
    print(f"X master loaded: rows={len(df_x)}, cols={df_x.shape[1]}")
    return df_x


# ------------------------------------------------------------------------------ 
# 3) Stats: VIF filter + Ordered Logit + QAP 
# ------------------------------------------------------------------------------ 
def filter_vif(df: pd.DataFrame, x_cols: list[str]) -> list[str]:
    keep = [c for c in x_cols if c in df.columns]
    while True:
        if len(keep) <= 1:
            print(f"VIF stop: remaining vars={keep}")
            return keep

        X = sm.add_constant(df[keep])
        vifs = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])],
            index=keep,
        ).replace([np.inf, -np.inf], np.nan)

        vmax = vifs.max(skipna=True)
        if pd.isna(vmax) or vmax <= Config.VIF_THRESH:
            break

        drop_var = vifs.idxmax()
        keep.remove(drop_var)
        print(f"Drop high VIF: {drop_var} (max VIF={vmax:.2f})")

    print(f"VIF cleaned: kept {len(keep)} vars")
    return keep


def _make_ordered_endog(y: pd.Series) -> pd.Categorical:
    y_clean = y.dropna()

    if Config.Y_QCUT_BINS is not None:
        nun = y_clean.nunique()
        if nun > Config.Y_MAX_LEVELS_BEFORE_BIN:
            y_bin = pd.qcut(y_clean, q=Config.Y_QCUT_BINS, duplicates="drop", labels=False)
            cats = sorted(pd.unique(y_bin))
            return pd.Categorical(y_bin, categories=cats, ordered=True)

    if pd.api.types.is_numeric_dtype(y_clean):
        cats = np.sort(pd.unique(y_clean))
        return pd.Categorical(y, categories=cats, ordered=True)

    cats = list(pd.unique(y_clean))
    return pd.Categorical(y, categories=cats, ordered=True)


def _fit_ordered_logit(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    df = df.dropna(subset=[y_col] + x_cols).copy()

    if df[y_col].nunique(dropna=True) < 2:
        raise ValueError(f"{y_col}: not enough variation to fit OrderedModel")

    df["Y_Fact"] = _make_ordered_endog(df[y_col])

    formula = "Y_Fact ~ " + " + ".join(x_cols)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        model = OrderedModel.from_formula(formula, df, distr="logit")
        res = model.fit(disp=0, method="bfgs", maxiter=200)
    return res


def _qap_worker(seed: int, df_x_small: pd.DataFrame, df_y_small: pd.DataFrame, y_col: str, x_cols: list[str], nodes: list[str]):
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(nodes)
    node_map = dict(zip(nodes, shuffled))

    y_perm = df_y_small.copy()
    y_perm["Source"] = y_perm["Source"].map(node_map).astype(str)
    y_perm["Target"] = y_perm["Target"].map(node_map).astype(str)

    y_perm["Source"], y_perm["Target"] = _order_pairs(y_perm["Source"], y_perm["Target"])
    y_perm["Key"] = y_perm["Source"] + "_" + y_perm["Target"]
    y_perm = y_perm.drop_duplicates("Key", keep="first")

    df_perm = pd.merge(df_x_small, y_perm[["Key", y_col]], on="Key", how="inner")
    if len(df_perm) == 0:
        return None

    try:
        res = _fit_ordered_logit(df_perm, y_col, x_cols)
        return res.params[x_cols]
    except Exception:
        return None


def _sig_code(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"


def run_qap(df_x: pd.DataFrame, df_y: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    df_x_small = df_x[["Key"] + x_cols].dropna().copy()
    df_y_small = df_y[["Key", "Source", "Target", y_col]].dropna().copy()

    nodes = sorted(set(df_y_small["Source"]).union(set(df_y_small["Target"])))


    print("Fitting real model...")
    df_real = pd.merge(df_x_small, df_y_small[["Key", y_col]], on="Key", how="inner")
    res_real = _fit_ordered_logit(df_real, y_col, x_cols)

    beta_obs = res_real.params[x_cols]
    conf_int = res_real.conf_int().loc[x_cols]

    # Wald p-values (model-based, may not be network-robust)
    try:
        wald_p = res_real.pvalues.loc[x_cols].astype(float)
    except Exception:
        from scipy.stats import norm
        z = (res_real.params[x_cols] / res_real.bse[x_cols]).astype(float)
        wald_p = (2 * (1 - norm.cdf(np.abs(z)))).astype(float)

    print(f"Running QAP permutations: N={Config.N_PERM}, n_jobs={Config.N_JOBS}")
    master_rng = np.random.default_rng(Config.RANDOM_SEED)
    seeds = master_rng.integers(0, 1_000_000, size=Config.N_PERM)

    perm_list = Parallel(n_jobs=Config.N_JOBS, verbose=5)(delayed(_qap_worker)(int(s), df_x_small, df_y_small, y_col, x_cols, nodes) for s in seeds)

    perm_list = [p for p in perm_list if p is not None]
    if len(perm_list) == 0:
        raise RuntimeError("All permutations failed. Consider fewer X vars or binning Y.")

    perm_df = pd.DataFrame(perm_list)

    rows = []
    for x in x_cols:
        obs = float(beta_obs[x])
        perms = perm_df[x].to_numpy()

        qap_p = (np.sum(np.abs(perms) >= np.abs(obs)) + 1) / (len(perms) + 1)

        ci_low = float(conf_int.loc[x][0])
        ci_high = float(conf_int.loc[x][1])
        crosses0 = (ci_low <= 0.0 <= ci_high)

        rows.append(
            {
                "Variable": x,
                "Coef": obs,
                "CI_Low": ci_low,
                "CI_High": ci_high,
                "CI_Cross0": bool(crosses0),
                "Wald_P": float(wald_p[x]),
                "Wald_Sig": _sig_code(float(wald_p[x])),
                "QAP_P": float(qap_p),
                "QAP_Sig": _sig_code(float(qap_p)),
                "Perm_OK": int(len(perms)),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------ 
# 4) Plot 
# ------------------------------------------------------------------------------ 
def plot_forest(res_df: pd.DataFrame, title: str, out_path: Path):
    df = res_df.sort_values("Coef", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvline(0, linestyle="--", color="gray", alpha=0.5)

    y_pos = np.arange(len(df))
    xerr = np.vstack([df["Coef"] - df["CI_Low"], df["CI_High"] - df["Coef"]])
    ax.errorbar(df["Coef"], y_pos, xerr=xerr, fmt="none", ecolor="black", capsize=3)

    color_map = {"***": "#E41A1C", "**": "#FF7F00", "*": "#377EB8", "ns": "gray"}
    colors = [color_map.get(s, "gray") for s in df["QAP_Sig"]]
    ax.scatter(df["Coef"], y_pos, c=colors, s=120, zorder=5)

    for i, row in df.iterrows():
        ax.text(row["CI_High"] + 0.05, i, row["QAP_Sig"], va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Variable"])
    ax.set_xlabel("Ordered Logit Coefficient")
    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ------------------------------------------------------------------------------ 
# 5) Main 
# ------------------------------------------------------------------------------ 
def process_category(category: str, df_x_master: pd.DataFrame):
    if category not in Config.FILES_Y:
        raise ValueError(f"Unknown category: {category}")

    base_path = Config.PATH_Y_INTENSITY if category == "Intensity" else Config.PATH_Y_BREADTH
    file_list = Config.FILES_Y[category]
    x_cols_all = list(Config.FILES_X.keys())

    for y_name, fname in file_list:
        print("-" * 60)
        print(f"Category={category}, Y={y_name}")
        print("-" * 60)

        full_path = base_path / fname
        if not full_path.exists():
            print(f"Missing Y file, skipped: {full_path}")
            continue

        df_y = load_network_data(full_path, y_name)
        if df_y.empty:
            print("Y data is empty after loading, skipped.")
            continue

        df_full = pd.merge(df_x_master, df_y[["Key", y_name]], on="Key", how="inner").dropna()
        print(f"Aligned sample size: {len(df_full)}")
        print(f"Y distinct levels: {df_full[y_name].nunique(dropna=True)}")

        final_x = filter_vif(df_full, x_cols_all)
        if len(final_x) == 0:
            print("No X variables left after VIF filtering, skipped.")
            continue

        res_df = run_qap(df_x_master, df_y, y_name, final_x)

        out_csv = Config.PATH_OUT / f"Result_{category}_{y_name}.csv"
        out_png = Config.PATH_OUT / f"Forest_{category}_{y_name}.png"

        res_df.to_csv(out_csv, index=False)
        plot_forest(res_df, f"Drivers of {y_name} ({category})", out_png)

        print(f"Saved: {out_csv}")
        print(f"Saved: {out_png}")


def main():
    try:
        df_x_master = get_master_x()
        process_category("Intensity", df_x_master)
        process_category("Breadth", df_x_master)
        print("All tasks completed.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
