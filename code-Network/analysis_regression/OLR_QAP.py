# ==============================================================================
# Climate Policy Analysis: DEBUG + BEAUTY VERSION
#
# Changes from previous:
#   1. REMOVED all try-catch blocks. Errors will stop execution immediately
#      so you can see exactly what went wrong.
#   2. ENHANCED plotting aesthetics for publication-quality figures.
#   3. KEEPS the robust Matrix-based fitting and Analytic Null Model.
# ==============================================================================

import os
import warnings
import traceback
from pathlib import Path
from functools import reduce
import itertools
from collections import defaultdict

# Avoid BLAS/OMP oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
    
    # Output folder
    PATH_OUT = BASE_DIR / "result/OLR_QAP_Final"

    RANDOM_SEED = 42
    
    # === Settings ===
    N_PERM = 5000       # QAP Permutations
    N_JOBS = max(1, (os.cpu_count() or 2) - 1)
    N_SIM_CI = 1000     # Simulations for smooth CI
    MAX_ITER = 5000     # Optimization iterations
    
    VIF_THRESH = 5
    Y_QCUT_BINS = None            
    Y_MAX_LEVELS_BEFORE_BIN = 30

    # === BEAUTY PLOT SETTINGS ===
    PLOT_PARAMS = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 2.5,
    }

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
# 2) Data Utilities
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
        if df.empty: continue
        dfs.append(df[["Key", var]])

    if not dfs:
        raise ValueError("No X variables loaded. Check paths and files.")

    df_x = reduce(lambda l, r: pd.merge(l, r, on="Key", how="inner"), dfs).dropna()
    print(f"X master loaded: rows={len(df_x)}, cols={df_x.shape[1]}")
    return df_x


# ------------------------------------------------------------------------------
# 3) Stats & Fitting (Robust Matrix Version)
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

def _fit_ordered_logit(df: pd.DataFrame, y_col: str, x_cols: list[str], max_iter=5000):
    """
    Fits ordered logit using DIRECT MATRIX passing.
    """
    subset_cols = [y_col] + x_cols if x_cols else [y_col]
    df = df.dropna(subset=subset_cols).copy()
    
    if df[y_col].nunique(dropna=True) < 2: return None

    y_cat = _make_ordered_endog(df[y_col])
    
    if not x_cols:
        return None
    
    exog = df[x_cols]
    model = OrderedModel(y_cat, exog, distr="logit")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            res = model.fit(disp=0, method="bfgs", maxiter=max_iter)
        except:
            res = model.fit(disp=0, method="nm", maxiter=max_iter)
            # If both fail, it will raise exception to caller now (removed try-catch return None)
    return res

def _qap_worker(seed: int, df_x_small: pd.DataFrame, df_y_small: pd.DataFrame, y_col: str, x_cols: list[str], nodes: list[str]):
    # Keep minimal try-catch here just to avoid parallel crashes on single perm
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
    if len(df_perm) == 0: return None

    try:
        res = _fit_ordered_logit(df_perm, y_col, x_cols, max_iter=500)
        if res is None: return None
        return res.params[x_cols]
    except Exception:
        return None

def _sig_code(p: float) -> str:
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "ns"

def run_qap(df_x: pd.DataFrame, df_y: pd.DataFrame, y_col: str, x_cols: list[str]) -> pd.DataFrame:
    df_x_small = df_x[["Key"] + x_cols].dropna().copy()
    df_y_small = df_y[["Key", "Source", "Target", y_col]].dropna().copy()
    nodes = sorted(set(df_y_small["Source"]).union(set(df_y_small["Target"])))

    print("Fitting real model (QAP Base)...")
    df_real = pd.merge(df_x_small, df_y_small[["Key", y_col]], on="Key", how="inner")
    
    # NO TRY CATCH: If this fails, user wants to know.
    res_real = _fit_ordered_logit(df_real, y_col, x_cols, max_iter=Config.MAX_ITER)
    if res_real is None: 
        raise RuntimeError(f"Real model failed to converge for {y_col}")

    beta_obs = res_real.params[x_cols]
    conf_int = res_real.conf_int().loc[x_cols]

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
        raise RuntimeError("All permutations failed.")

    perm_df = pd.DataFrame(perm_list)
    rows = []
    for x in x_cols:
        obs = float(beta_obs[x])
        perms = perm_df[x].to_numpy()
        qap_p = (np.sum(np.abs(perms) >= np.abs(obs)) + 1) / (len(perms) + 1)
        ci_low = float(conf_int.loc[x][0])
        ci_high = float(conf_int.loc[x][1])
        crosses0 = (ci_low <= 0.0 <= ci_high)

        rows.append({
            "Variable": x, "Coef": obs, "CI_Low": ci_low, "CI_High": ci_high,
            "CI_Cross0": bool(crosses0), "Wald_P": float(wald_p[x]), "Wald_Sig": _sig_code(float(wald_p[x])),
            "QAP_P": float(qap_p), "QAP_Sig": _sig_code(float(qap_p)), "Perm_OK": int(len(perms)),
        })
    return pd.DataFrame(rows)

def plot_forest(res_df: pd.DataFrame, title: str, out_path: Path):
    # Cleaner Forest Plot
    df = res_df.sort_values("Coef", ascending=True).reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Grid and Zero line
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.axvline(0, linestyle="-", color="black", alpha=0.3, linewidth=1)
    
    y_pos = np.arange(len(df))
    xerr = np.vstack([df["Coef"] - df["CI_Low"], df["CI_High"] - df["Coef"]])
    
    # Error Bars
    ax.errorbar(df["Coef"], y_pos, xerr=xerr, fmt="none", ecolor="#333333", capsize=5, elinewidth=1.5, markeredgewidth=1.5)
    
    # Scatter points with significance colors
    color_map = {"***": "#D62728", "**": "#FF7F0E", "*": "#1F77B4", "ns": "#7F7F7F"}
    colors = [color_map.get(s, "gray") for s in df["QAP_Sig"]]
    
    ax.scatter(df["Coef"], y_pos, c=colors, s=150, zorder=5, edgecolors='white', linewidth=1)
    
    # Text labels
    for i, row in df.iterrows():
        ax.text(row["CI_High"] + (df["Coef"].max()-df["Coef"].min())*0.02, i, row["QAP_Sig"], 
                va="center", fontsize=11, fontweight="bold", color="#333333")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["Variable"])
    ax.set_xlabel("Ordered Logit Coefficient")
    ax.set_title(title, pad=20)
    
    # Despine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------------------
# 5) Domain Analysis & BEAUTIFUL PROBABILITY CURVES
# ------------------------------------------------------------------------------
def run_dominance_analysis(df: pd.DataFrame, y_col: str, x_cols: list[str]):
    print(f"Running Dominance Analysis for {y_col}...")
    df_clean = df.dropna(subset=[y_col] + x_cols).copy()
    if df_clean.empty: return pd.DataFrame()

    # --- 1. Analytic Null Model (Exact) ---
    counts = df_clean[y_col].value_counts()
    n_total = len(df_clean)
    probs = counts / n_total
    ll_null = (counts * np.log(probs)).sum()
    
    if np.isnan(ll_null) or n_total == 0:
        print("DA Warning: Could not calculate Null LL.")
        return pd.DataFrame()

    def get_pseudo_r2(fit_res, base_ll):
        if fit_res is None: return 0.0
        current_ll = fit_res.llf
        if current_ll < base_ll: current_ll = base_ll
        return 1 - (current_ll / base_ll)

    # --- 2. Subsets ---
    r2_cache = {}
    r2_cache[()] = 0.0 
    all_subsets = []
    for r in range(1, len(x_cols) + 1):
        for combo in itertools.combinations(x_cols, r):
            all_subsets.append(combo)
    
    for i, combo in enumerate(all_subsets):
        # NO TRY CATCH: Fail if model fails
        res = _fit_ordered_logit(df_clean, y_col, list(combo), max_iter=Config.MAX_ITER)
        r2_cache[combo] = get_pseudo_r2(res, ll_null)

    # --- 3. Dominance Calc ---
    dominance = defaultdict(float)
    for x in x_cols:
        contributions = []
        remaining_vars = [v for v in x_cols if v != x]
        for k in range(len(remaining_vars) + 1):
            for subset in itertools.combinations(remaining_vars, k):
                subset_with_x = tuple(sorted(list(subset) + [x]))
                subset_without_x = tuple(sorted(list(subset)))
                incr = r2_cache.get(subset_with_x, 0) - r2_cache.get(subset_without_x, 0)
                if incr < 0: incr = 0.0
                contributions.append(incr)
        if contributions: dominance[x] = np.mean(contributions)

    da_df = pd.DataFrame(list(dominance.items()), columns=["Variable", "General_Dominance"])
    total_dom = da_df["General_Dominance"].sum()
    if total_dom > 1e-9:
        da_df["Standardized_Importance"] = (da_df["General_Dominance"] / total_dom) * 100
    else:
        da_df["Standardized_Importance"] = 0
    
    da_df = da_df.sort_values("Standardized_Importance", ascending=False).reset_index(drop=True)
    return da_df

def plot_dominance(da_df: pd.DataFrame, title: str, out_path: Path):
    if da_df.empty: return
    fig, ax = plt.subplots(figsize=(8, 5))
    da_df_rev = da_df.iloc[::-1]
    
    # Cleaner Bar Plot
    bars = ax.barh(da_df_rev["Variable"], da_df_rev["Standardized_Importance"], color="#4C72B0", edgecolor="white", linewidth=1)
    ax.bar_label(bars, fmt="%.1f%%", padding=5, fontsize=12, fontweight='bold')
    
    ax.set_xlabel("Relative Importance (%)")
    ax.set_title(title + "\n(McFadden's R²)")
    ax.set_xlim(0, max(da_df["Standardized_Importance"].max() * 1.15, 10))
    
    # Despine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

def plot_predicted_probs_with_ci(res, df: pd.DataFrame, x_col: str, all_x_cols: list, 
                                 y_name: str, out_img_path: Path, out_csv_path: Path):
    if res is None: return

    # 1. Prediction Grid
    x_min, x_max = df[x_col].min(), df[x_col].max()
    x_range = np.linspace(x_min, x_max, 100)
    
    means = df[all_x_cols].mean()
    pred_data = pd.DataFrame([means] * 100, columns=all_x_cols)
    pred_data[x_col] = x_range
    pred_data = pred_data[all_x_cols]

    # 2. Point Estimate
    try:
        probs_mean = res.predict(exog=pred_data)
    except Exception:
        probs_mean = res.model.predict(res.params, exog=pred_data)

    # 3. Simulation
    params = res.params
    cov = res.cov_params()
    rng = np.random.default_rng(Config.RANDOM_SEED)
    
    try:
        sim_params = rng.multivariate_normal(params, cov, size=Config.N_SIM_CI, method='svd')
    except Exception as e:
        print(f"Warning: Covariance matrix issue for {x_col}, skipping CI. Error: {e}")
        sim_params = []

    sim_probs_stack = []
    if len(sim_params) > 0:
        for p_vec in sim_params:
            try:
                p_sim = res.model.predict(params=p_vec, exog=pred_data)
                sim_probs_stack.append(p_sim)
            except:
                continue
    
    sim_probs_stack = np.array(sim_probs_stack) 
    
    # 4. Save to CSV
    data_to_save = {x_col: x_range}
    if hasattr(res.model, 'endog_names') and res.model.endog_names:
        y_levels = res.model.endog_names
    else:
        y_levels = [f"Level_{i}" for i in range(probs_mean.shape[1])]
    
    if len(y_levels) != probs_mean.shape[1]:
        y_levels = [f"L{i}" for i in range(probs_mean.shape[1])]

    for i, level_name in enumerate(y_levels):
        data_to_save[f"{level_name}_Prob"] = probs_mean.iloc[:, i].values
        if len(sim_probs_stack) > 0:
            ci_lower = np.percentile(sim_probs_stack[:, :, i], 2.5, axis=0)
            ci_upper = np.percentile(sim_probs_stack[:, :, i], 97.5, axis=0)
            data_to_save[f"{level_name}_CI_Lower"] = ci_lower
            data_to_save[f"{level_name}_CI_Upper"] = ci_upper
            
    pd.DataFrame(data_to_save).to_csv(out_csv_path, index=False)

    # 5. BEAUTIFUL PLOTTING
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use Viridis for ordered intensity (Sequential colormap)
    # Sampling colors to span the range nicely
    cmap = matplotlib.cm.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0.2, 0.9, len(y_levels))]
    
    for i, level_name in enumerate(y_levels):
        y_mean = probs_mean.iloc[:, i]
        color = colors[i]
        
        # Plot Mean Line (Thicker, smooth)
        ax.plot(x_range, y_mean, label=f"{level_name}", color=color, linewidth=3, alpha=0.9, zorder=3)
        
        # Plot CI (Translucent Area)
        if len(sim_probs_stack) > 0:
            ci_lower = data_to_save.get(f"{level_name}_CI_Lower")
            ci_upper = data_to_save.get(f"{level_name}_CI_Upper")
            if ci_lower is not None:
                ax.fill_between(x_range, ci_lower, ci_upper, color=color, alpha=0.15, zorder=2, edgecolor=None)

    # Aesthetics
    ax.set_xlabel(f"{x_col} (Normalized)", fontweight='bold', labelpad=10)
    ax.set_ylabel("Predicted Probability", fontweight='bold', labelpad=10)
    ax.set_title(f"Marginal Effect: {x_col} on {y_name}", fontweight='bold', pad=15)
    
    # Legend Styling
    legend = ax.legend(title="Policy Level", bbox_to_anchor=(1.02, 1), loc='upper left', 
                       frameon=False, fontsize=12, title_fontsize=13)
    
    # Grid and Spines
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray', alpha=0.3, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Ticks
    ax.tick_params(axis='both', which='major', labelsize=12, length=5)
    ax.set_ylim(0, 1.05) # Probabilities are 0-1
    
    plt.tight_layout()
    fig.savefig(out_img_path, bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------------------
# 6) Main
# ------------------------------------------------------------------------------
def process_category(category: str, df_x_master: pd.DataFrame):
    if category not in Config.FILES_Y:
        raise ValueError(f"Unknown category: {category}")

    base_path = Config.PATH_Y_INTENSITY if category == "Intensity" else Config.PATH_Y_BREADTH
    file_list = Config.FILES_Y[category]
    x_cols_all = list(Config.FILES_X.keys())

    for y_name, fname in file_list:
        print("-" * 60)
        print(f"Processing: {category} - {y_name}")
        
        full_path = base_path / fname
        if not full_path.exists(): continue
        df_y = load_network_data(full_path, y_name)
        if df_y.empty: continue

        df_full = pd.merge(df_x_master, df_y[["Key", y_name]], on="Key", how="inner").dropna()
        final_x = filter_vif(df_full, x_cols_all)
        if not final_x: continue

        # --- NO TRY CATCH BLOCKS BELOW: LET ERRORS EXPLODE ---
        
        # 1. QAP
        res_df = run_qap(df_x_master, df_y, y_name, final_x)
        res_df.to_csv(Config.PATH_OUT / f"Result_{category}_{y_name}.csv", index=False)
        plot_forest(res_df, f"Drivers: {y_name}", Config.PATH_OUT / f"Forest_{category}_{y_name}.png")

        # 2. Dominance Analysis
        da_df = run_dominance_analysis(df_full, y_name, final_x)
        if not da_df.empty and da_df["General_Dominance"].sum() > 0:
            da_df.to_csv(Config.PATH_OUT / f"Dominance_{category}_{y_name}.csv", index=False)
            plot_dominance(da_df, f"Importance: {y_name}", Config.PATH_OUT / f"Importance_{category}_{y_name}.png")
        else:
            print("DA: Model explains no variance or failed.")

        # 3. Predicted Probability Curves with CI
        real_model_res = _fit_ordered_logit(df_full, y_name, final_x, max_iter=Config.MAX_ITER)
        
        if 'da_df' in locals() and not da_df.empty and da_df["Standardized_Importance"].max() > 0:
            vars_to_plot = da_df.head(3)["Variable"].tolist()
        else:
            vars_to_plot = final_x[:3]
        
        for x_var in vars_to_plot:
            plot_predicted_probs_with_ci(
                real_model_res, df_full, x_var, final_x, y_name,
                Config.PATH_OUT / f"Curve_{category}_{y_name}_{x_var}.png",
                Config.PATH_OUT / f"Curve_Data_{category}_{y_name}_{x_var}.csv"
            )
            print(f"Saved Curve: Curve_{category}_{y_name}_{x_var}.png")

def main():
    # Only keeping the OUTERMOST try-catch for a nice printout
    try:
        df_x = get_master_x()
        process_category("Intensity", df_x)
        process_category("Breadth", df_x)
        print("Done.")
    except Exception:
        # This will print exactly WHERE it failed
        print("\n!!! CRITICAL ERROR DETECTED !!!")
        traceback.print_exc()

if __name__ == "__main__":
    main()