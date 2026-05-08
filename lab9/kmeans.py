"""
K-means clustering from scratch (no scikit-learn) on used-car data.

Sections map to lab questions Qn-1 … Qn-11.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------------------------------------------------------
# Qn–1: Data Preprocessing
# -----------------------------------------------------------------------------

DATA_PATH = Path(__file__).resolve().parent / "car_data.csv"


def load_and_preprocess(csv_path: Path) -> pd.DataFrame:
    """
    Load dataset from CSV, drop irrelevant columns, handle missing values.

    Drops: car name, selling price (per assignment).
    """
    df = pd.read_csv(csv_path)

    # Normalize column names for robust matching (handles Car_Name vs car name, etc.)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    name_cols = [c for c in df.columns if "name" in c and "car" in c]
    if not name_cols:
        name_cols = [c for c in df.columns if c in ("car_name", "name")]
    price_cols = [c for c in df.columns if "selling" in c and "price" in c]
    if not price_cols:
        price_cols = [c for c in df.columns if c in ("selling_price",)]

    drop_cols = list(dict.fromkeys(name_cols + price_cols))
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Missing values: numeric → median, categorical/text → mode
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else "missing"
                df[col] = df[col].fillna(fill)

    return df


# -----------------------------------------------------------------------------
# Qn–2: Handling Categorical Data
# -----------------------------------------------------------------------------
# K-means uses Euclidean distance on numbers. Categorical labels have no natural
# numeric distance unless encoded. Using `.astype("category").cat.codes` maps each
# distinct category to an integer 0..k-1 so distances are defined. This is a simple
# ordinal/nominal encoding (not one-hot); it assumes ordered spacing between codes
# which is not always theoretically ideal for nominal data, but satisfies the lab
# hint and keeps the feature vector compact for distance-based clustering.


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert fuel, seller_type, transmission (and owner if non-numeric) to codes."""
    df = df.copy()

    col_aliases = {
        "fuel": ["fuel", "fuel_type"],
        "seller_type": ["seller_type", "seller"],
        "transmission": ["transmission"],
        "owner": ["owner"],
    }
    resolved = {}
    for logical, candidates in col_aliases.items():
        for c in candidates:
            if c in df.columns:
                resolved[logical] = c
                break

    for logical, col in resolved.items():
        s = df[col].astype(str).str.strip()
        df[col] = s.astype("category").cat.codes

    return df


# -----------------------------------------------------------------------------
# Qn–3: Feature Selection
# -----------------------------------------------------------------------------

FEATURE_KEYS = ("year", "km_driven", "fuel", "transmission", "owner")


def resolve_feature_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map logical names to actual column names in the dataframe."""
    lower = {c.lower(): c for c in df.columns}
    mapping: dict[str, str] = {}

    year_c = next((lower[k] for k in ("year",) if k in lower), None)
    km_c = next(
        (lower[k] for k in ("km_driven", "kms_driven", "kilometers_driven") if k in lower),
        None,
    )
    fuel_c = next((lower[k] for k in ("fuel", "fuel_type") if k in lower), None)
    trans_c = next((lower[k] for k in ("transmission",) if k in lower), None)
    owner_c = next((lower[k] for k in ("owner",) if k in lower), None)

    if year_c:
        mapping["year"] = year_c
    if km_c:
        mapping["km_driven"] = km_c
    if fuel_c:
        mapping["fuel"] = fuel_c
    if trans_c:
        mapping["transmission"] = trans_c
    if owner_c:
        mapping["owner"] = owner_c

    missing = [k for k in FEATURE_KEYS if k not in mapping]
    if missing:
        raise ValueError(f"Could not resolve columns for: {missing}. Found: {list(df.columns)}")

    return mapping


def select_feature_matrix(df: pd.DataFrame, col_map: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    ordered_cols = [col_map[k] for k in FEATURE_KEYS]
    X_df = df[ordered_cols].copy()
    X_df.columns = list(FEATURE_KEYS)
    return X_df, list(FEATURE_KEYS)


# -----------------------------------------------------------------------------
# Qn–4: Feature Scaling
# -----------------------------------------------------------------------------
# Features have different units/scales (year ~ thousands, km_driven ~ tens of
# thousands). Without scaling, large-magnitude features dominate Euclidean distance,
# biasing clusters toward those dimensions. Min–max normalization puts each feature
# on a comparable [0, 1] range so each contributes fairly to distance.


def minmax_fit_transform(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    denom = maxs - mins
    denom[denom == 0] = 1.0
    X_scaled = (X - mins) / denom
    return X_scaled.astype(np.float64), mins, maxs


def minmax_inverse_transform(X_scaled: np.ndarray, mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
    return X_scaled * (maxs - mins) + mins


# -----------------------------------------------------------------------------
# Qn–5–8: K-means (initialize, assign, update, iterate until convergence)
# -----------------------------------------------------------------------------


def _euclid2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances from each row of a to vector b."""
    return np.sum((a - b) ** 2, axis=1)


def kmeans(
    X: np.ndarray,
    k: int,
    *,
    max_iter: int = 500,
    tol: float = 1e-6,
    random_state: int | None = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    K-means with random initialization from distinct data rows.

    Returns (cluster_labels, centroids, num_iterations).
    """
    rng = np.random.default_rng(random_state)
    n, d = X.shape

    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, n_samples]; got k={k}, n={n}")

    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx].copy()

    labels = np.zeros(n, dtype=np.int64)

    for it in range(max_iter):
        # Qn–6: assignment
        dists = np.empty((n, k))
        for j in range(k):
            dists[:, j] = _euclid2(X, centroids[j])
        labels = np.argmin(dists, axis=1)

        # Qn–7: update
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                # empty cluster: re-seed at a random data point
                new_centroids[j] = X[rng.integers(n)]
            else:
                new_centroids[j] = X[mask].mean(axis=0)

        # Qn–8: stop when centroids stabilize
        if np.allclose(new_centroids, centroids, atol=tol, rtol=0):
            return labels, new_centroids, it + 1

        centroids = new_centroids

    return labels, centroids, max_iter


# -----------------------------------------------------------------------------
# Qn–9: Evaluation — Within-Cluster Sum of Squares (WCSS)
# -----------------------------------------------------------------------------


def compute_wcss(X: np.ndarray, clusters: np.ndarray, centroids: np.ndarray) -> float:
    total = 0.0
    k = centroids.shape[0]
    for j in range(k):
        mask = clusters == j
        if not np.any(mask):
            continue
        diff = X[mask] - centroids[j]
        total += float(np.sum(diff**2))
    return total


def kmeans_wcss_for_k(X: np.ndarray, k: int, random_state: int = 42) -> float:
    labels, centroids, _ = kmeans(X, k, random_state=random_state)
    return compute_wcss(X, labels, centroids)


# -----------------------------------------------------------------------------
# Qn–10: Visualization helpers
# -----------------------------------------------------------------------------


def _maybe_show() -> None:
    """Show plot when an interactive backend is available; always close figure."""
    backend = matplotlib.get_backend().lower()
    if backend not in ("agg", "canvasagg"):
        plt.show()
    plt.close()


def plot_clusters_2d(
    year_orig: np.ndarray,
    km_orig: np.ndarray,
    labels: np.ndarray,
    centroids_scaled: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    *,
    year_idx: int,
    km_idx: int,
    out_path: Path | None = None,
) -> None:
    """Scatter km_driven (x) vs year (y); centroids transformed back to original scale."""
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(
        km_orig,
        year_orig,
        c=labels,
        cmap="tab10",
        vmin=0,
        vmax=max(int(labels.max()), 2),
        alpha=0.65,
        edgecolors="k",
        s=28,
    )

    c_full = minmax_inverse_transform(centroids_scaled, mins, maxs)
    plt.scatter(
        c_full[:, km_idx],
        c_full[:, year_idx],
        c="red",
        marker="X",
        s=220,
        linewidths=2,
        label="Centroids",
        zorder=5,
    )

    plt.xlabel("km_driven")
    plt.ylabel("year")
    plt.title("K-means clusters (km_driven vs year)")
    plt.colorbar(sc, label="cluster")
    plt.legend()
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    _maybe_show()


def plot_elbow(k_values: list[int], wcss_values: list[float], out_path: Path | None = None) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss_values, "bo-", linewidth=2, markersize=8)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("WCSS")
    plt.title("Elbow plot (Within-Cluster Sum of Squares)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    _maybe_show()


# -----------------------------------------------------------------------------
# Qn–11: Analysis (interpretation prints in main)
# -----------------------------------------------------------------------------


def interpret_clusters(
    year_mean: np.ndarray,
    km_mean: np.ndarray,
) -> tuple[int | None, int | None]:
    """
    Heuristic labels: 'premium' ~ newer + lower km; 'older' ~ lowest mean year.
    Returns (idx_premium, idx_older) cluster indices.
    """
    # Older cars: lowest average year
    idx_older = int(np.argmin(year_mean))

    # Premium proxy: high year and low km (min-max composite score)
    y_norm = (year_mean - year_mean.min()) / (year_mean.max() - year_mean.min() + 1e-9)
    km_inv = 1.0 - (km_mean - km_mean.min()) / (km_mean.max() - km_mean.min() + 1e-9)
    score = y_norm + km_inv
    idx_premium = int(np.argmax(score))

    return idx_premium, idx_older


def main() -> None:
    print("=== Qn–1: Load & preprocess ===")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Place your CSV at {DATA_PATH} (expected columns include year, km driven, fuel, etc.)."
        )

    df_raw = load_and_preprocess(DATA_PATH)
    print(f"Rows: {len(df_raw)}, columns after drop: {list(df_raw.columns)}")

    print("\n=== Qn–2: Encode categoricals ===")
    df_enc = encode_categoricals(df_raw)
    print(
        "Why numeric codes: K-means minimizes Euclidean distance in ℝⁿ. Raw strings have no "
        "coordinates, so each category is mapped to an integer code. Distance between codes is "
        "then defined (note: for nominal categories the numeric gaps are arbitrary unless you "
        "use one-hot or domain-specific embeddings)."
    )

    print("\n=== Qn–3: Feature selection ===")
    col_map = resolve_feature_columns(df_enc)
    X_df, feature_names = select_feature_matrix(df_enc, col_map)
    print(f"Features used: {feature_names}")

    print("\n=== Qn–4: Scaling (min–max) ===")
    X = X_df.to_numpy(dtype=np.float64)
    X_scaled, mins, maxs = minmax_fit_transform(X)
    print(
        "Scaling maps each feature to [0, 1] using column min/max so distance treats "
        "year and km_driven on comparable footing."
    )

    K = 3
    print(f"\n=== Qn–5–8: K-means (K={K}, init = random rows) ===")
    labels, centroids, iters = kmeans(X_scaled, K, random_state=42)
    print(f"Converged in {iters} iteration(s).")

    print("\n=== Expected output: cluster assignments (first 20) ===")
    print(labels[:20])

    print("\n=== Centroids (scaled feature space) ===")
    print(pd.DataFrame(centroids, columns=feature_names))

    centroids_orig = minmax_inverse_transform(centroids, mins, maxs)
    print("\n=== Centroids (original units) ===")
    print(pd.DataFrame(centroids_orig, columns=feature_names))

    wcss_k3 = compute_wcss(X_scaled, labels, centroids)
    print(f"\n=== Qn–9: WCSS (k={K}) === {wcss_k3:.4f}")

    print("\n=== Elbow: WCSS for k = 1..10 ===")
    ks = list(range(1, 11))
    wcss_list: list[float] = []
    for k in ks:
        w = kmeans_wcss_for_k(X_scaled, k, random_state=42)
        wcss_list.append(w)
        print(f"  k={k:2d}  WCSS={w:.4f}")

    out_dir = Path(__file__).resolve().parent
    plot_elbow(ks, wcss_list, out_path=out_dir / "elbow_plot.png")
    plot_clusters_2d(
        X_df["year"].to_numpy(),
        X_df["km_driven"].to_numpy(),
        labels,
        centroids,
        mins,
        maxs,
        year_idx=feature_names.index("year"),
        km_idx=feature_names.index("km_driven"),
        out_path=out_dir / "clusters_km_vs_year.png",
    )

    print("\n=== Qn–11: Interpretation ===")
    ym = np.array([centroids_orig[j, feature_names.index("year")] for j in range(K)])
    km_m = np.array([centroids_orig[j, feature_names.index("km_driven")] for j in range(K)])
    idx_premium, idx_older = interpret_clusters(ym, km_m)

    for j in range(K):
        print(
            f"Cluster {j}: mean year ≈ {ym[j]:.1f}, mean km_driven (centroid) ≈ {km_m[j]:.0f}"
        )

    print(
        f"\nPremium proxy (newer + relatively lower km): cluster {idx_premium}. "
        "This cluster tends toward higher model years and lower odometer readings."
    )
    print(
        f"Older cars proxy (lowest mean year): cluster {idx_older}. "
        "This cluster groups listings with the oldest average manufacturing years."
    )


if __name__ == "__main__":
    main()
