"""
Customer Segmentation — Hierarchical Clustering with Gower Distance

Groups telecom customers into distinct behavioral segments using
agglomerative hierarchical clustering on a mixed dataset of continuous
and categorical variables. Gower distance handles both data types,
enabling segmentation without restricting the analysis to numeric
features only.

Author: Amir Vaziri
"""

import os

import gower
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "churn_clean.csv")
CLEANED_DATA_PATH = os.path.join("data", "cleaned_churn_data.csv")
OUTPUT_DIR = "Figures"

CONTINUOUS_COLS = [
    "Tenure", "MonthlyCharge", "Bandwidth_GB_Year",
    "Income", "Outage_sec_perweek",
]
CATEGORICAL_COLS = [
    "Contract", "InternetService", "TechSupport",
    "DeviceProtection", "StreamingTV",
    "PaymentMethod", "PaperlessBilling",
]
SELECTED_COLS = CONTINUOUS_COLS + CATEGORICAL_COLS

MIN_CLUSTERS = 2
MAX_CLUSTERS = 6


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Select features, fill missing values, cast categorical columns to
    string, and normalize continuous columns with MinMaxScaler."""
    df = df[SELECTED_COLS].copy()

    # Treat missing InternetService entries as a distinct "no internet" category
    # rather than dropping rows — preserves 2,129 observations
    df["InternetService"] = df["InternetService"].fillna("None")

    # Cast categorical columns to object dtype for Gower compatibility.
    # astype(str) on pandas 3.x returns StringDtype; gower requires object.
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str).astype(object)

    # Scale continuous columns to [0, 1] so they contribute equally to Gower distance
    scaler = MinMaxScaler()
    df[CONTINUOUS_COLS] = scaler.fit_transform(df[CONTINUOUS_COLS])

    return df


# ---------------------------------------------------------------------------
# Cluster selection
# ---------------------------------------------------------------------------

def find_optimal_threshold(linked: np.ndarray) -> tuple[float, float]:
    """Return the cut threshold and jump size for the most separated
    cluster solution within MIN_CLUSTERS to MAX_CLUSTERS.

    Consecutive differences between sorted merge distances are used as a
    proxy for the natural cut point — the largest jump within the
    target cluster range marks where merging becomes most costly."""
    distances = linked[:, 2]
    diffs = np.diff(distances)

    best_t = None
    best_jump = -1.0

    for i in range(len(distances) - 1):
        jump = diffs[i]
        t = distances[i + 1]
        k = len(set(fcluster(linked, t=t, criterion="distance")))
        if MIN_CLUSTERS <= k <= MAX_CLUSTERS and jump > best_jump:
            best_jump = jump
            best_t = t

    return best_t, best_jump


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_dendrogram(
    linked: np.ndarray,
    threshold: float,
    jump: float,
    distances: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Plot and save the hierarchical clustering dendrogram with the
    selected cut threshold annotated as a horizontal reference line."""
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linked, truncate_mode="level", p=5, ax=ax)

    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Clusters")
    ax.set_ylabel("Distance")

    # Annotate the selected cut threshold
    t_idx = np.searchsorted(distances, threshold) - 1
    prev = distances[max(t_idx, 0)]
    ax.axhline(y=threshold, linestyle="--", lw=2, color="green", alpha=0.6)
    ax.text(
        x=0, y=threshold + 0.2,
        s=f"Δ={jump:.2f} ({prev:.2f}→{threshold:.2f})",
        fontsize=9,
    )

    plt.tight_layout()
    save_path = os.path.join(output_dir, "DendrogramPlot.jpg")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_clusters(
    df: pd.DataFrame,
    clusters: np.ndarray,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Project cluster assignments into 2D via PCA and save a labeled
    scatter plot. PCA is used for visualization only — clustering is
    performed in the full feature space."""
    # One-hot encode categorical columns for PCA input
    df_encoded = pd.get_dummies(df.drop("Cluster", axis=1))
    df_scaled = StandardScaler().fit_transform(df_encoded)
    pca_result = PCA(n_components=2).fit_transform(df_scaled)

    unique_labels = np.unique(clusters)
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for label in unique_labels:
        mask = clusters == label
        ax.scatter(
            pca_result[mask, 0], pca_result[mask, 1],
            c=color_map[label], label=f"Cluster {label}", alpha=0.7, s=10,
        )

    ax.set_title("Customer Clusters (PCA Projection)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend(title="Cluster", loc="best")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "Clusters.jpg")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_cluster_summary(df: pd.DataFrame) -> None:
    """Print mean numeric values and top categorical value per cluster."""
    print(f"\n{'─' * 60}\n  Cluster Numeric Summary\n{'─' * 60}")
    print(df.groupby("Cluster").mean(numeric_only=True).to_string())

    print(f"\n{'─' * 60}\n  Cluster Categorical Breakdown\n{'─' * 60}")
    for col in df.select_dtypes(include="object").columns:
        print(f"\n  {col}:")
        print(df.groupby("Cluster")[col].value_counts(normalize=True).to_string())


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full customer segmentation pipeline.

    Steps:
        1. Load and preprocess the raw dataset.
        2. Compute the Gower distance matrix.
        3. Perform agglomerative hierarchical clustering (average linkage).
        4. Find the optimal cut threshold via linkage distance gap analysis.
        5. Assign cluster labels and generate a dendrogram plot.
        6. Visualize clusters with a PCA projection.
        7. Print a cluster summary.
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 14
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and preprocess
    # ------------------------------------------------------------------
    print(f"\nLoading: {DATA_PATH}")
    df_raw = pd.read_csv(DATA_PATH)
    print(f"  Raw    — rows: {len(df_raw):,}  columns: {df_raw.shape[1]}")

    df = preprocess(df_raw)
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"  Clean  — rows: {len(df):,}  features: {df.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Gower distance matrix
    #    Handles mixed continuous + categorical features in one unified
    #    similarity measure — required for Gower-based clustering
    # ------------------------------------------------------------------
    print("\nComputing Gower distance matrix...")
    gower_dist = gower.gower_matrix(df)

    # ------------------------------------------------------------------
    # 3. Hierarchical clustering (average linkage)
    #    Average linkage balances sensitivity to cluster structure and
    #    robustness to noise, unlike single or complete linkage extremes
    # ------------------------------------------------------------------
    print("Running hierarchical clustering (average linkage)...")
    linked = linkage(gower_dist, method="average")

    # ------------------------------------------------------------------
    # 4. Optimal cut threshold via gap analysis
    # ------------------------------------------------------------------
    threshold, jump = find_optimal_threshold(linked)
    distances = linked[:, 2]
    clusters_preview = fcluster(linked, t=threshold, criterion="distance")
    n_clusters = len(np.unique(clusters_preview))
    print(f"\n  Threshold: {threshold:.4f}  Jump: {jump:.4f}  Clusters: {n_clusters}")

    # ------------------------------------------------------------------
    # 5. Dendrogram
    # ------------------------------------------------------------------
    print("\nPlotting dendrogram...")
    plot_dendrogram(linked, threshold, jump, distances)

    # ------------------------------------------------------------------
    # 6. Cluster assignment + PCA visualization
    # ------------------------------------------------------------------
    clusters = fcluster(linked, t=threshold, criterion="distance")
    df["Cluster"] = clusters

    print("Plotting PCA cluster projection...")
    plot_clusters(df, clusters)

    # ------------------------------------------------------------------
    # 7. Cluster summary
    # ------------------------------------------------------------------
    print_cluster_summary(df)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
