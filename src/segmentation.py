import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "CC_GENERAL.csv"
RESULTS = ROOT / "results"
SEED = 42
K_RANGE = range(2, 11)

REDUNDANT = ["CUST_ID", "PURCHASES", "CASH_ADVANCE_TRX", "PURCHASES_TRX", "TENURE"]
SKEWED = [
    "BALANCE",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
]


def save_csv(frame, name, index=False):
    frame.to_csv(RESULTS / name, sep=";", decimal=",", index=index, encoding="utf-8-sig", float_format="%.4f")


def save_fig(name):
    plt.tight_layout()
    plt.savefig(RESULTS / name, dpi=150)
    plt.close()


def load_and_clean(df):
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": int(df.duplicated().sum()),
        "missing": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
    }
    clean = df.copy()
    no_payments = clean["MINIMUM_PAYMENTS"].isna() & (clean["PAYMENTS"] == 0)
    report["missing_minpay_with_zero_payments"] = int(no_payments.sum())
    clean.loc[no_payments, "MINIMUM_PAYMENTS"] = 0
    for column in ["MINIMUM_PAYMENTS", "CREDIT_LIMIT"]:
        clean[column] = clean[column].fillna(clean[column].median())
    numeric = clean.select_dtypes(include="number")
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    outliers = ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum()
    report["outliers_iqr"] = {k: int(v) for k, v in outliers.items()}
    report["rows_with_any_outlier"] = int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).any(axis=1).sum())
    return clean, report


def prepare_features(clean):
    features = clean.drop(columns=REDUNDANT).copy()
    for column in SKEWED:
        features[column] = np.log1p(features[column])
    scaled = StandardScaler().fit_transform(features)
    return features, scaled


def plot_eda(clean):
    numeric = clean.drop(columns=["CUST_ID"])
    summary = numeric.describe().T
    summary["skew"] = numeric.skew()
    save_csv(summary.reset_index(names="feature"), "summary_statistics.csv")

    key = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT", "PAYMENTS", "PURCHASES_FREQUENCY"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, column in zip(axes.ravel(), key):
        sns.histplot(clean[column], bins=40, ax=ax, color="#4C72B0")
        ax.set_title(column)
        ax.set_xlabel("")
        ax.set_ylabel("Кількість клієнтів")
    save_fig("01_distributions.png")

    plt.figure(figsize=(11, 9))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
    plt.title("Кореляції ознак")
    save_fig("02_correlation.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [("BALANCE", "CASH_ADVANCE"), ("CREDIT_LIMIT", "PURCHASES"), ("PURCHASES", "PAYMENTS")]
    for ax, (x, y) in zip(axes, pairs):
        ax.scatter(clean[x], clean[y], s=5, alpha=0.3, color="#4C72B0")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    save_fig("03_scatter.png")


def scan_k(scaled):
    rows = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(scaled)
        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled, model.labels_),
                "davies_bouldin": davies_bouldin_score(scaled, model.labels_),
                "calinski_harabasz": calinski_harabasz_score(scaled, model.labels_),
                "min_cluster_share": np.bincount(model.labels_).min() / len(scaled),
            }
        )
    scan = pd.DataFrame(rows)
    save_csv(scan, "k_selection.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(scan["k"], scan["inertia"], marker="o")
    axes[0].set_title("Метод ліктя")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Інерція")
    axes[1].plot(scan["k"], scan["silhouette"], marker="o", color="#C44E52")
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")
    save_fig("04_k_selection.png")
    return scan


def stability(scaled, k):
    labels = [KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(scaled) for seed in range(10)]
    scores = [adjusted_rand_score(labels[i], labels[j]) for i in range(10) for j in range(i + 1, 10)]
    return float(np.mean(scores)), float(np.min(scores))


def raw_baseline(clean, k):
    raw = StandardScaler().fit_transform(clean.drop(columns=["CUST_ID"]))
    labels = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(raw)
    return float(silhouette_score(raw, labels)), [int(v) for v in np.bincount(labels)]


def zero_payment_check(clean, scaled, labels, k):
    zero = (clean["PAYMENTS"] == 0).values
    keep = ~zero
    six = KMeans(n_clusters=6, n_init=20, random_state=SEED).fit_predict(scaled)
    smallest = int(np.bincount(six).argmin())
    scan = {}
    for candidate in K_RANGE:
        part = KMeans(n_clusters=candidate, n_init=20, random_state=SEED).fit_predict(scaled[keep])
        scan[candidate] = float(silhouette_score(scaled[keep], part))
    part = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit_predict(scaled[keep])
    return {
        "zero_payment_clients": int(zero.sum()),
        "k6_smallest_cluster_size": int((six == smallest).sum()),
        "k6_smallest_cluster_zero_payment": int(zero[six == smallest].sum()),
        "zero_payment_by_cluster": [int(zero[labels == c].sum()) for c in range(k)],
        "silhouette_without_zero_payment": scan,
        "ari_with_vs_without": float(adjusted_rand_score(labels[keep], part)),
    }


def plot_silhouette(scaled, labels, k):
    values = silhouette_samples(scaled, labels)
    plt.figure(figsize=(8, 6))
    lower = 10
    palette = sns.color_palette("tab10", k)
    for cluster in range(k):
        part = np.sort(values[labels == cluster])
        plt.fill_betweenx(np.arange(lower, lower + len(part)), 0, part, color=palette[cluster])
        plt.text(-0.08, lower + len(part) / 2, str(cluster))
        lower += len(part) + 10
    plt.axvline(values.mean(), color="black", linestyle="--")
    plt.xlabel("Значення silhouette")
    plt.ylabel("Клієнти за кластерами")
    plt.yticks([])
    save_fig("06_silhouette.png")


def plot_pca(scaled, labels, feature_names):
    pca = PCA(n_components=2, random_state=SEED)
    points = pca.fit_transform(scaled)
    plt.figure(figsize=(9, 7))
    sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=labels, palette="tab10", s=12, alpha=0.6, linewidth=0)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.legend(title="Кластер")
    save_fig("05_pca_clusters.png")
    loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=["PC1", "PC2"])
    save_csv(loadings.reset_index(names="feature"), "pca_loadings.csv")
    return [float(v) for v in pca.explained_variance_ratio_]


def plot_profiles(clean, features, labels):
    profile = clean.drop(columns=["CUST_ID"]).groupby(labels).mean()
    profile.insert(0, "share", np.bincount(labels) / len(labels))
    profile.insert(0, "size", np.bincount(labels))
    medians = clean.drop(columns=["CUST_ID"]).groupby(labels).median()
    save_csv(profile.reset_index(names="cluster"), "cluster_profiles_mean.csv")
    save_csv(medians.reset_index(names="cluster"), "cluster_profiles_median.csv")

    scaled_features = pd.DataFrame(StandardScaler().fit_transform(features), columns=features.columns)
    z_profile = scaled_features.groupby(labels).mean()
    plt.figure(figsize=(12, 8))
    sns.heatmap(z_profile, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 12})
    plt.ylabel("Кластер")
    plt.title("Середні нормалізовані значення ознак за кластерами")
    save_fig("07_cluster_heatmap.png")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, column in zip(axes.ravel(), ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PRC_FULL_PAYMENT"]):
        sns.barplot(x=profile.index, y=profile[column].values, ax=ax, color="#4C72B0")
        ax.set_title(f"Середнє {column}")
        ax.set_xlabel("Кластер")
        ax.set_ylabel("")
    save_fig("08_cluster_bars.png")


def main(k=None):
    RESULTS.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = pd.read_csv(DATA_PATH)
    clean, report = load_and_clean(df)
    plot_eda(clean)
    features, scaled = prepare_features(clean)
    scan = scan_k(scaled)
    if k is None:
        k = int(scan.loc[scan["silhouette"].idxmax(), "k"])
    model = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(scaled)
    labels = model.labels_
    ari_mean, ari_min = stability(scaled, k)
    raw_silhouette, raw_sizes = raw_baseline(clean, k)
    plot_silhouette(scaled, labels, k)
    explained = plot_pca(scaled, labels, features.columns)
    plot_profiles(clean, features, labels)
    assignments = clean[["CUST_ID"]].assign(cluster=labels)
    save_csv(assignments, "customer_clusters.csv")
    report.update(
        {
            "features_used": list(features.columns),
            "k": k,
            "cluster_sizes": [int(v) for v in np.bincount(labels)],
            "silhouette": float(silhouette_score(scaled, labels)),
            "davies_bouldin": float(davies_bouldin_score(scaled, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(scaled, labels)),
            "ari_mean": ari_mean,
            "ari_min": ari_min,
            "raw_baseline_silhouette": raw_silhouette,
            "raw_baseline_sizes": raw_sizes,
            "pca_explained": explained,
            "zero_payment_check": zero_payment_check(clean, scaled, labels, k),
        }
    )
    (RESULTS / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(scan.round(4).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
