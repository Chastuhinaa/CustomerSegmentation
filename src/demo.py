import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score

from segmentation import DATA_PATH, SEED, SKEWED, load_and_clean, prepare_features

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def load():
    df = pd.read_csv(DATA_PATH)
    clean, report = load_and_clean(df)
    features, scaled = prepare_features(clean)
    return df, clean, report, features, scaled


def fit(scaled, k=4):
    return KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(scaled)


def show_data():
    df, *_ = load()
    print("Рядків і колонок:", df.shape)
    print("Дублікатів рядків:", df.duplicated().sum(), "| унікальних CUST_ID:", df["CUST_ID"].nunique())
    print(df.iloc[:5, :7].round(2).to_string(index=False))
    print("Пропуски:")
    print(df.isna().sum()[lambda s: s > 0].to_string())


def show_missing():
    df, clean, *_ = load()
    missing = df["MINIMUM_PAYMENTS"].isna()
    zero = df["PAYMENTS"] == 0
    print("Пропусків MINIMUM_PAYMENTS:", missing.sum())
    print("  із них PAYMENTS = 0 (замінено на 0):", (missing & zero).sum())
    print("  із них PAYMENTS > 0 (замінено медіаною):", (missing & ~zero).sum())
    print("Медіана, якою заповнено MINIMUM_PAYMENTS:", round(clean.loc[missing & ~zero, "MINIMUM_PAYMENTS"].iloc[0], 2))
    print("Медіана, якою заповнено CREDIT_LIMIT:", clean.loc[df["CREDIT_LIMIT"].isna(), "CREDIT_LIMIT"].iloc[0])
    print("Пропусків після очищення:", clean.isna().sum().sum())


def show_outliers():
    _, clean, report, features, _ = load()
    total = report["rows"]
    outliers = report["rows_with_any_outlier"]
    print(f"Клієнтів із хоча б одним викидом за IQR: {outliers} із {total} ({outliers / total:.1%})")
    table = pd.DataFrame({"до": clean[SKEWED].skew(), "після log1p": features[SKEWED].skew()})
    print(table.round(2).to_string())


def show_features():
    _, _, _, features, scaled = load()
    print("Ознак для кластеризації:", features.shape[1])
    table = pd.DataFrame({"середнє": scaled.mean(axis=0), "ст. відхилення": scaled.std(axis=0)}, index=features.columns)
    print(table.round(4).abs().to_string())


def show_model():
    _, _, _, features, scaled = load()
    model = fit(scaled)
    print("Розміри кластерів:", np.bincount(model.labels_))
    print("Ітерацій у найкращому запуску:", model.n_iter_)
    print("Інерція з моделі:", round(model.inertia_, 2))
    manual = ((scaled - model.cluster_centers_[model.labels_]) ** 2).sum()
    print("Інерція, порахована вручну:", round(manual, 2))
    print("Silhouette Score:", round(silhouette_score(scaled, model.labels_), 4))
    print(pd.DataFrame(model.cluster_centers_, columns=features.columns).round(2).T.to_string())


def show_silhouette(index=0):
    _, clean, _, _, scaled = load()
    labels = fit(scaled).labels_
    distances = np.sqrt(((scaled - scaled[index]) ** 2).sum(axis=1))
    own = labels == labels[index]
    a = distances[own].sum() / (own.sum() - 1)
    b = min(distances[labels == c].mean() for c in set(labels) - {labels[index]})
    print("Клієнт:", clean["CUST_ID"].iloc[index], "| кластер:", labels[index])
    print("a, середня відстань до свого кластера:", round(a, 4))
    print("b, середня відстань до найближчого чужого:", round(b, 4))
    print("(b - a) / max(a, b):", round((b - a) / max(a, b), 4))
    print("silhouette_samples:", round(silhouette_samples(scaled, labels)[index], 4))


def show_zero_payments():
    _, clean, _, _, scaled = load()
    zero = (clean["PAYMENTS"] == 0).values
    six = KMeans(n_clusters=6, n_init=20, random_state=SEED).fit_predict(scaled)
    print("Клієнтів із PAYMENTS = 0:", zero.sum())
    print("k = 6, розміри:", np.bincount(six))
    print("k = 6, таких клієнтів у кластерах:", [int(zero[six == c].sum()) for c in range(6)])
    for k in (4, 6):
        all_rows = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(scaled)
        part = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(scaled[~zero])
        print(f"k = {k}: silhouette зі всіма {silhouette_score(scaled, all_rows):.4f}, без них {silhouette_score(scaled[~zero], part):.4f}")


def show_stability():
    _, _, _, _, scaled = load()
    first = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(scaled)
    for seed in (1, 2, 3):
        other = KMeans(n_clusters=4, n_init=10, random_state=seed).fit_predict(scaled)
        print(f"random_state 0 і {seed}: ARI = {adjusted_rand_score(first, other):.4f}")


def show_k(k):
    _, _, _, _, scaled = load()
    model = fit(scaled, k)
    print(f"k = {k}")
    print("Розміри кластерів:", np.bincount(model.labels_))
    print("Інерція:", round(model.inertia_, 2))
    print("Silhouette Score:", round(silhouette_score(scaled, model.labels_), 4))


def show_client(client):
    _, clean, _, _, scaled = load()
    matches = np.flatnonzero(clean["CUST_ID"] == client)
    if not len(matches):
        print("Клієнта не знайдено:", client)
        return
    index = matches[0]
    model = fit(scaled)
    distances = np.sqrt(((model.cluster_centers_ - scaled[index]) ** 2).sum(axis=1))
    print("Клієнт:", client, "| кластер:", model.labels_[index])
    print("Відстані до центрів:", distances.round(3))
    print(clean.iloc[index].drop("CUST_ID").round(2).to_string())


COMMANDS = {
    "data": show_data,
    "missing": show_missing,
    "outliers": show_outliers,
    "features": show_features,
    "model": show_model,
    "silhouette": show_silhouette,
    "zero": show_zero_payments,
    "stability": show_stability,
}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name == "k" and len(sys.argv) > 2:
        show_k(int(sys.argv[2]))
    elif name == "client" and len(sys.argv) > 2:
        show_client(sys.argv[2])
    elif name in COMMANDS:
        COMMANDS[name]()
    else:
        print("Команди:", ", ".join(COMMANDS), ", k <число>, client <CUST_ID>")


if __name__ == "__main__":
    main()
