"""Golden fixtures for the clustering algorithm family. Run via ``make fixtures``
or ``python3 -m gen.gen_algorithms`` from ``stats-claw/reference``.

A single, fixed-seed blob dataset is shared across every algorithm so the
adjusted-Rand-index comparisons all reference the same ground truth. Each
generator records the dataset, the ``scikit-learn`` reference labels, and any
identifiable scalars (k-means inertia, DBSCAN core-sample count, discovered
cluster count) the Rust suite asserts exactly.

A follow-up wave extends this module with
decomposition (PCA, factor analysis, ICA), embedding (t-SNE, UMAP, LLE), PELT
change-point, and GMM fixtures. Stochastic embeddings record the reference
trustworthiness so the Rust suite compares against a data-derived quality
threshold rather than an irreproducible exact embedding.
"""
import numpy as np
import ruptures
import sklearn
from sklearn.cluster import (
    DBSCAN,
    AffinityPropagation,
    AgglomerativeClustering,
    KMeans,
    MeanShift,
    SpectralClustering,
)
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.manifold import (
    TSNE,
    LocallyLinearEmbedding,
    trustworthiness,
)
from sklearn.mixture import GaussianMixture

from ._common import write_fixture

VER = sklearn.__version__
SEED = 42
N_SAMPLES = 150
CENTERS = 3
CLUSTER_STD = 0.60


def _blobs():
    """Return the shared (data, true_labels) blob dataset as plain lists."""
    data, true_labels = make_blobs(
        n_samples=N_SAMPLES,
        centers=CENTERS,
        cluster_std=CLUSTER_STD,
        random_state=SEED,
    )
    return data, data.tolist(), [int(c) for c in true_labels]


def gen_kmeans():
    data, data_list, _ = _blobs()
    model = KMeans(n_clusters=CENTERS, n_init=10, random_state=SEED).fit(data)
    write_fixture(
        "algo_kmeans_blobs",
        {
            "data": data_list,
            "k": CENTERS,
            "labels": [int(c) for c in model.labels_],
            "inertia": float(model.inertia_),
        },
        library="sklearn.cluster.KMeans",
        version=VER,
        seed=SEED,
    )


def gen_dbscan():
    data, data_list, _ = _blobs()
    model = DBSCAN(eps=0.8, min_samples=5).fit(data)
    write_fixture(
        "algo_dbscan_blobs",
        {
            "data": data_list,
            "eps": 0.8,
            "min_samples": 5,
            "labels": [int(c) for c in model.labels_],
            "core_sample_count": int(len(model.core_sample_indices_)),
        },
        library="sklearn.cluster.DBSCAN",
        version=VER,
    )


def gen_hierarchical():
    data, data_list, _ = _blobs()
    model = AgglomerativeClustering(n_clusters=CENTERS, linkage="ward").fit(data)
    write_fixture(
        "algo_hierarchical_blobs",
        {
            "data": data_list,
            "k": CENTERS,
            "linkage": "ward",
            "labels": [int(c) for c in model.labels_],
        },
        library="sklearn.cluster.AgglomerativeClustering",
        version=VER,
    )


def gen_mean_shift():
    data, data_list, _ = _blobs()
    model = MeanShift(bandwidth=1.5).fit(data)
    write_fixture(
        "algo_mean_shift_blobs",
        {
            "data": data_list,
            "bandwidth": 1.5,
            "labels": [int(c) for c in model.labels_],
            "n_clusters": int(len(set(model.labels_))),
        },
        library="sklearn.cluster.MeanShift",
        version=VER,
    )


def gen_affinity():
    data, data_list, _ = _blobs()
    model = AffinityPropagation(
        damping=0.9, preference=-50.0, random_state=SEED
    ).fit(data)
    write_fixture(
        "algo_affinity_blobs",
        {
            "data": data_list,
            "damping": 0.9,
            "preference": -50.0,
            "labels": [int(c) for c in model.labels_],
            "n_clusters": int(len(model.cluster_centers_indices_)),
        },
        library="sklearn.cluster.AffinityPropagation",
        version=VER,
        seed=SEED,
    )


def gen_spectral():
    data, data_list, _ = _blobs()
    model = SpectralClustering(
        n_clusters=CENTERS,
        affinity="rbf",
        gamma=1.0,
        assign_labels="kmeans",
        random_state=SEED,
    ).fit(data)
    write_fixture(
        "algo_spectral_blobs",
        {
            "data": data_list,
            "k": CENTERS,
            "gamma": 1.0,
            "labels": [int(c) for c in model.labels_],
        },
        library="sklearn.cluster.SpectralClustering",
        version=VER,
        seed=SEED,
    )


# --- Wave-2: decomposition / embedding / change-point / GMM ---------------

DECOMP_SEED = 7
DECOMP_SAMPLES = 60
DECOMP_FEATURES = 5


def _decomp_data():
    """Return a shared (array, list) dataset for decomposition/embedding fixtures.

    A fixed-seed blob set in 5-D with 3 centres: enough structure for PCA to find
    a clear leading-variance direction and for neighbour embeddings to preserve
    locality, while staying small enough for an O(n^2) Jacobi eigensolver in Rust.
    """
    data, _ = make_blobs(
        n_samples=DECOMP_SAMPLES,
        centers=CENTERS,
        cluster_std=1.0,
        n_features=DECOMP_FEATURES,
        random_state=DECOMP_SEED,
    )
    return data, data.tolist()


def gen_pca():
    data, data_list = _decomp_data()
    k = 2
    model = PCA(n_components=k, svd_solver="full").fit(data)
    transformed = model.transform(data)
    reconstructed = model.inverse_transform(transformed)
    recon_error = float(np.mean((data - reconstructed) ** 2))
    write_fixture(
        "algo_pca",
        {
            "data": data_list,
            "k": k,
            "components": model.components_.tolist(),
            "explained_variance": model.explained_variance_.tolist(),
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            "reconstruction_error": recon_error,
        },
        library="sklearn.decomposition.PCA",
        version=VER,
        seed=DECOMP_SEED,
    )


def gen_pelt():
    """Piecewise-constant 1-D signal with three known level shifts."""
    rng = np.random.default_rng(SEED)
    segments = [
        np.full(40, 0.0),
        np.full(40, 5.0),
        np.full(40, -3.0),
        np.full(40, 2.0),
    ]
    clean = np.concatenate(segments)
    signal = clean + rng.normal(scale=0.25, size=clean.size)
    penalty = 10.0
    algo = ruptures.Pelt(model="l2", min_size=2, jump=1).fit(
        signal.reshape(-1, 1)
    )
    breakpoints = algo.predict(pen=penalty)
    write_fixture(
        "algo_pelt",
        {
            "signal": signal.tolist(),
            "penalty": penalty,
            "min_size": 2,
            "breakpoints": [int(b) for b in breakpoints],
        },
        library="ruptures.Pelt(model=l2)",
        version=ruptures.__version__,
        seed=SEED,
    )


def gen_factor_analysis():
    data, data_list = _decomp_data()
    k = 2
    model = FactorAnalysis(n_components=k, random_state=SEED).fit(data)
    transformed = model.transform(data)
    reconstructed = transformed @ model.components_ + model.mean_
    recon_error = float(np.mean((data - reconstructed) ** 2))
    write_fixture(
        "algo_factor_analysis",
        {
            "data": data_list,
            "k": k,
            "reconstruction_error": recon_error,
        },
        library="sklearn.decomposition.FactorAnalysis",
        version=VER,
        seed=SEED,
    )


def gen_ica():
    """Two independent source signals linearly mixed, recovered by FastICA."""
    rng = np.random.default_rng(SEED)
    n = 200
    time = np.linspace(0, 8, n)
    s1 = np.sign(np.sin(2.0 * time))
    s2 = np.sin(3.0 * time)
    sources = np.c_[s1, s2]
    sources += 0.05 * rng.normal(size=sources.shape)
    sources -= sources.mean(axis=0)
    mixing = np.array([[1.0, 1.0], [0.5, 2.0]])
    mixed = sources @ mixing.T
    model = FastICA(n_components=2, random_state=SEED, whiten="unit-variance")
    recovered = model.fit_transform(mixed)
    write_fixture(
        "algo_ica",
        {
            "mixed": mixed.tolist(),
            "k": 2,
            "sources": recovered.tolist(),
        },
        library="sklearn.decomposition.FastICA",
        version=VER,
        seed=SEED,
    )


def gen_lle():
    data, data_list = _decomp_data()
    k = 2
    n_neighbors = 10
    embedding = LocallyLinearEmbedding(
        n_components=k, n_neighbors=n_neighbors, random_state=SEED
    ).fit_transform(data)
    trust = float(trustworthiness(data, embedding, n_neighbors=5))
    write_fixture(
        "algo_lle",
        {
            "data": data_list,
            "k": k,
            "n_neighbors": n_neighbors,
            "trustworthiness": trust,
        },
        library="sklearn.manifold.LocallyLinearEmbedding",
        version=VER,
        seed=SEED,
    )


def gen_tsne():
    data, data_list = _decomp_data()
    k = 2
    embedding = TSNE(
        n_components=k, random_state=SEED, perplexity=10.0, init="pca"
    ).fit_transform(data)
    trust = float(trustworthiness(data, embedding, n_neighbors=5))
    write_fixture(
        "algo_tsne",
        {
            "data": data_list,
            "k": k,
            "trustworthiness": trust,
        },
        library="sklearn.manifold.TSNE",
        version=VER,
        seed=SEED,
    )


def gen_umap():
    """UMAP fixture WITHOUT a umap-learn reference.

    ``umap-learn`` is not installed and UMAP is stochastic, so exact equivalence is
    infeasible. The fixture records only the dataset and a conservative
    trustworthiness target the stats-claw embedding must clear; correctness is the
    quality measure, not byte-identity with a reference embedding.
    """
    data, data_list = _decomp_data()
    write_fixture(
        "algo_umap",
        {
            "data": data_list,
            "k": 2,
            "n_neighbors": 10,
            "trustworthiness_target": 0.90,
        },
        library="none (umap-learn absent; quality-metric only)",
        version="n/a",
        seed=SEED,
    )


def gen_gmm():
    data, data_list, _ = _blobs()
    model = GaussianMixture(
        n_components=CENTERS,
        covariance_type="full",
        random_state=SEED,
        n_init=5,
    ).fit(data)
    write_fixture(
        "algo_gmm",
        {
            "data": data_list,
            "k": CENTERS,
            "labels": [int(c) for c in model.predict(data)],
            "bic": float(model.bic(data)),
            "aic": float(model.aic(data)),
        },
        library="sklearn.mixture.GaussianMixture",
        version=VER,
        seed=SEED,
    )


def main():
    """Regenerate every clustering golden fixture."""
    gen_kmeans()
    gen_dbscan()
    gen_hierarchical()
    gen_mean_shift()
    gen_affinity()
    gen_spectral()
    gen_pca()
    gen_pelt()
    gen_factor_analysis()
    gen_ica()
    gen_lle()
    gen_tsne()
    gen_umap()
    gen_gmm()


if __name__ == "__main__":
    main()
