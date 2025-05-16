import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, OPTICS
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from random import sample
from numpy.random import uniform
import warnings

# Alternativa più generica se il valore di n_jobs nel messaggio può variare
warnings.filterwarnings(
    action='ignore',
    message="overridden to 1 by setting random_state. Use no seed for parallelism.", # Parte più generica del messaggio
    category=UserWarning,
    module='umap.umap_'
)



# todo: standardizzare e normalizzare tutte le features
def run_kmeans_clustering(df, n_clusters=3, visualize=True, umap_comp=5):
    context_cols = ['player']
    feature_cols = [col for col in df.columns if col not in context_cols]

    # 1. Filtro a bassa varianza
    # selector = VarianceThreshold(threshold=0.005)
    # X_high_var = selector.fit_transform(df[feature_cols])
    # selected_features = [col for col, var in zip(feature_cols, selector.variances_) if var > 0.005]

    # 2. Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    print(feature_cols)

    # PCA con n componenti scelte in base alla varianza spiegata
    # pca = PCA(n_components=0.9)  # Mantieni abbastanza PC da spiegare il 90% della varianza
    # X_reduced = pca.fit_transform(X_scaled)
    # 2. Riduzione dimensionale con UMAP per clustering
    reducer = umap.UMAP(n_components=umap_comp, random_state=42)
    X_reduced = reducer.fit_transform(X_scaled)
    print(len(X_reduced[0]))

    # 3. KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_reduced)

    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels

    # 4. Valutazione silhouette
    score = silhouette_score(X_reduced, cluster_labels)
    print(f"silhouette = {score:.3f}")

    # Calcolo del DBI
    dbi = davies_bouldin_score(X_reduced, cluster_labels)
    print(f"Dbi score (Agglomerative + UMAP): {dbi:.3f}")

    # 5. PCA per visualizzazione
    if visualize:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_reduced)
        plt.figure(figsize=(8, 6))
        for cluster_id in range(n_clusters):
            idx = df_with_clusters['cluster'] == cluster_id
            plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f"Cluster {cluster_id}", alpha=0.7)
        plt.title("Clustering dei profili di gioco (PCA)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df_with_clusters, silhouette_score, dbi


def run_dbscan_clustering(df, eps=0.5, min_samples=5, visualize=True):
    context_cols = ['player']
    feature_cols = [col for col in df.columns if col not in context_cols]

    X = df[feature_cols].values

    # Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # riduzione features con umap
    reducer = umap.UMAP(n_components=5, random_state=42)
    X_reduced = reducer.fit_transform(X_scaled)

    # Clustering con DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = db.fit_predict(X_reduced)
    score = silhouette_score(X_reduced, cluster_labels)
    print("Silhouette score dbscan: {:.3f}".format(score))

    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels

    # Visualizzazione con PCA
    if visualize:
        # Visualizzazione 2D
        reducer_2d = umap.UMAP(n_components=2, random_state=42)
        X_2d = reducer_2d.fit_transform(X_scaled)
        plt.figure(figsize=(8, 6))
        for cluster_id in sorted(set(cluster_labels)):
            idx = cluster_labels == cluster_id
            label = f"Cluster {cluster_id}" if cluster_id != -1 else "Rumore"
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=label, alpha=0.7)
        plt.title("DBSCAN clustering (umap)")
        plt.xlabel("umap 1")
        plt.ylabel("umap 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df_with_clusters


def hopkins(X, sampling_size=0.1, n_iterations=30, random_state=None):
    """
    Calcola la Hopkins statistic ripetuta n_iterations volte per robustezza.

    Parametri:
        X (numpy array o DataFrame): dati da analizzare (devono essere scalati)
        sampling_size (float): percentuale di punti da campionare (0 < s < 1)
        n_iterations (int): numero di iterazioni da eseguire
        random_state (int): seme per riproducibilità

    Ritorna:
        mean_H (float): media delle Hopkins statistic
        std_H (float): deviazione standard delle Hopkins statistic
        values (list): valori individuali delle iterazioni
    """
    if isinstance(X, np.ndarray) is False:
        X = X.values

    np.random.seed(random_state)
    n_samples = X.shape[0]
    m = int(sampling_size * n_samples)
    d = X.shape[1]

    hopkins_values = []

    for _ in range(n_iterations):
        # campiona m punti dal dataset
        idx = np.random.choice(np.arange(n_samples), m, replace=False)
        X_sample = X[idx]

        # genera m punti casuali nello stesso spazio
        X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
        X_random = np.random.uniform(X_min, X_max, (m, d))

        # nearest neighbor: distanza minima da un punto "vero"
        nn = NearestNeighbors(n_neighbors=2)  # 1 è se stesso, 2 è il più vicino diverso
        nn.fit(X)

        u_distances = np.array([
            nn.kneighbors([x], 2, return_distance=True)[0][0][1]
            for x in X_sample
        ])
        w_distances = np.array([
            nn.kneighbors([x], 1, return_distance=True)[0][0][0]
            for x in X_random
        ])

        H = np.sum(w_distances) / (np.sum(w_distances) + np.sum(u_distances))
        hopkins_values.append(H)

    mean_H = np.mean(hopkins_values)
    std_H = np.std(hopkins_values)

    return mean_H, std_H, hopkins_values


def explained_variance_from_features(df):
    # 1. Standardizza il dataset (escludi colonne non numeriche se necessario)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)  # df contiene solo colonne numeriche

    # 2. PCA con tutte le componenti
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # 3. Varianza spiegata per ogni componente
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var_ratio)

    # 4. Plot della varianza spiegata
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_var_ratio) + 1), cumulative_var, marker='o', linestyle='--')
    plt.title('Varianza spiegata cumulativa dopo PCA')
    plt.xlabel('Numero di componenti principali')
    plt.ylabel('Varianza spiegata cumulativa')
    plt.grid(True)
    plt.show()

    # 5. (Opzionale) Stampare le percentuali
    for i, ratio in enumerate(explained_var_ratio, 1):
        print(f"PC{i}: {ratio:.4f} ({cumulative_var[i - 1]:.4f} cumulativa)")


import umap
from sklearn.mixture import GaussianMixture


def run_umap_gmm_clustering(df, context_cols=None, n_clusters=3, n_neighbors=15, min_dist=0.1,
                            proba_threshold=0.9, visualize=True):
    """
    Applica UMAP per riduzione + GMM per clustering, con gestione dei punti ambigui come rumore.

    Args:
        df: DataFrame di input con feature numeriche.
        context_cols: colonne da escludere (es. 'player').
        n_clusters: numero di componenti GMM.
        n_neighbors: parametro UMAP.
        min_dist: parametro UMAP.
        proba_threshold: soglia minima per assegnare un punto a un cluster (altrimenti è rumore).
        visualize: se True, visualizza il clustering in 2D.

    Returns:
        df_clustered: DataFrame con colonna 'cluster' (+ eventuale 'probability')
    """
    if context_cols is None:
        context_cols = []

    feature_cols = [col for col in df.columns if col not in context_cols]
    X = df[feature_cols].values

    # 1. Filtro a bassa varianza
    selector = VarianceThreshold(threshold=0.005)
    X_high_var = selector.fit_transform(df[feature_cols])

    # 2. Standardizzazione
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_high_var)

    # 3. UMAP
    reducer = umap.UMAP(n_components=8, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # 4. GMM
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
    gmm.fit(X_umap)
    cluster_probs = gmm.predict_proba(X_umap)
    max_probs = cluster_probs.max(axis=1)
    predicted_labels = gmm.predict(X_umap)

    print(predicted_labels, len(predicted_labels))
    print(max_probs, len(max_probs))

    # Applica soglia → rumore se bassa confidenza
    cluster_labels = np.where(max_probs < proba_threshold, -1, predicted_labels)

    import seaborn as sns
    sns.histplot(max_probs, bins=20)
    plt.title("Distribuzione massima probabilità di appartenenza GMM")
    plt.show()

    # 5. Silhouette solo sui punti validi
    valid_mask = cluster_labels != -1
    print(cluster_labels, len(cluster_labels))
    if valid_mask.sum() > 1:
        sil_score = silhouette_score(X_umap[valid_mask], cluster_labels[valid_mask])
        print(f"Silhouette score (UMAP + GMM, validi): {sil_score:.3f}")
    else:
        print("Troppi punti ambigui per calcolare il silhouette score")

    # 6. Visualizzazione
    if visualize:
        plt.figure(figsize=(8, 6))
        for label in np.unique(cluster_labels):
            idx = cluster_labels == label
            color = 'black' if label == -1 else None
            plt.scatter(X_umap[idx, 0], X_umap[idx, 1], label=f"Cluster {label}", alpha=0.7, color=color)
        plt.title("Clustering con UMAP + GMM (con soglia probabilità)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 7. Ritorno
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    df_clustered['gmm_confidence'] = max_probs  # facoltativo
    return df_clustered


import hdbscan
from sklearn.preprocessing import StandardScaler


def run_umap_hdbscan_clustering(df, context_cols=None, min_cluster_size=5, visualize=True):
    """
    Applica UMAP + HDBSCAN al dataset per trovare cluster non sferici e rilevare rumore.

    Args:
        df: DataFrame con feature numeriche + opzionali colonne descrittive.
        context_cols: colonne non numeriche da escludere (es. 'player').
        min_cluster_size: dimensione minima di un cluster in HDBSCAN.
        visualize: se True, mostra grafico in 2D.

    Returns:
        df_clustered: DataFrame originale con colonna 'cluster' aggiunta.
    """

    if context_cols is None:
        context_cols = []

    # 1. Prepara i dati
    feature_cols = [col for col in df.columns if col not in context_cols]
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    # 2. Riduzione dimensionale con UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # 3. Clustering con HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, prediction_data=True)
    cluster_labels = clusterer.fit_predict(X_umap)

    score = silhouette_score(X_umap, cluster_labels)
    print(f"Silhouette score (UMAP + HDBSCAN): {score:.3f}")

    # 4. Visualizzazione
    if visualize:
        plt.figure(figsize=(8, 6))
        for label in set(cluster_labels):
            idx = cluster_labels == label
            color = 'black' if label == -1 else None  # rumore in nero
            plt.scatter(X_umap[idx, 0], X_umap[idx, 1], label=f"Cluster {label}", alpha=0.7, color=color)
        plt.title("Clustering con UMAP + HDBSCAN")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 5. Restituisci il DataFrame con etichette
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    return df_clustered


from sklearn.cluster import AgglomerativeClustering


def run_umap_agglomerative_clustering(df, context_cols=None, n_clusters=3,
                                      n_components_umap=5, linkage='ward', visualize=True):
    """
    Applica UMAP + Agglomerative Clustering e visualizza i risultati.

    Args:
        df: DataFrame contenente solo feature numeriche (più eventualmente colonne di contesto).
        context_cols: colonne da escludere (es. 'player').
        n_clusters: numero di cluster desiderati.
        n_components_umap: dimensioni UMAP per il clustering (tipico 5).
        linkage: tipo di collegamento ('ward', 'average', 'complete', 'single').
        visualize: se True, visualizza in 2D.

    Returns:
        DataFrame con cluster assegnati.
    """
    if context_cols is None:
        context_cols = []

    # 1. Estrai solo le feature numeriche
    feature_cols = [col for col in df.columns if col not in context_cols]
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    # 2. Riduzione dimensionale con UMAP per clustering
    reducer = umap.UMAP(n_components=n_components_umap, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)

    # 3. Agglomerative Clustering
    agglom = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    cluster_labels = agglom.fit_predict(X_umap)

    # 4. Silhouette score
    sil_score = silhouette_score(X_umap, cluster_labels)
    print(f"Silhouette score (Agglomerative + UMAP): {sil_score:.3f}")

    # Calcolo del DBI
    dbi = davies_bouldin_score(X_umap, cluster_labels)
    print(f"Dbi score (Agglomerative + UMAP): {dbi:.3f}")

    # 5. Visualizzazione 2D (con nuova UMAP a 2D solo per visual)
    if visualize:
        reducer_2d = umap.UMAP(n_components=2, random_state=42)
        X_2d = reducer_2d.fit_transform(X_scaled)
        plt.figure(figsize=(8, 6))
        for label in np.unique(cluster_labels):
            idx = cluster_labels == label
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"Cluster {label}", alpha=0.7)
        plt.title("Agglomerative Clustering (visualizzato in 2D UMAP)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 6. Restituzione
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    return df_clustered, sil_score, dbi


def run_umap_optics_clustering(df, reduction='umap', context_cols=None, min_pts=7, xi=0.08,
                               min_cluster_size=0.5, visualize=True, noise_perc=0.60):
    """
    Execute dimensionality reduction and OPTICS clustering on a dataset. The function
    supports both UMAP and PCA for dimensionality reduction. It generates various
    visualizations including a 2D clustering plot and a reachability plot for the
    OPTICS clustering. The resulting DataFrame includes an additional column
    assigning a cluster label to each row.

    the parameter 'eps' of optics is inf by default
    :param df: The input DataFrame containing the dataset to be clustered.
    :param reduction: The method for dimensionality reduction. It can be
        either 'umap' or 'pca'. If not specified, defaults to 'umap'.
    :param min_pts: The minimum number of points to form a dense region in OPTICS
        clustering. Defaults to 7.
    :param xi: A parameter for OPTICS that determines the minimum steepness on the
        reachability plot to identify clusters. Defaults to 0.08.
    :param min_cluster_size: The minimum relative size of a cluster in OPTICS
        as a fraction of the total number of data points. Defaults to 0.1.
    :return: A new DataFrame with an additional column named 'cluster',
        containing the cluster labels assigned to each data point.
    :rtype: pandas.DataFrame
    """
    if context_cols is None:
        context_cols = []

    feature_cols = [col for col in df.columns if col not in context_cols]
    X = df[feature_cols].values
    X_scaled = StandardScaler().fit_transform(X)

    # ↓ Riduzione dimensionale
    if reduction == 'umap':
        reducer = umap.UMAP(n_components=5)#, random_state=42)
        X_reduced = reducer.fit_transform(X_scaled)
    elif reduction == 'pca':
        pca = PCA(n_components=0.9)
        X_reduced = pca.fit_transform(X_scaled)
    else:
        X_reduced = X_scaled

    # ↓ OPTICS clustering
    optics = OPTICS(min_samples=min_pts, xi=xi, min_cluster_size=min_cluster_size)
    cluster_labels = optics.fit_predict(X_reduced)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    if n_clusters != 1:
        print(f"OPTICS ha trovato {n_clusters} cluster")
        # ↓ Silhouette
        valid_mask = cluster_labels != -1
        # se il numero dei punti non di rumore maggiore di noise_perc allora calcolo lo score
        if valid_mask.sum() < df.shape[0] * noise_perc:
            score = silhouette_score(X_reduced[valid_mask], cluster_labels[valid_mask])
            print(f"Silhouette score (OPTICS, senza rumore): {score:.3f}")
            print(valid_mask.sum(), df.shape[0]*(1-noise_perc))
        else:
            print("Troppi punti classificati come rumore per calcolare il silhouette score")
            visualize = False
            print(valid_mask.sum(), df.shape[0]*(1-noise_perc))
    else:
        visualize = False

    if visualize:
        # Visualizzazione 2D
        reducer_2d = umap.UMAP(n_components=2, random_state=42)
        X_2d = reducer_2d.fit_transform(X_scaled)

        plt.figure(figsize=(8, 6))
        for label in np.unique(cluster_labels):
            idx = cluster_labels == label
            color = 'black' if label == -1 else None
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"Cluster {label}", alpha=0.7, color=color)
        plt.title("Clustering con OPTICS + UMAP (2D)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Reachability Plot
        reachability = optics.reachability_
        ordering = optics.ordering_

        plt.figure(figsize=(12, 5))
        space = np.arange(len(X_scaled))
        for klass in np.unique(cluster_labels):
            color = 'k' if klass == -1 else plt.cm.tab10(klass % 10)
            Xk = space[cluster_labels[ordering] == klass]
            Rk = reachability[ordering][cluster_labels[ordering] == klass]
            plt.plot(Xk, Rk, color, marker='.', linestyle='None', markersize=3, label=f"Cluster {klass}")
        plt.plot(space, reachability[ordering], 'k-', alpha=0.5)
        plt.title('Reachability Plot (OPTICS)')
        plt.ylabel('Reachability distance')
        plt.xlabel('Punti ordinati')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Output finale
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    return df_clustered
