import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from random import sample
from numpy.random import uniform
import umap
import warnings
import seaborn as sns



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
    

def run_kmeans_clustering(df, n_clusters=4, visualize=True, dimensionality_reduction="no reduction"):

    feature_cols = [col for col in df.columns if col not in ["player"]]
    values = df[feature_cols].values
    
    # 3. KMeans
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    cluster_labels = kmeans.fit_predict(values)

    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels

    # 4. Valutazione silhouette
    score = silhouette_score(values, cluster_labels)
    print(f"Silhouette score (KMeans): {score:.3f}")

    # Calcolo del DBI
    dbi = davies_bouldin_score(values, cluster_labels)
    print(f"DBI score (KMeans): {dbi:.3f}")

    # 5. Visualizzazione clusters
    if visualize:
        if dimensionality_reduction == "umap8":
            reducer = umap.UMAP(n_components=2)
            X_visualize = reducer.fit_transform(values)
        elif dimensionality_reduction == "pca" or dimensionality_reduction == "no reduction":
            pca = PCA(n_components=2)
            X_visualize = pca.fit_transform(values)
        else:
            X_visualize = values
            
        plt.figure(figsize=(8, 6))
        for cluster_id in range(n_clusters):
            idx = df_with_clusters['cluster'] == cluster_id
            plt.scatter(X_visualize[idx, 0], X_visualize[idx, 1], label=f"Cluster {cluster_id}", alpha=0.7)

        plt.title("Clustering with KMeans + "+dimensionality_reduction)
        plt.xlabel(dimensionality_reduction+" 1")
        plt.ylabel(dimensionality_reduction+" 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return 


def run_kmeans_clustering_multiple(df, n_clusters=4, n_runs=50, visualize=True, dimensionality_reduction="no reduction"):
    feature_cols = [col for col in df.columns if col not in ["player"]]
    values = df[feature_cols].values

    best_score = -np.inf
    best_model = None
    best_labels = None
    best_seed = None

    for seed in range(n_runs):
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
        cluster_labels = kmeans.fit_predict(values)
        score = silhouette_score(values, cluster_labels)

        if score > best_score:
            best_score = score
            best_model = kmeans
            best_labels = cluster_labels
            best_seed = seed

    # Applica il clustering migliore
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = best_labels

    # Valutazione finale
    print(f"Best seed: {best_seed}")
    print(f"Silhouette score (best): {best_score:.3f}")
    dbi = davies_bouldin_score(values, best_labels)
    print(f"DBI score (best): {dbi:.3f}")

    # Visualizzazione del clustering migliore
    if visualize:
        if dimensionality_reduction == "umap8":
            reducer = umap.UMAP(n_components=2)
            X_visualize = reducer.fit_transform(values)
        elif dimensionality_reduction == "pca":
            pca = PCA(n_components=2)
            X_visualize = pca.fit_transform(values)
        else:
            X_visualize = values

        plt.figure(figsize=(8, 6))
        for cluster_id in range(n_clusters):
            idx = df_with_clusters['cluster'] == cluster_id
            plt.scatter(X_visualize[idx, 0], X_visualize[idx, 1], label=f"Cluster {cluster_id}", alpha=0.7)

        plt.title(f"KMeans Clustering (best seed={best_seed}) + {dimensionality_reduction}")
        plt.xlabel(f"{dimensionality_reduction} 1")
        plt.ylabel(f"{dimensionality_reduction} 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return df_with_clusters


def run_umap_gmm_clustering(df, n_clusters=4,
                            proba_threshold=0.9, visualize=True, umap_comp=2):
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
    context_cols = ["player"]

    feature_cols = [col for col in df.columns if col not in context_cols]
    values = df[feature_cols].values

    # GMM
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full')
    gmm.fit(values)

    # Probabilities of each point of belonging to the clusters
    cluster_probs = gmm.predict_proba(values)
    max_probs = cluster_probs.max(axis=1)
    predicted_labels = gmm.predict(values)

    # Probability threshold application: noise if below the treshold
    cluster_labels = np.where(max_probs < proba_threshold, -1, predicted_labels)

    if visualize:
        sns.histplot(max_probs, bins=20)
        plt.title("Max Cluster Membership Probability Distribution")
        plt.show()

    # 5. Silhouette solo sui punti validi
    valid_mask = cluster_labels != -1
    sil_score = silhouette_score(values[valid_mask], cluster_labels[valid_mask])
    print(f"Silhouette score (UMAP + GMM): {sil_score:.3f}")
    dbi_score = davies_bouldin_score(values[valid_mask], cluster_labels[valid_mask])
    print(f"DBI score (UMAP + GMM): {dbi_score:.3f}")


    # 6. Visualizzazione
    if visualize:
        if umap_comp==8:
            reducer = umap.UMAP(n_components=2)
            X_visualize = reducer.fit_transform(values)
        else:
            X_visualize = values
        plt.figure(figsize=(8, 6))
        for label in np.unique(cluster_labels):
            idx = cluster_labels == label
            color = 'black' if label == -1 else None
            plt.scatter(X_visualize[idx, 0], X_visualize[idx, 1], label=f"Cluster {label}", alpha=0.7, color=color)
        plt.title("Clustering UMAP + GMM")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 7. Ritorno
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    return df_clustered


def run_umap_agglomerative_clustering(df, n_clusters=4, linkage='ward', visualize=True, umap_comp = 2):
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
    context_cols = ["player"]

    # 1. Estrai solo le feature numeriche
    feature_cols = [col for col in df.columns if col not in context_cols]
    values = df[feature_cols].values

    # 3. Agglomerative Clustering
    agglom = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    cluster_labels = agglom.fit_predict(values)

    # 4. Silhouette score
    sil_score = silhouette_score(values, cluster_labels)
    print(f"Silhouette score (Agglomerative + UMAP): {sil_score:.3f}")

    # Calcolo del DBI
    dbi = davies_bouldin_score(values, cluster_labels)
    print(f"Dbi score (Agglomerative + UMAP): {dbi:.3f}")

    # 5. Visualizzazione 2D 
    if visualize:
        if umap_comp == 8:
            reducer_2d = umap.UMAP(n_components=2)
            X_2d = reducer_2d.fit_transform(values)
        else:
            X_2d = values
        plt.figure(figsize=(8, 6))
        for label in np.unique(cluster_labels):
            idx = cluster_labels == label
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"Cluster {label}", alpha=0.7)
        plt.title("Agglomerative Clustering (2D UMAP)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 6. Restituzione
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    return df_clustered


def run_umap_optics_clustering(df, reduction='umap', min_pts=3, xi=0.05,
                               min_cluster_size=0.1, visualize=True, noise_perc=0.30, umap_comp = 2):
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
        clustering. Defaults to 5.
    :param xi: A parameter for OPTICS that determines the minimum steepness on the
        reachability plot to identify clusters. Defaults to 0.08.
    :param min_cluster_size: The minimum relative size of a cluster in OPTICS
        as a fraction of the total number of data points. Defaults to 0.1.
    :return: A new DataFrame with an additional column named 'cluster',
        containing the cluster labels assigned to each data point.
    :rtype: pandas.DataFrame
    """

    context_cols = ["player"]

    feature_cols = [col for col in df.columns if col not in context_cols]

    values = df[feature_cols].values

    # OPTICS clustering
    optics = OPTICS(min_samples=min_pts, xi=xi, min_cluster_size=min_cluster_size)
    cluster_labels = optics.fit_predict(values)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print("Number of clusters: ", n_clusters)
    if n_clusters != 1:
        print(f"OPTICS found {n_clusters} cluster")
        valid_mask = cluster_labels != -1
        invalid_mask = cluster_labels == -1
        # se il numero dei punti di rumore è minore del numero di punti accettati come rumore allora calcolo lo score
        if invalid_mask.sum() < df.shape[0] * noise_perc:
            
            silhouette = silhouette_score(values[valid_mask], cluster_labels[valid_mask])
            print(f"Silhouette score (OPTICS, withoute noise): {silhouette:.3f}")
            dbi = davies_bouldin_score(values[valid_mask], cluster_labels[valid_mask])
            print(f"DBI score (OPTICS, withoute noise): {dbi:.3f}")

            print("Noise points: "+ str(invalid_mask.sum())+ "\nAccepted max number of point classified as noise: " + str(df.shape[0]*(noise_perc))) 
        else:
            print("The noise percentage exceed the user defined treshold.")
            visualize = True
            print("Noise points: "+ str(invalid_mask.sum())+ "\nAccepted max number of point classified as noise, based on noise percentage: " + str(df.shape[0]*(noise_perc)))

    if visualize:
        if reduction == "umap" and umap_comp==8:
            # Visualizzazione 2D
            reducer_2d = umap.UMAP(n_components=2)
            X_2d = reducer_2d.fit_transform(values)
        elif reduction == "pca":
            pca = PCA(n_components=2)  
            X_2d = pca.fit_transform(values)
        else:
            X_2d = values

        plt.figure(figsize=(8, 6))
        for label in np.unique(cluster_labels):
            idx = cluster_labels == label
            color = 'black' if label == -1 else None
            plt.scatter(X_2d[idx, 0], X_2d[idx, 1], label=f"Cluster {label}", alpha=0.7, color=color)
        plt.title("Clustering with OPTICS + "+reduction)
        plt.xlabel(reduction+" 1")
        plt.ylabel(reduction+" 2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Reachability Plot
        reachability = optics.reachability_
        ordering = optics.ordering_

        plt.figure(figsize=(12, 5))
        space = np.arange(len(values))
        for klass in np.unique(cluster_labels):
            color = 'k' if klass == -1 else plt.cm.tab10(klass % 10)
            Xk = space[cluster_labels[ordering] == klass]
            Rk = reachability[ordering][cluster_labels[ordering] == klass]
            plt.plot(Xk, Rk, color, marker='.', linestyle='None', markersize=3, label=f"Cluster {klass}")
        plt.plot(space, reachability[ordering], 'k-', alpha=0.5)
        plt.title('Reachability Plot (OPTICS)')
        plt.ylabel('Reachability distance')
        plt.xlabel('Ordered points')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Output finale
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    return df_clustered
