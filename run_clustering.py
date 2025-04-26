from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#todo: standardizzare e normalizzare tutte le features
def run_clustering(df, n_clusters=3, visualize=True):
    """
    Applica clustering KMeans al DataFrame df con n_clusters specificato.
    Ritorna il DataFrame con la colonna 'cluster' aggiunta.
    """
    # 1. Estrai solo le feature numeriche (tutto tranne le colonne di contesto)
    context_cols = ['player', 'point_type', 'surface', 'phase']
    feature_cols = [col for col in df.columns if col not in context_cols]

    X = df[feature_cols].values

    # 2. Standardizza le feature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Applica KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_scaled)

    # 4. Aggiungi etichette al DataFrame originale
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels

    # 5. PCA per visualizzazione 2D
    if visualize:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

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

    return df_with_clusters
