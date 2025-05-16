from typing import Dict, List, Tuple, Counter
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances


def aggregate_features_by_cluster(df_clustered: pd.DataFrame, cluster_col='cluster', agg_funcs=None):
    """
    Calcola feature aggregate per ogni cluster.

    Parameters:
        df_clustered (pd.DataFrame): Dataset con le feature e le etichette di cluster.
        cluster_col (str): Nome della colonna che contiene le etichette dei cluster.
        agg_funcs (dict or list): Funzioni di aggregazione da applicare (default: media e std).

    Returns:
        pd.DataFrame: Tabella con le feature aggregate per ogni cluster.
    """
    if agg_funcs is None:
        agg_funcs = ['mean']

    # Rimuovi eventuali colonne non numeriche (o adattalo se hai categoriche)
    numeric_cols = [col for col in df_clustered if col != cluster_col and col != 'player']

    # Group by cluster e calcolo aggregati
    grouped = df_clustered.groupby(cluster_col)[numeric_cols].agg(agg_funcs)

    # Pulizia del multi-index se più funzioni
    if isinstance(agg_funcs, list) and len(agg_funcs) > 1:
        grouped.columns = ['_'.join(col) for col in grouped.columns]

    return grouped.reset_index()


def cluster_feature_deltas(df, cluster_col='cluster'):
    """
    Calcola le deviazioni delle feature dei centroidi dalla media globale.

    Parameters:
        df (pd.DataFrame): Dataset con le feature e le etichette di cluster.
        cluster_col (str): Nome della colonna che contiene le etichette dei cluster.

    Returns:
        pd.DataFrame: Tabella con le deviazioni delle feature per ogni cluster.
    """

    # Calcola la media globale di ogni colonna (escludendo la colonna cluster e player)
    mean_all = df.drop(columns=[cluster_col, "player"]).mean()

    # Calcola il centroide di ogni cluster (media delle feature per cluster)
    cluster_means = df.groupby(cluster_col).mean(numeric_only=True)

    # Calcola le deviazioni
    deltas = cluster_means - mean_all

    return deltas


def analyze_cluster_profiles(df, cluster_col='cluster', point_context="on_serve", top_n=5):
    """
    Analizza i profili dei cluster e stampa le feature più importanti.

    Parameters:
        df (pd.DataFrame): Dataset con le feature e le etichette di cluster.
        cluster_col (str): Nome della colonna che contiene le etichette dei cluster.
        top_n (int): Numero di feature più importanti da stampare per cluster.
    """

    deltas = cluster_feature_deltas(df, cluster_col)

    # Calcola la deviazione standard di ogni feature sull'intero dataset
    feature_std = df.drop(columns=[cluster_col, 'player']).std()

    for cluster in deltas.index:
        print(f"\n--- Cluster {cluster} ---")
        cluster_deltas = deltas.loc[cluster]

        # Normalizza le deviazioni usando Z-score
        normalized_deltas = cluster_deltas / feature_std

        # Ordina le deviazioni normalizzate per valore assoluto
        important_features = normalized_deltas.abs().sort_values(ascending=False)
        describing_features = ['average_shot_length', 'net_points_rate',
                               'net_points_won_rate', 'winners_rate',
                               'unforced_errors_rate', 'slices_rate',
                               'dropshots_rate']
        if point_context == "on response":
            describing_features.append('average_response_depth')
        else:
            describing_features.append('ace_rate')
        # print(f"Feature più importanti (top {top_n}):")
        # for feature, delta in important_features.head(top_n).items():
        #   sign = "+" if normalized_deltas[feature] > 0 else "-"
        #   print(f"  {feature}: {sign} {delta:.4f}")
        for feature in describing_features:
            if feature in normalized_deltas:
                delta = normalized_deltas[feature]
                sign = "+" if delta > 0 else "-"
                print(f"  {feature}: {sign} {abs(delta):.4f}")

        # Ottieni e stampa i giocatori nel cluster corrente
        players_in_cluster = df[df[cluster_col] == cluster]['player'].unique().tolist()
        print(f"\nGiocatori nel Cluster {cluster}:")
        if players_in_cluster:
            for player_name in players_in_cluster:
                print(f"  - {player_name}")
        else:
            print("  Nessun giocatore trovato per questo cluster.")


def calculate_centroid_similarity(
        clustered_data,
        metric: str = "euclidean"
) -> pd.DataFrame:
    """
    Calcola la similarità tra i centroidi dei cluster di contesti diversi.

    Parameters:
        clustered_data (list): lista di tuple, dove il primo elemento della
         tupla è il contesto, il secondo il dataframe con le etichette dei clusters
        metric (str): Metrica di distanza/similarità da usare
            ('euclidean', 'manhattan', 'correlation', 'js', 'wasserstein').
            'js' per Jensen-Shannon, 'wasserstein' per Wasserstein.

    Returns:
        pd.DataFrame: DataFrame contenente la matrice di similarità tra i cluster.
    """

    all_centroids = {}
    for context, df_clustered in clustered_data:
        if not df_clustered.empty:
            # Usa la funzione aggregate_features_by_cluster per ottenere i centroidi
            centroids = aggregate_features_by_cluster(df_clustered)
            all_centroids[context] = centroids

    # Calcola tutte le coppie di cluster tra contesti diversi
    cluster_pairs = [
        (context1, cluster1, context2, cluster2)
        for context1, centroids1 in all_centroids.items()
        for cluster1 in centroids1.index
        for context2, centroids2 in all_centroids.items()
        for cluster2 in centroids2.index
        if context1 != context2  # Evita confronti tra cluster dello stesso contesto
    ]

    similarity_data = []
    for context1, cluster1, context2, cluster2 in cluster_pairs:
        # Verifica che i contesti e i cluster siano validi
        if (
                context1 in all_centroids
                and context2 in all_centroids
                and cluster1 in all_centroids[context1].index
                and cluster2 in all_centroids[context2].index
        ):
            centroid1 = all_centroids[context1].loc[cluster1].values.reshape(1, -1)
            centroid2 = all_centroids[context2].loc[cluster2].values.reshape(1, -1)

            # Calcola la similarità/distanza in base alla metrica scelta
            if metric == "euclidean":
                similarity = 1 - pairwise_distances(centroid1, centroid2, metric="euclidean")[
                    0][0]
            else:
                raise ValueError(f"Metrica non supportata: {metric}")

            similarity_data.append(
                {
                    "context1": context1,
                    "cluster1": cluster1,
                    "context2": context2,
                    "cluster2": cluster2,
                    "similarity": similarity,
                }
            )

    similarity_df = pd.DataFrame(similarity_data)
    return similarity_df


def visualize_similarity_matrix(
        similarity_df: pd.DataFrame,
        clustered_data,
        metric_name: str = "Euclidean",
        show_plot: bool = True) -> None:
    """
    Visualizza la matrice di similarità tra i cluster usando un heatmap.

    Parameters:
        similarity_df (pd.DataFrame): DataFrame contenente le similarità tra i cluster.
        clustered_data (List): Lista di tuple, dove il primo elemento è il contesto
         e il secondo il dataframe dei dati clusterizzati per i contesti.
        metric_name (str): Nome della metrica usata per la visualizzazione.
        show_plot (bool): Flag per decidere se mostrare il plot.
    """
    # Crea un insieme di tutti i contesti e cluster univoci
    all_contexts = set()
    all_clusters = set()
    for context, df_clustered in clustered_data:
        if not df_clustered.empty:
            all_contexts.add(context)
            all_clusters.update(df_clustered["cluster"].unique())

    # Crea un indice completo e colonne per la tabella pivot
    index = pd.MultiIndex.from_product(
        [sorted(all_contexts), sorted(all_clusters)], names=["context1", "cluster1"]
    )
    columns = pd.MultiIndex.from_product(
        [sorted(all_contexts), sorted(all_clusters)], names=["context2", "cluster2"]
    )

    # Crea una tabella pivot con indici e colonne completi
    pivot_df = similarity_df.pivot_table(
        index=["context1", "cluster1"],
        columns=["context2", "cluster2"],
        values="similarity",
    ).reindex(index=index, columns=columns)

    if show_plot:
        plt.figure(figsize=(16, 14))  # Aumenta la dimensione della figura
        sns.heatmap(pivot_df, annot=True, cmap="viridis")
        plt.title(f"Similarità tra Cluster ({metric_name} Distance)")
        plt.xlabel("Contesto2, Cluster2")
        plt.ylabel("Contesto1, Cluster1")
        plt.show()


def create_player_trajectories(
        clustered_data,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Crea le traiettorie dei giocatori attraverso i cluster nei diversi contesti.

    Parameters:
        clustered_data (List): Lista di tuple dove il prim elemento è il contesto
        mentre il secondo è il DataFrame con le etichette dei cluster, e una colonna 'player'.

    Returns:
        dict: Dizionario dove le chiavi sono i giocatori e i valori sono le loro
            traiettorie (lista di tuple: (contesto, cluster)).
    """
    player_trajectories = {}
    for context, df_clustered in clustered_data.items():
        if not df_clustered.empty:
            for player, cluster in df_clustered[["player", "cluster"]].values:
                if player not in player_trajectories:
                    player_trajectories[player] = []
                player_trajectories[player].append((context, cluster))
    return player_trajectories


def visualize_player_trajectories(
        player_trajectories: Dict[str, List[Tuple[str, int]]],
        top_n: int = 10,
        print_trajectories: bool = True
) -> None:
    """
    Visualizza le traiettorie dei giocatori più frequenti.

    Parameters:
        player_trajectories (dict): Dizionario delle traiettorie dei giocatori.
        top_n (int): Numero di traiettorie più frequenti da visualizzare.
        print_trajectories (bool): Flag per decidere se stampare le traiettorie.
    """
    # Conta le frequenze delle traiettorie
    trajectory_counts = Counter(tuple(traj) for traj in player_trajectories.values())

    # Ottieni le traiettorie più comuni
    most_common_trajectories = trajectory_counts.most_common(top_n)

    if print_trajectories:
        print("\n--- Traiettorie dei Giocatori ---")
        for i, (trajectory, count) in enumerate(most_common_trajectories):
            print(f"\nTraiettoria {i + 1} (Count: {count})")
            for context, cluster in trajectory:
                print(f"  {context}: Cluster {cluster}")
