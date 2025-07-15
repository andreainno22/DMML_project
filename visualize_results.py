import os
from typing import Dict, List, Tuple, Counter
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np

cluster_description = {"on response_ on clay, cluster 0": "Defensive, consistent from the baseline.",
                       "on response_ on clay, cluster 1": "Aggressive since the response, variety of shots, high number of net discents",
                       "on response_ on clay, cluster 2": "High-risk, aggressive baseliner, high usage of dropshot, not good responder to the service",
                       "on response_ on clay, cluster 3": "Excelent service responder, conservative and low error-prone, high number of net discents",
                       "on response_ on grass, cluster 0": "High-risk, aggressive baseliner, high usage of dropshot",
                       "on response_ on grass, cluster 1": "Defensive, consistent from the baseline, sufference in the net descents",
                       "on response_ on grass, cluster 2": "Very good responder to the service, very good in net points but low net descents, low-risk point builder",
                       "on response_ on grass, cluster 3": "Aggressive, low error-prone, bad service responder",
                       "on response_ on hard, cluster 0": "Defensive, untouchable from the baseline, low error-prone, dropshots user",
                       "on response_ on hard, cluster 1": "High-risk, aggressive baseliner, he tries to shorten the point",
                       "on response_ on hard, cluster 2": "Aggressive but low error-prone, low usage of slice shots",
                       "on response_ on hard, cluster 3": "Defensive, slice constructor, high number net discents",
                       "on serve_ on clay, cluster 0": "Aggressive, high-risk from the baseline",
                       "on serve_ on clay, cluster 1": "Defensive, low-risk from the baseline",
                       "on serve_ on clay, cluster 2": "Defensive, low-risk, variety of shots",
                       "on serve_ on clay, cluster 3": "Big server, serve and volley user",
                       "on serve_ on grass, cluster 0":"Error-prone, not big server",
                       "on serve_ on grass, cluster 1":"Good server, silce user, serve and volley user",
                       "on serve_ on grass, cluster 2":"Defensive, low error-prone, baseline builder",
                       "on serve_ on grass, cluster 3":"Big server, aggressive",
                       "on serve_ on hard, cluster 0":"Big server, aggressive",
                       "on serve_ on hard, cluster 1":"Good server, serve and volley user",
                       "on serve_ on hard, cluster 2":"Defensive, low-risk, variety of shots",
                       "on serve_ on hard, cluster 3":"Not a big server, defensive"}


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
    mean_all = df.mean(numeric_only=True)

    # Calcola il centroide di ogni cluster (media delle feature per cluster)
    cluster_means = df.groupby(cluster_col).mean(numeric_only=True)

    # Calcola le deviazioni in percentuale
    deltas_percent = (cluster_means - mean_all) / mean_all

    return deltas_percent


def analyze_cluster_profiles(df, context, cluster_col='cluster', cluster_id=None,
                             all_players=False):
    """
    Analizza i profili dei cluster e stampa le feature più importanti.

    Parameters:
        df (pd.DataFrame): Dataset con le feature e le etichette di cluster.
        cluster_col (str): Nome della colonna che contiene le etichette dei cluster.
        :param cluster_id:
        :param context:
    """

    deltas_in_percentage = cluster_feature_deltas(df, cluster_col)

    # Calcola la deviazione standard di ogni feature sull'intero dataset
    feature_std = df.drop(columns=[cluster_col, 'player']).std()

    # calcola la media di ogni feature nell'intero dataset
    feature_mean = df.drop(columns=[cluster_col, 'player']).mean()

    if cluster_id is not None:
        deltas_in_percentage = deltas_in_percentage[deltas_in_percentage.index == cluster_id]

    for cluster in deltas_in_percentage.index:
        key = context+f", cluster {cluster}"
        print(f"\n--- Cluster {cluster} ---\n {cluster_description[key]}")
        cluster_deltas = deltas_in_percentage.loc[cluster]

        # Ordina le deviazioni normalizzate per valore assoluto
        describing_features = ['average_shot_length', 'net_points_rate',
                               'net_points_won_rate', 'winners_rate',
                               'unforced_errors_rate', 'slices_rate',
                               'dropshots_rate']
        if "on response" in context:
            describing_features.append('average_response_depth')
        else:
            describing_features.append('ace_rate')

        # stampa di quanto le features del centroide del cluster si discostano dalla media generale
        for feature in describing_features:
            if feature in deltas_in_percentage:
                delta = cluster_deltas[feature]
                sign = "+" if delta > 0 else "-"
                print(f"  {feature}: {sign} {abs(delta):.4f}")

        if all_players:
            # Ottieni e stampa i giocatori nel cluster corrente
            players_in_cluster = df[df[cluster_col] == cluster]['player'].unique().tolist()
            print(f"\nPlayers in the cluster {cluster}:")
        else:
            # stampa solo i 5 giocatori più importanti del cluster
            file_path = os.path.join('top_players_per_context', f"{context}, cluster {cluster}")
            players_in_cluster_df = pd.read_csv(file_path, low_memory=False)
            players_in_cluster = players_in_cluster_df['player'].tolist()
            print(f"\nTop players in the cluster {cluster}:")
        for player_name in players_in_cluster:
            print(f"  - {player_name}")


def calculate_centroid_similarity(
        clustered_data: List[Tuple[str, pd.DataFrame]],
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
        clustered_data: List[Tuple[str, pd.DataFrame]],
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
        plt.title(f"Similarity between Clusters ({metric_name} Distance)")
        plt.xlabel("Context2, Cluster2")
        plt.ylabel("Context1, Cluster1")
        plt.show()


def create_player_trajectories(
        clustered_data: List[Tuple[str, pd.DataFrame]],
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
    for context, df_clustered in clustered_data:
        if not df_clustered.empty:
            for player, cluster in df_clustered[["player", "cluster"]].values:
                if player not in player_trajectories:
                    player_trajectories[player] = []
                player_trajectories[player].append((context, cluster))
    return player_trajectories


def display_top_player_trajectories(
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


def visualize_player_trajectory(
        player_trajectories: Dict[str, List[Tuple[str, int]]],
        player: str,
        dfs_clustered: List[Tuple[str, pd.DataFrame]],  # Aggiungi il DataFrame come parametro
        cluster_col: str = 'cluster',  # Aggiungi cluster_col come parametro
):
    """
    Visualizza la traiettoria del giocatore utilizzando i profili dei cluster.

    Parameters:
        player_trajectories (Dict[str, List[Tuple[str, int]]]): Dizionario delle traiettorie dei giocatori.
        player (str): Nome del giocatore da visualizzare.
        cluster_col (str): Nome della colonna che contiene le etichette dei cluster.
        :param dfs_clustered:
    """
    if player not in player_trajectories:
        print(f"Player {player} not found.")
        return

    trajectory = player_trajectories[player]
    print(f"Player trajectory for {player}:\n")
    for context, cluster in trajectory:
        context_cleaned = context.replace(".csv", "")
        context_to_print = context_cleaned.replace("_", "")
        print(f"\nContext = {context_to_print}, Cluster = {cluster}")
        df = next(df for ctx, df in dfs_clustered if ctx == context)
        # Analizza il profilo del cluster
        analyze_cluster_profiles(df, context=context_cleaned, cluster_col=cluster_col, cluster_id=cluster)
