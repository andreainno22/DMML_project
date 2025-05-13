import pandas as pd


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
        agg_funcs = ['mean', 'std']

    # Rimuovi eventuali colonne non numeriche (o adattalo se hai categoriche)
    numeric_cols = [col for col in df_clustered if col != cluster_col]

    # Group by cluster e calcolo aggregati
    grouped = df_clustered.groupby(cluster_col)[numeric_cols].agg(agg_funcs)

    # Pulizia del multi-index se più funzioni
    if isinstance(agg_funcs, list) and len(agg_funcs) > 1:
        grouped.columns = ['_'.join(col) for col in grouped.columns]

    return grouped.reset_index()


def cluster_feature_deltas(df, cluster_col='cluster'):
    mean_all = df.drop(columns=[cluster_col]).mean()
    cluster_means = df.groupby(cluster_col).mean()
    return cluster_means.subtract(mean_all)


