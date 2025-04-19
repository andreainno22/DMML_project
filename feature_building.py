import numpy as np
import pandas as pd
from costants import SURFACE


# todo: cambiare il dataset e fare wide encoding con info di contesto
def extract_aggregated_features(patterns_list):
    """
    Estrae feature numeriche aggregate da una lista di pattern frequenti.
    Ora include anche % drop shot e % slice.
    """
    from get_mapping_dictionaries import get_mapping_dictionaries

    shot_types_map = get_mapping_dictionaries("shot_types")
    shot_depth_map = get_mapping_dictionaries("shot_depth")

    if not patterns_list:
        return np.zeros(9)

    wins = []
    supports = []
    total_shots = 0
    n_forehand = n_backhand = n_volley = 0
    n_deep = 0
    n_dropshot = 0
    n_slice = 0
    n_out_wides = 0
    n_body = 0
    n_down_the_T = 0

    for seq, support, win, _ in patterns_list:
        wins.append(win)
        supports.append(support)

        for shot in seq:

            shot_type = shot[0]

            total_shots += 1

            if shot_type in '4':  # serve
                n_out_wides += 1
            elif shot_type in '5':
                n_body += 1
            elif shot_type in '6':
                n_down_the_T += 1
            # Tipo colpo
            if shot_type in ['f', 'r', 'l', 'u', 'h', 'j']:  # forehand e varianti
                n_forehand += 1
            elif shot_type in ['b', 's', 'm', 'y', 'i', 'k']:  # backhand e varianti
                n_backhand += 1
            if shot_type in ['v', 'z']:  # volley
                n_volley += 1
            if shot_type in ['u', 'y']:  # drop shots
                n_dropshot += 1
            if shot_type in ['r', 's']:  # slice
                n_slice += 1

    return np.array([
        np.mean(wins),
        n_backhand / total_shots if total_shots else 0,
        n_forehand / total_shots if total_shots else 0,
        n_volley / total_shots if total_shots else 0,
        n_dropshot / total_shots if total_shots else 0,
        n_slice / total_shots if total_shots else 0,
        n_deep / total_shots if total_shots else 0,
        len(patterns_list),  # varietà dei pattern
        np.mean(supports)  # supporto medio
    ])


def build_final_dataset(top_patterns, pattern_counts_by_context, all_pattern_lists):
    """
    Costruisce il DataFrame finale per il clustering, unendo:
    - feature aggregate
    - frequenze dei top pattern
    - informazioni su giocatore, contesto, superficie, fase del punto

    Returns:
        pd.DataFrame con una riga per ogni (player + context), pronto per il clustering.
    """
    rows = []

    for pattern_dict in all_pattern_lists:
        for context, pattern_list in pattern_dict.items():
            # --- parsing delle info di contesto
            player = context.split()[0]
            context_type = " ".join(context.split()[1:-2])  # es: on serve with the 1st
            surface = SURFACE  # usa la variabile globale
            phase = "opening"  # per ora fissa, può diventare parametro in futuro

            # --- estrazione feature
            agg_features = extract_aggregated_features(pattern_list)  # vettore np.array di shape (9,)
            pattern_counts = pattern_counts_by_context.get(context, [0] * len(top_patterns))

            row = {
                "player": player,
                "point_type": context_type,
                "surface": surface,
                "phase": phase,
                "avg_win_rate": agg_features[0],
                "%backhand": agg_features[1],
                "%forehand": agg_features[2],
                "%volley": agg_features[3],
                "%drop_shot": agg_features[4],
                "%slice": agg_features[5],
                "%deep_shots": agg_features[6],
                "pattern_variety": agg_features[7],
                "avg_support": agg_features[8],
            }

            # Aggiungi le feature dei pattern (una colonna per ciascuno)
            for i, pattern in enumerate(top_patterns):
                pattern_str = "pattern_" + "_".join(pattern)  # es: pattern_f36_f17_b15
                row[pattern_str] = pattern_counts[i]

            rows.append(row)

    return pd.DataFrame(rows)
