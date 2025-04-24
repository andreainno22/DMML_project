import numpy as np
import pandas as pd


# todo: cambiare il dataset e fare wide encoding con info di contesto
# todo: integrale le feature generali: surface con + perc di vittoria, lunghezza media dei colpi, direzione preferita al servizio
# todo: slice, dropshot vanno considerati in generale e non nei pattern, perchè rischiano di non apparire nei top patterns
def extract_aggregated_features(patterns_list):
    """
    Estrae feature numeriche aggregate da una lista di pattern frequenti.
    Include percentuali PONDERATE (in base al supporto) di:
    forehand, backhand, volley, drop shot, slice, e colpi profondi.
    """
    import numpy as np

    if not patterns_list:
        return np.zeros(9)

    wins = []
    supports = []
    total_weighted_shots = 0

    # Pesati per supporto
    f_count = b_count = v_count = drop_count = slice_count = deep_count = 0

    for seq, support, win, _ in patterns_list:
        wins.append(win)
        supports.append(support)

        for shot in seq:
            if len(shot) < 1:
                continue
            shot_type = shot[0]
            # shot depth non viene considerato, pensare se usarlo solo per la risposta al servizio
            # shot_depth = shot[2] if len(shot) > 2 else None

            total_weighted_shots += support  # ogni colpo pesato per ricorrenza

            # Forehand e backhand (con varianti)
            if shot_type in ['f', 'r', 'l', 'u', 'h', 'j']:
                f_count += support
            elif shot_type in ['b', 's', 'm', 'y', 'i', 'k']:
                b_count += support

            # Volley
            if shot_type in ['v', 'z']:
                v_count += support

            # Drop shot
            if shot_type in ['u', 'y']:
                drop_count += support

            # Slice
            if shot_type in ['r', 's']:
                slice_count += support

    # Evita divisione per 0
    if total_weighted_shots == 0:
        return np.zeros(9)

    return np.array([
        # feature inutile il numero medio di vittorie
        # np.mean(wins),
        b_count / total_weighted_shots,
        f_count / total_weighted_shots,
        v_count / total_weighted_shots,
        drop_count / total_weighted_shots,
        slice_count / total_weighted_shots,
        len(patterns_list),
        np.mean(supports)
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
            context_type = " ".join(context.split()[1:-1])  # es: on serve with the 1st
            surface = context.split()[-1]  # usa la variabile globale
            phase = "opening"  # per ora fissa, può diventare parametro in futuro

            # --- estrazione feature
            agg_features = extract_aggregated_features(pattern_list)  # vettore np.array di shape (9,)
            pattern_counts = pattern_counts_by_context.get(context, [0] * len(top_patterns))

            row = {
                "player": player,
                "point_type": context_type,
                "surface": surface,
                "phase": phase,
                "%backhand": agg_features[0],
                "%forehand": agg_features[1],
                "%volley": agg_features[2],
                "%drop_shot": agg_features[3],
                "%slice": agg_features[4],
                "pattern_variety": agg_features[5],
                "avg_support": agg_features[6],
            }

            # Aggiungi le feature dei pattern (una colonna per ciascuno)
            for i, pattern in enumerate(top_patterns):
                pattern_str = "pattern_" + "_".join(pattern)  # es: pattern_f36_f17_b15
                row[pattern_str] = pattern_counts[i]

            rows.append(row)

    return pd.DataFrame(rows)
