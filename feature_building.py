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
    f_count = b_count = v_count = drop_count = slice_count = 0

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
        b_count / total_weighted_shots,
        f_count / total_weighted_shots,
        v_count / total_weighted_shots,
        drop_count / total_weighted_shots,
        slice_count / total_weighted_shots,
        len(patterns_list),
        np.mean(supports)
    ])


def build_generic_features(context, shots):
    # todo: finire di definire le feature generali, controllare se sono corrette o no
    total_points = shots.shape[0]
    volley_codes = {'v', 'z', 'o', 'p', 'h', 'i', 'j'}
    not_volley_codes = {'f', 'b', 's', 'r', 'u', 'y'}
    slice_codes = {'r', 's'}
    dropshots_codes = {'u', 'y'}

    if str.__contains__(context, "on serve"):
        # se len <= 2 vuol dire che c'è stato solo servizio e risposta o solo servizio
        total_points_started = shots[shots['shots'].apply(len) > 2].shape[0]
        # se len = 1 vuol dire che c'è stato o un ace o un doppio fallo, senza tentativo di risposta
        total_points_with_return = shots[shots['shots'].apply(len) > 1].shape[0]
        # numero di colpi totale escludendo il servizio
        total_no_service_shots = shots['shots'].apply(lambda seq: max(0, len(seq) - 1)).sum()
        # conta gli ace
        ace_num = shots[(shots['shots'].apply(len) == 1) & (shots['outcome'] == '*')].shape[0]
        ace_rate = ace_num / total_points
    else:
        # se len <= 1 vuol dire che c'è stato solo servizio e risposta o solo servizio
        total_points_started = shots[shots['shots'].apply(len) > 1].shape[0]
        # se len = 0 vuol dire che c'è stato o un ace o un doppio fallo, senza tentativo di risposta
        total_points_with_return = shots[shots['shots'].apply(len) > 0].shape[0]
        # numero di colpi totale escludendo il servizio
        total_no_service_shots = shots['shots'].apply(len).sum()
        # punto in risposta
        shot_depth_map = {'7': 1, '8': 2, '9': 3}
        # 2. Filtra i punti con almeno una risposta (lunghezza > 0)
        points_with_returns = shots[shots['shots'].apply(len) > 0].copy()  # Usa .copy() per sicurezza

        # 3. & 4. Estrai il secondo colpo e mappa al valore di profondità
        def get_numeric_depth(sequence):
            """Estrae il codice del secondo colpo e lo mappa a un valore numerico."""
            if len(sequence) > 0:
                if sequence[0][2] in shot_depth_map.keys():
                    second_shot_code = sequence[0][2]  # Indice 0 per il primo colpo (risposta)
                    # Usa .get() per ottenere il valore dalla mappa, restituendo None se il codice non c'è
                    return shot_depth_map.get(second_shot_code, None)
                else:
                    return None
            return None  # Non dovrebbe accadere per come ho filtrato, ma è sicuro

        # Applica la funzione per creare una nuova colonna con i valori di profondità
        # .loc per assegnare alla copia 'points_with_returns'
        points_with_returns.loc[:, 'return_depth_value'] = points_with_returns['shots'].apply(get_numeric_depth)

        # 5. Calcola la media dei valori di profondità validi
        # .mean() ignora automaticamente i valori NaN/None
        avg_resp_depth = points_with_returns['return_depth_value'].mean()

    avg_shot_length = shots['shots'].apply(len).mean()

    total_volley_points = shots[
        shots['shots'].apply(lambda sequence:
                             # Condizione 1: C'è almeno un codice volley nella sequenza?
                             (any(code in sequence for code in volley_codes) and
                              # Condizione 2: Il carattere '-' NON è presente nella sequenza?
                              '-' not in sequence
                              ) or
                             # oppure il punto è non volley
                             (any(code in sequence for code in not_volley_codes) and
                              # ma giocato a rete
                              '=' not in sequence))
    ].shape[0]
    net_points_rate = total_volley_points / total_points_started

    net_points_won = shots[(shots['PtWinner'] == 1) &
                           (shots['shots'].apply(lambda sequence:
                                                 # Condizione 1: C'è almeno un codice volley nella sequenza?
                                                 (any(code in sequence for code in
                                                      volley_codes) and
                                                  # Condizione 2: Il carattere '-' NON è presente nella sequenza?
                                                  '-' not in sequence
                                                  ) or
                                                 # oppure il punto è non volley
                                                 (any(code in sequence for code in
                                                      not_volley_codes) and
                                                  # ma giocato a rete
                                                  '=' not in sequence))
                            )].shape[0]
    net_points_won_rate = net_points_won / total_volley_points

    # escludo dai total points quelli per cui non c'è stata risposta
    winners = shots[(shots['PtWinner'] == 1) & (shots['outcome'] == '*') & (len(shots['shots']) > 1)].shape[0]
    winners_rate = winners / total_points_with_return

    # escludo dai total points quelli per cui non c'è stata risposta
    unforced_errors_rate = \
        shots[(shots['shots'].apply(len) > 1) & (shots['outcome'] == '#') & (shots['PtWinner'] == 0)].shape[
            0] / total_points_with_return

    per_sequence_shots_with_slice = shots['shots'].apply(
        lambda list_of_shots: sum(any(code in shot for code in slice_codes)
                                  for shot in list_of_shots))
    slices = per_sequence_shots_with_slice.sum()
    slices_rate = slices / total_no_service_shots

    per_sequence_shots_with_dropshot = shots['shots'].apply(
        lambda list_of_shots: sum(any(code in shot for code in dropshots_codes)
                                  for shot in list_of_shots))
    dropshots = per_sequence_shots_with_dropshot.sum()
    dropshots_rate = dropshots / total_no_service_shots

    per_sequence_shots_with_crosscourt = shots['shots'].apply(
        lambda list_of_shots: sum("1" in shot for shot in list_of_shots))
    crosscourt = per_sequence_shots_with_crosscourt.sum()
    crosscourt_rate = crosscourt / total_no_service_shots

    per_sequence_shots_with_middle = shots['shots'].apply(
        lambda list_of_shots: sum("2" in shot for shot in list_of_shots))
    middle = per_sequence_shots_with_middle.sum()
    middle_rate = middle / total_no_service_shots

    per_sequence_shots_with_dt_line = shots['shots'].apply(
        lambda list_of_shots: sum("3" in shot for shot in list_of_shots))
    dt_line = per_sequence_shots_with_dt_line.sum()
    dt_line_rate = dt_line / total_no_service_shots


def build_opening_phase_features(context, filtered_shots):
    pass


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
