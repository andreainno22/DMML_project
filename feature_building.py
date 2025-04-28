import numpy as np
import pandas as pd
from collections import Counter

# todo: modificare extract_aggregated_features, o eliminare o aggiungere solo bag of patterns
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
        total_no_service_shots = shots['shots'].apply(lambda seq: len(seq) - 1).sum()
        # conta gli ace
        ace_num = shots[(shots['shots'].apply(len) == 1) & (shots['outcome'].str.contains(r'\*', regex=True))].copy().shape[0]
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

        # Applica la funzione per creare una nuova colonna con i valori di profondità
        # .loc per assegnare alla copia 'points_with_returns'
        points_with_returns.loc[:, 'return_depth_value'] = points_with_returns['shots'].apply(
            lambda seq: next(
                # Cerca il valore corrispondente al primo codice di profondità trovato...
                (shot_depth_map[code] for code in shot_depth_map if code in seq[0]),
                # ...restituisci None se nessun codice viene trovato nel primo colpo
                None
            )
            # Esegui quanto sopra solo se la sequenza non è vuota, altrimenti restituisci None
            if len(seq) > 0 else None
        )

        # 5. Calcola la media dei valori di profondità validi
        # .mean() ignora automaticamente i valori NaN/None
        avg_resp_depth = points_with_returns['return_depth_value'].mean()

    avg_shot_length = shots['shots'].apply(len).mean()

    # numero di punti con discesa a rete
    total_volley_points = shots[
        shots['shots'].apply(lambda sequence:
                             any(code in shot and '=' not in shot for shot in sequence for code in volley_codes) or
                             any(code in shot and '-' in shot for shot in sequence for code in not_volley_codes)
                             )
    ].shape[0]
    net_points_rate = total_volley_points / total_points_started

    # numero di punti vinti dopo essere stato a rete almeno una vota durante lo scambio
    net_points_won = shots[(shots['won_by_player'] is True) &
                           (shots['shots'].apply(lambda sequence:
                                                 any(code in shot and '=' not in shot for shot in sequence for code in
                                                     volley_codes) or
                                                 any(code in shot and '-' in shot for shot in sequence for code in
                                                     not_volley_codes)
                                                 ))].shape[0]
    if total_volley_points != 0:
        net_points_won_rate = net_points_won / total_volley_points
    else:
        net_points_won_rate = 0

    # escludo dai total points quelli per cui non c'è stata risposta
    winners = shots[(shots['won_by_player'] is True) & (shots['outcome'].str.contains(r'\*', regex=True)) & (len(shots['shots']) > 1)].shape[0]
    winners_rate = winners / total_points_with_return

    # escludo dai total points quelli per cui non c'è stata risposta
    unforced_errors_rate = \
        shots[(shots['shots'].apply(len) > 1) & (shots['outcome'].str.contains(r'\#', regex=True)) & (shots['won_by_player'] is False)].shape[
            0] / total_points_with_return

    # percentuale di slice su colpi totali
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

    generic_features = {'average_shot_length': avg_shot_length, 'net_points_rate': net_points_rate,
            'net_points_won_rate': net_points_won_rate, 'winners_rate': winners_rate,
            'unforced_errors_rate': unforced_errors_rate, 'slices_rate': slices_rate,
            'dropshots_rate': dropshots_rate, 'crosscourt_rate': crosscourt_rate,
            'middlecourt_rate': middle_rate, 'down_the_line_rate': dt_line_rate}

    if str.__contains__(context, "on serve"):
        return generic_features | {'ace_rate': ace_rate}
    else:
        return generic_features | {'average_response_depth': avg_resp_depth}


def build_opening_phase_features(context, filtered_shots):
    """
    Estrae feature sulle categorie di colpi nei primi 3 colpi della sequenza.
    Macro-categorie:
    - forehand_ground
    - backhand_ground
    - slice_shot
    - net_shot
    - other_shot
    """

    from get_mapping_dictionaries import get_mapping_dictionaries
    shot_types = get_mapping_dictionaries("shot_types")

    # Mapping dei codici in macro-categorie
    def map_to_macro_category(shot_code):
        if shot_code in ['f']:  # forehand groundstroke
            return 'forehand_ground'
        elif shot_code in ['b']:  # backhand groundstroke
            return 'backhand_ground'
        elif shot_code in ['r', 's']:  # slices
            return 'slice_shot'
        elif shot_code in ['v', 'z', 'o', 'p', 'h', 'i', 'j']:  # net play shots
            return 'net_shot'
        elif shot_code in ['u', 'y']:
            return 'dropshot'

    # Contatori per ogni posizione
    counts = {
        '1st': Counter(),
        '2nd': Counter(),
        '3rd': Counter()
    }

    total_points = len(filtered_shots)

    for shots_seq in filtered_shots['shots']:
        for idx, shot in enumerate(shots_seq):  # filtered shots sono già i primi 3 colpi del punto
            shot_type = shot[0]  # solo il primo carattere definisce il tipo di colpo
            macro_category = map_to_macro_category(shot_type)

            if idx == 0:
                counts['1st'][macro_category] += 1
            elif idx == 1:
                counts['2nd'][macro_category] += 1
            elif idx == 2:
                counts['3rd'][macro_category] += 1

    # Ora costruisco le feature normalizzate
    feature_dict = {}
    for position in ['1st', '2nd', '3rd']:
        for category in ['forehand_ground', 'backhand_ground', 'slice_shot', 'net_shot', 'other_shot']:
            key = f"{context}_opening_{position}_{category}"
            feature_dict[key] = counts[position][category] / total_points

    return feature_dict

# todo: aggiustare
def build_final_dataset(generic_features, opening_phase_features):
    """
    Costruisce il dataset finale combinando:
    - le feature generali (generic_features)
    - le feature sull'opening phase (opening_phase_features)
    - (opzionale: in futuro) i bag of patterns
    """

    final_records = []

    for context, shots in list_of_shots.items():
        # Combina tutto
        record = {
            'context': context.split(",")[0],  # es. "Sinner on serve with the 1st"
            'surface': context.split(",")[1].replace('on ', '').strip()  # es. "hard"
        }
        record.update(generic_features)
        record.update(opening_phase_features)

        final_records.append(record)

    # Costruisci il DataFrame finale
    final_df = pd.DataFrame(final_records)

    return final_df

