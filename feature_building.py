import numpy as np
import pandas as pd
from collections import Counter, defaultdict


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


total_players_with_not_null_avg_resp_depth_per_context = {}
avg_resp_depth_per_context = {}
dict_of_players_with_none_avg_resp_depth_per_context = {}


def build_generic_features(player, context, shots):
    total_points = shots.shape[0]
    volley_codes = {'v', 'z', 'o', 'p', 'h', 'i', 'j', 'k'}
    not_volley_codes = {'f', 'b', 's', 'r', 'u', 'y'}
    slice_codes = {'r', 's'}
    dropshots_codes = {'u', 'y'}
    base_context = context.split(f"{player} ", 1)[1]

    # queste 3 variabili servono per gestire il filling della variabile ave_depth_response quando non presente.
    # La logica è che viene riempita con la media del contesto. Per farlo prima bisogna tenere traccia di tutte le avg_depth_resp del
    # contesto e alla fine fare la media per il contesto. Nel frattempo è necessario mantenere in
    # dict_of_players_with_none_avg_resp_depth_per_context i nomi dei giocatori con la variabile nulla, che verrà
    # riempita in build_final_dataset
    total_players_with_not_null_avg_resp_depth_per_context.setdefault(base_context, 0)
    avg_resp_depth_per_context.setdefault(base_context, 0)
    dict_of_players_with_none_avg_resp_depth_per_context.setdefault(base_context, [])

    if str.__contains__(context, "on serve"):
        # se len <= 2 vuol dire che c'è stato solo servizio e risposta o solo servizio
        total_points_started = shots[shots['shots'].apply(len) > 2].shape[0]
        # se len = 1 vuol dire che c'è stato o un ace o un doppio fallo, senza tentativo di risposta
        total_points_with_return = shots[shots['shots'].apply(len) > 1].shape[0]
        # numero di colpi totale escludendo il servizio
        total_no_service_shots = shots['shots'].apply(lambda seq: len(seq) - 1).sum()
        # conta gli ace
        ace_num = \
            shots[(shots['shots'].apply(len) == 1) & (shots['outcome'].str.contains(r'\*', regex=True))].copy().shape[0]
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

        if not np.isnan(avg_resp_depth):
            total_players_with_not_null_avg_resp_depth_per_context[base_context] += 1
            avg_resp_depth_per_context[base_context] += avg_resp_depth
        else:
            total_players_with_not_null_avg_resp_depth_per_context[base_context] += 1
            dict_of_players_with_none_avg_resp_depth_per_context[base_context].append(player)

        # 6. Rimuovi la colonna temporanea
        points_with_returns.drop(columns=['return_depth_value'], inplace=True)
        # 7. Assegna la media alla variabile
        # if avg_resp_depth is None:
        # alcuni giocatori non hanno info su depth in alcuni contesti
        # avg_resp_depth = 0.0

    # è il numero medio di colpi fatti dal giocatore, non proprio la lunghezza dello scambio
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
    net_points_won = shots[(shots['won_by_player'] == True) &
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
    # se "on serve" escludo i punti finiti con il servizio, per evitare di contare gli ace
    winners = shots[(shots['won_by_player'] == True) & (shots['outcome'].str.contains(r'\*', regex=True)) & (
            len(shots['shots']) > 1 if "on serve" in context else len(shots['shots']) > 0)].shape[0]
    winners_rate = winners / total_points_with_return

    # escludo dai total points quelli per cui non c'è stata risposta
    unforced_errors_rate = \
        shots[(len(shots['shots']) > 1 if "on serve" in context else len(shots['shots']) > 0) & (shots['outcome'].str.contains(r'\@', regex=True)) & (
                shots['won_by_player'] == False)].shape[
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
    Estrae feature sulle categorie di colpi nei primi SHOT_LENGTH (tipicamente 3) colpi della sequenza.
    Se il contesto è 'on serve', il 1° colpo viene mappato usando serve_direction.
    Altrimenti (contesto 'on response'), anche il 1° colpo usa la mappatura standard.
    Il 2° e 3° colpo usano sempre la mappatura standard.

    Macro-categorie per punti IN RISPOSTA (e per 2°/3° colpo al servizio):
    - forehand_ground
    - backhand_ground
    - slice_shot
    - net_shot -> TOLTA, INUTILE NEI PRIMI 3 COLPI
    - dropshot -> TOLTA, INUTILE NEI PRIMI 3 COLPI

    Macro-categorie aggiuntive SOLO per il 1° COLPO AL SERVIZIO:
    - wide_serve
    - body_serve
    - down_the_T_serve
    (Derivate da `serve_direction`)
    """
    # Assicurati che questa funzione sia importabile correttamente
    from get_mapping_dictionaries import get_mapping_dictionaries
    shot_types = get_mapping_dictionaries("shot_types")  # Usato per riferimento sotto
    serve_directions = get_mapping_dictionaries("serve_directions")  # Dizionario {codice: nome_categoria_servizio}
    UNKNOWN_SERVE_CATEGORY = 'unknown direction'  # Assicurati che questo sia il valore esatto nel dizionario

    # Funzione di mappatura per colpi di scambio (non servizio)
    def map_rally_shot(shot_code):
        # Nota: shot_types non viene usato direttamente qui, si usano nomi categoria fissi
        # Se shot_types contenesse {'f': 'forehand_ground', ...} potresti usarlo
        if shot_code == 'f':  # forehand groundstroke
            return 'forehand_ground'
        elif shot_code == 'b':  # backhand groundstroke
            return 'backhand_ground'
        elif shot_code in ['r', 's']:  # slices
            return 'slice_shot'
        return None  # Codice non riconosciuto come colpo di scambio

    # Funzione di mappatura specifica per il servizio, IGNORA UNKNOWN_SERVE_CATEGORY
    def map_serve_direction_shot(shot_code):
        category = serve_directions.get(shot_code)
        # Se la categoria è quella da ignorare, trattala come non riconosciuta (None)
        if category == UNKNOWN_SERVE_CATEGORY:
            return None
        return category

    # Contatori per ogni posizione (1st, 2nd, 3rd)
    counts = defaultdict(Counter)  # Usa defaultdict per semplicità

    total_points = len(filtered_shots)
    if total_points == 0:
        print(f"WARNING: filtered_shots is empty for context: {context}")

    is_serve_context = "on serve" in context

    # Itera sulle sequenze filtrate (già tagliate a SHOT_LENGTH)
    for shots_seq in filtered_shots['shots']:
        for idx, shot in enumerate(shots_seq):
            if not shot:
                continue  # Salta eventuali stringhe vuote

            position_label = f"{idx + 1}{'st' if idx == 0 else 'nd' if idx == 1 else 'rd'}"  # 1st, 2nd, 3rd
            shot_code = shot[0]  # Prendi solo il primo carattere come codice tipo
            macro_category = None

            # Applica la logica di mappatura condizionale
            if idx == 0 and is_serve_context:
                # Primo colpo nel contesto di servizio -> usa mappa servizio
                macro_category = map_serve_direction_shot(shot_code)
            else:
                # Secondo/terzo colpo, oppure qualsiasi colpo in contesto di risposta -> usa mappa scambio
                macro_category = map_rally_shot(shot_code)

            # Incrementa il contatore solo se la categoria è valida
            if macro_category:
                counts[position_label][macro_category] += 1

    feature_dict = {}
    rally_categories = ['forehand_ground', 'backhand_ground', 'slice_shot']

    if is_serve_context:
        # Contesto Servizio: Calcola feature servizio (1st) e scambio (2nd, 3rd)
        for serve_code, category_name in serve_directions.items():
            if category_name == UNKNOWN_SERVE_CATEGORY:
                continue
            key = f"opening_1st_{category_name.lower().replace(' ', '_')}"
            feature_dict[key] = counts['1st'][category_name] / total_points

        for position in ['2nd', '3rd']:
            for category in rally_categories:
                key = f"opening_{position}_{category}"
                feature_dict[key] = counts[position][category] / total_points
    else:
        # Contesto Risposta: Calcola solo feature scambio (1st, 2nd, 3rd)
        for position in ['1st', '2nd', '3rd']:
            for category in rally_categories:
                key = f"opening_{position}_{category}"
                feature_dict[key] = counts[position][category] / total_points

            # --- Assicura Chiavi Mancanti (Contesto-dipendente) ---
    all_possible_keys = set()
    if is_serve_context:
        # Contesto Servizio: Chiavi servizio (1st) e scambio (2nd, 3rd)
        for serve_cat in serve_directions.values():
            if serve_cat == UNKNOWN_SERVE_CATEGORY: continue
            all_possible_keys.add(f"opening_1st_{serve_cat.lower().replace(' ', '_')}")
        for position in ['2nd', '3rd']:
            for rally_cat in rally_categories:
                all_possible_keys.add(f"opening_{position}_{rally_cat}")
    else:
        # Contesto Risposta: Solo chiavi scambio (1st, 2nd, 3rd)
        for position in ['1st', '2nd', '3rd']:
            for rally_cat in rally_categories:
                all_possible_keys.add(f"opening_{position}_{rally_cat}")

    # Completa il dizionario con 0.0 per le chiavi mancanti definite per questo contesto
    final_feature_dict = {key: feature_dict.get(key, 0.0) for key in all_possible_keys}
    return final_feature_dict
    # --- Fine Assicura Chiavi Mancanti ---


# todo: da aggiungere i bag of patterns (se utili)
def build_final_dataset(all_features_by_context):
    """
    Costruisce un dizionario di DataFrame, uno per ogni contesto base,
    contenente le features combinate (generiche + opening phase) per ogni giocatore.

    Args:
        all_features_by_context: Dizionario {base_context: [lista di dizionari feature per player]}

    Returns:
        dict: Dizionario {base_context: pd.DataFrame}, dove ogni DataFrame
              contiene le features combinate per quel contesto, con una riga per giocatore.
    """
    final_datasets = {}
    for base_context, player_features_list in all_features_by_context.items():
        if player_features_list:  # Verifica se ci sono dati per questo contesto
            # Crea il DataFrame dalla lista di dizionari
            context_df = pd.DataFrame(player_features_list)

            # Riordina le colonne mettendo 'player' per prima
            if 'player' in context_df.columns:
                cols = ['player'] + [col for col in context_df.columns if col != 'player']
                context_df = context_df[cols]

            # Imposta 'player' come indice se preferito
            context_df = context_df.set_index('player')

            # riempie il dato mancante di avg_depth_response con la media del contesto
            if "on response" in base_context:
                for player in dict_of_players_with_none_avg_resp_depth_per_context[base_context]:
                    if np.isnan(context_df.loc[player, 'average_response_depth']):
                        context_df.loc[player, 'average_response_depth'] = avg_resp_depth_per_context[base_context] / \
                                                                           total_players_with_not_null_avg_resp_depth_per_context[
                                                                               base_context]
            final_datasets[base_context] = context_df
        else:
            # Se non ci sono features per un contesto, crea un DataFrame vuoto
            print(f"Attenzione: Nessuna feature trovata per il contesto '{base_context}'")
            final_datasets[base_context] = pd.DataFrame()  # Ritorna un DF vuoto
    counts = []

    # conta per contesto il numero di giocatori che non hanno info sulla profondità della risposta
    for context, df in final_datasets.items():
        if "on response" in context:
            counts.append(df['average_response_depth'].isna().sum())
    print(f"Numero totale di NaN in average_response_depth: {counts}")
    return final_datasets
