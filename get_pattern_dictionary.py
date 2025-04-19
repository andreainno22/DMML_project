import pandas as pd

from costants import PLAYERS, SURFACE, SHOT_LENGTH, TOP_PATTERNS
from feature_building import extract_aggregated_features, build_final_dataset
from get_df_by_player import get_service_points_df, get_response_points_df
from get_freq_shots_seqs import get_freq_shots_seqs
from get_shots import get_shots_in_2nd_serve_points, get_shots_in_1st_serve_points
from get_shots_wo_opponent_shots import get_shots_by_receiver, get_shots_by_server
from collections import Counter

# Funzione per filtrare e tagliare i colpi a SHOT_LENGTH
def filter_and_trim_shots(shots_df):
    shots_df = shots_df[shots_df['shots'].apply(len) >= SHOT_LENGTH]
    shots_df["shots"] = shots_df['shots'].apply(lambda x: x[:SHOT_LENGTH])
    return shots_df


# todo: ora come ora vengono considerati solo i primi 3 colpi dei punti come patterns
def get_pattern_dictionary(df):
    all_pattern_lists = []
    for player in PLAYERS:
        # estraggo il dataframe con i punti in cui sinner è al servizio
        player_on_service = get_service_points_df(player, df, SURFACE)
        # estraggo il dataframe con i punti in cui sinner è in risposta
        player_on_response = get_response_points_df(player, df, SURFACE)

        # estraggo i punti in cui il giocatore è al servizio e ne ricavo i primi SHOT_LENGTH colpi
        player_shots_1st_service = filter_and_trim_shots(
            get_shots_by_server(get_shots_in_1st_serve_points(player_on_service)))
        player_shots_2nd_service = filter_and_trim_shots(
            get_shots_by_server(get_shots_in_2nd_serve_points(player_on_service)))

        # estraggo i punti in cui il giocatore è in risposta e ne ricavo i primi SHOT_LENGTH colpi
        player_shots_1st_serve_response = filter_and_trim_shots(
            get_shots_by_receiver(get_shots_in_1st_serve_points(player_on_response)))
        player_shots_2nd_serve_response = filter_and_trim_shots(
            get_shots_by_receiver(get_shots_in_2nd_serve_points(player_on_response)))

        list_of_shots = {player + " on serve with the 1st": player_shots_1st_service,
                         player + " on serve with the 2nd": player_shots_2nd_service,
                         player + " on response with the 1st": player_shots_1st_serve_response,
                         player + " on response with the 2nd": player_shots_2nd_serve_response}

        # itera per ogni context, ovvero ogni punto di vista (serve o response) e per ogni colpo (1st o 2nd)
        for point_name, shots in list_of_shots.items():
            pattern_list_in_context = []
            # stampa la descrizione delle sequenze
            print("\n" + point_name + ":\n")
            # stampa i risultati
            for seq, support, win_percentage, most_frequent_outcome in get_freq_shots_seqs(shots):
                pattern_list_in_context.append((seq, support, win_percentage, most_frequent_outcome))
                print(
                    f"Sequence: {seq}, Support: {support}, Win percentage: {win_percentage:.2f}%, Most frequent outcome: {most_frequent_outcome}")
                extract_aggregated_features(pattern_list_in_context)
            all_pattern_lists.append({point_name: pattern_list_in_context})
    return build_pattern_vocabulary(all_pattern_lists)


def build_pattern_vocabulary(all_pattern_lists):
    """
    all_pattern_lists: lista di dizionari con contesti come chiavi e liste di pattern come valori
    """
    global_counter = Counter()
    context_patterns_count = {}

    # Itera su ogni lista di pattern per contesto
    for pattern_list in all_pattern_lists:
        for context, patterns in pattern_list.items():
            # Inizializza un contatore per il contesto se non esiste
            if context not in context_patterns_count:
                context_patterns_count[context] = Counter()

            # Aggiorna i contatori
            for seq, support, _, _ in patterns:
                global_counter[seq] += 1
                context_patterns_count[context][seq] = support

    # Estrai i pattern più frequenti
    top_patterns = [seq for seq, _ in global_counter.most_common(TOP_PATTERNS)]

    # Conta i pattern per ogni contesto basandoti sui top_patterns
    pattern_counts_by_context = {
        context: [context_patterns_count[context].get(p, 0) for p in top_patterns]
        for context in context_patterns_count
    }

    return build_final_dataset(top_patterns, pattern_counts_by_context, all_pattern_lists)
