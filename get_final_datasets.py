import pandas as pd

from costants import PLAYERS, SHOT_LENGTH, TOP_PATTERNS, MIN_PLAYERS, SURFACES, PLAYER_SURFACES_DICT
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
    list_of_shots = {}
    for player in PLAYERS:
        for surface in PLAYER_SURFACES_DICT[player]:
            # estraggo il dataframe con i punti in cui sinner è al servizio
            player_on_service = get_service_points_df(player, df, surface)

            # estraggo il dataframe con i punti in cui sinner è in risposta
            player_on_response = get_response_points_df(player, df, surface)

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

            # crea un dizionario contenente ogni constesto, dove ogni contesto è composto da nome giocatore, tipo di punto e superficie
            list_of_shots = list_of_shots | {
                player + " on serve with the 1st, on " + surface: player_shots_1st_service.copy(),
                player + " on serve with the 2nd, on " + surface: player_shots_2nd_service.copy(),
                player + " on response with the 1st, on " + surface: player_shots_1st_serve_response.copy(),
                player + " on response with the 2nd, on " + surface: player_shots_2nd_serve_response.copy()}

        # itera per ogni context, ovvero ogni punto di vista (serve o response) e per ogni colpo (1st o 2nd)
        for point_name, shots in list_of_shots.items():
            pattern_list_in_context = []
            # stampa la descrizione delle sequenze
            # print("\n" + point_name + ":\n")
            # stampa i risultati
            for seq, support, win_percentage, most_frequent_outcome in get_freq_shots_seqs(shots):
                pattern_list_in_context.append((seq, support, win_percentage, most_frequent_outcome))
                # print(
                #    f"Sequence: {seq}, Support: {support}, Win percentage: {win_percentage:.2f}%, Most frequent outcome: {most_frequent_outcome}")
                extract_aggregated_features(pattern_list_in_context)
            all_pattern_lists.append({point_name: pattern_list_in_context})
    return build_pattern_vocabulary(all_pattern_lists)


from collections import Counter, defaultdict


def build_pattern_vocabulary(all_pattern_lists):
    """
    Costruisce il vocabolario globale dei pattern frequenti,
    selezionando solo quelli con buona diffusione tra i giocatori.

    Args:
        all_pattern_lists: lista di dizionari {context: [ (pattern, support, win%, outcome) ]}
        min_players: numero minimo di giocatori che devono usare un pattern per tenerlo

    Returns:
        DataFrame finale con vettori per giocatore (via build_final_dataset)
    """

    global_support_counter = Counter()
    pattern_player_set = defaultdict(set)
    context_patterns_count = {}

    # Itera su ogni lista di pattern per contesto
    for pattern_list in all_pattern_lists:
        for context, patterns in pattern_list.items():
            player = context.split()[0]  # estrai il nome del giocatore
            if context not in context_patterns_count:
                context_patterns_count[context] = Counter()

            for seq, support, _, _ in patterns:
                global_support_counter[seq] += support
                pattern_player_set[seq].add(player)
                context_patterns_count[context][seq] = support

    # Filtra i pattern usati da almeno `MIN_PLAYERS` diversi
    filtered_patterns = [
        (seq, global_support_counter[seq])
        for seq in global_support_counter
        if len(pattern_player_set[seq]) >= MIN_PLAYERS
    ]

    # Ordina per supporto e prendi i top-N
    top_patterns = [
        seq for seq, _ in sorted(filtered_patterns, key=lambda x: x[1], reverse=True)[:TOP_PATTERNS]
    ]

    # Costruisci i BoP vettori per ogni contesto
    pattern_counts_by_context = {
        context: [context_patterns_count[context].get(p, 0) for p in top_patterns]
        for context in context_patterns_count
    }

    return build_final_dataset(top_patterns, pattern_counts_by_context, all_pattern_lists)
