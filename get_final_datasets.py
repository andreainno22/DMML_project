import pandas as pd

from costants import PLAYERS, SHOT_LENGTH, TOP_PATTERNS, MIN_PLAYERS, SURFACES, PLAYER_SURFACES_DICT
from feature_building import extract_aggregated_features, build_final_dataset, build_generic_features, \
    build_opening_phase_features
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

            # estraggo i punti in cui il giocatore è al servizio
            player_shots_1st_service = get_shots_by_server(get_shots_in_1st_serve_points(player_on_service))
            player_shots_2nd_service = get_shots_by_server(get_shots_in_2nd_serve_points(player_on_service))

            # estraggo i punti in cui il giocatore è in risposta
            player_shots_1st_serve_response = get_shots_by_receiver(get_shots_in_1st_serve_points(player_on_response))
            player_shots_2nd_serve_response = get_shots_by_receiver(get_shots_in_2nd_serve_points(player_on_response))

            # crea un dizionario contenente ogni constesto, dove ogni contesto è composto da nome giocatore, tipo di punto e superficie
            list_of_shots = list_of_shots | {
                player + " on serve with the 1st, on " + surface: player_shots_1st_service.copy(),
                player + " on serve with the 2nd, on " + surface: player_shots_2nd_service.copy(),
                player + " on response with the 1st, on " + surface: player_shots_1st_serve_response.copy(),
                player + " on response with the 2nd, on " + surface: player_shots_2nd_serve_response.copy()}

        # itera per ogni context, ovvero ogni punto di vista (serve o response) e per ogni colpo (1st o 2nd)
        for context, shots in list_of_shots.items():
            pattern_list_in_context = []
            # filtro i colpi per >= di SHOT_LENGTH e taglio ai primi SHOT_LENGTH
            filtered_shots = filter_and_trim_shots(shots)
            for seq, support, win_percentage, most_frequent_outcome in get_freq_shots_seqs(filtered_shots):
                #itera su tutti i pattern trovati e li aggiunge al dataset finale, con le features generali
                pattern_list_in_context.append((seq, support, win_percentage, most_frequent_outcome))
                # print(
                #    f"Sequence: {seq}, Support: {support}, Win percentage: {win_percentage:.2f}%, Most frequent outcome: {most_frequent_outcome}")
            all_pattern_lists.append({context: pattern_list_in_context})
            # costruisce le features generiche a partire da tutti i punti con lunghezza
            generic_features = build_generic_features(context, shots)
            opening_phase_features = build_opening_phase_features(context, filtered_shots)
    return build_pattern_vocabulary(all_pattern_lists)


from collections import Counter, defaultdict


def build_pattern_vocabulary(all_pattern_lists):
    """
    Costruisce un vocabolario di top pattern SEPARATO per ogni contesto,
    filtrando quelli usati da almeno MIN_PLAYERS.

    Returns:
        top_patterns_per_context: dizionario {context: [top pattern list]}
        pattern_counts_by_context: dizionario {context: [count vettore]}
        build_final_dataset(...) → dataset finale con BoP per ciascun contesto
    """

    from collections import defaultdict, Counter

    context_pattern_counter = {}  # {context: Counter(pattern: support)}
    pattern_player_set_per_context = defaultdict(lambda: defaultdict(set))  # {context: {pattern: {players}}}
    context_patterns = {}

    # 1. Costruzione contatori per ogni contesto
    for pattern_list in all_pattern_lists:
        for context, patterns in pattern_list.items():
            player = context.split()[0]

            if context not in context_pattern_counter:
                context_pattern_counter[context] = Counter()

            for seq, support, _, _ in patterns:
                context_pattern_counter[context][seq] += support
                pattern_player_set_per_context[context][seq].add(player)

    # 2. Selezione dei top pattern per ciascun contesto
    top_patterns_per_context = {}
    pattern_counts_by_context = {}

    for context, counter in context_pattern_counter.items():
        filtered_patterns = [
            (seq, support)
            for seq, support in counter.items()
            if len(pattern_player_set_per_context[context][seq]) >= MIN_PLAYERS
        ]

        # Ordina e prendi i TOP_PATTERNS più supportati
        top_patterns = [
            seq for seq, _ in sorted(filtered_patterns, key=lambda x: x[1], reverse=True)[:TOP_PATTERNS]
        ]

        # Costruisci il vettore di conteggi per questo contesto
        pattern_counts = [counter.get(p, 0) for p in top_patterns]

        # Salva
        top_patterns_per_context[context] = top_patterns
        pattern_counts_by_context[context] = pattern_counts

    # 3. Costruzione dataset finale
    return build_final_dataset(top_patterns_per_context, pattern_counts_by_context, all_pattern_lists)

