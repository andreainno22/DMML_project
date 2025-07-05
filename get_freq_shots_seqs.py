import pandas as pd
from pymining import seqmining
from costants import SURFACES
from get_df_by_player import get_service_points_df, get_response_points_df
from utils import filter_and_trim_shots
from get_shots import get_shots_in_1st_serve_points, get_shots_in_2nd_serve_points
from get_shots_wo_opponent_shots import get_shots_by_server, get_shots_by_receiver


# Verifica che gli elementi di seq siano in x nello stesso ordine
def is_subsequence(seq, x):
    it = iter(x)
    return all(item in it for item in seq)


def get_freq_shots_seqs(shots):
    """
    Count the frequency of each shot sequence in the list of shots.
    """

    min_support = len(shots["shots"]) * 0.008  # Supporto minimo per le sequenze frequenti
    min_length = 3  # Lunghezza minima della sequenza

    # Trova tutte le sequenze frequenti
    freq_shots_seqs = seqmining.freq_seq_enum(shots.shots, min_support=min_support)

    # Calcola la percentuale di vittoria per ogni sequenza frequente
    results = []

    shots['shots'] = shots['shots'].apply(tuple)

    for seq, support in freq_shots_seqs:
        # Filtra i punti in cui la sequenza appare
        matches = shots[shots['shots'].apply(lambda x: x if is_subsequence(seq, x) else None).notnull()]

        # Calcola la percentuale di vittoria
        win_percentage = matches['won_by_player'].mean() * 100  # Media dei valori booleani (True = 1, False = 0)
        most_frequent_outcome = matches['outcome'].mode()[0] if not matches['outcome'].empty else None

        # Aggiungi la sequenza, il supporto e la percentuale di vittoria ai risultati
        results.append((seq, support, win_percentage, most_frequent_outcome))

    # Filtra per lunghezza minima
    results = [(seq, support, win_percentage, most_frequent_outcome) for
               seq, support, win_percentage, most_frequent_outcome in results if len(seq) >= min_length]

    # Sort by support
    results.sort(key=lambda x: x[1], reverse=True)

    return results


def frequent_shots_py_player(player, df, surface, context):

    if "on serve" in context:
        player_on_service = get_service_points_df(player, df, surface)
        # estraggo i punti in cui il giocatore è al servizio
        shots_1st_service = get_shots_by_server(get_shots_in_1st_serve_points(player_on_service))
        shots_2nd_service = get_shots_by_server(get_shots_in_2nd_serve_points(player_on_service))
        shots = pd.concat([shots_1st_service.copy(),
                           shots_2nd_service.copy()])

    else:
        # estraggo il dataframe con i punti in cui sinner è in risposta
        player_on_response = get_response_points_df(player, df, surface)
        # estraggo i punti in cui il giocatore è in risposta
        shots_1st_response = get_shots_by_receiver(get_shots_in_1st_serve_points(player_on_response))
        shots_2nd_response = get_shots_by_receiver(get_shots_in_2nd_serve_points(player_on_response))
        shots = pd.concat([shots_1st_response.copy(), shots_2nd_response.copy()])

    # Only shots with length >= 3
    filtered_shots = filter_and_trim_shots(shots.copy())
    for result in get_freq_shots_seqs(filtered_shots):
        yield result

