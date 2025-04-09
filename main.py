import pandas as pd
from attr.validators import min_len
from pymining import seqmining
import re
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from win32gui import UpdateLayeredWindow

from get_df_by_player import get_response_points_df, get_service_points_df
from get_shots import get_shots_in_1st_serve_points, get_shots_in_2nd_serve_points
from get_shots_wo_opponent_shots import get_shots_by_server, get_shots_by_receiver


# Verifica che gli elementi di seq siano in x nello stesso ordine
def is_subsequence(seq, x):
    it = iter(x)
    return all(item in it for item in seq)


# todo: per trovare sequenze di colpi contigue, potrei semplicemente filtrare per punti di lunghezza n e poi impostare min length a n/2
# todo: provare con punti di lunghezza diversa, con i punti piu lunghi abbassare il supporto
def main():
    """
    Main function to execute the script.
    """

    # vorrei applicare freq_seq_enum a un dataframe, dove gli itemset sono i colpi, codificati come nella funzione decode_point, e una sequenza è un punto intero.
    # Per ogni punto, decodifico i colpi e creo una sequenza di colpi
    df = pd.read_csv('charting-m-points-2020s.csv', dtype={'2nd': str}, low_memory=False)

    player = "Jannik_Sinner"
    # estraggo il dataframe con i punti in cui sinner è al servizio
    player_on_service = get_service_points_df(player, df)
    # estraggo il dataframe con i punti in cui sinner è in risposta
    player_on_response = get_response_points_df(player, df)

    # indicizza da 0 a len(player_on_service)
    # player_on_service = player_on_service.reset_index(drop=True)

    # indicizza da 0 a len(player_on_response)
    # player_on_response = player_on_response.reset_index(drop=True)

    print("Shots by " + player + " in points on service with the 1st in:\n",
          get_shots_by_server(get_shots_in_1st_serve_points(player_on_service)))
    sinner_shots_2nd_service = get_shots_by_server(get_shots_in_2nd_serve_points(player_on_service))
    print("Shots by " + player + " in points on service with the 2nd in:\n", sinner_shots_2nd_service)

    # per i punti in risposta se droppo i punti con len < 3 mi rimangono comunque punti con lunghezza 1 (la risposta)
    sinner_shots_1st_serve_response = get_shots_by_receiver(get_shots_in_1st_serve_points(player_on_response))
    print("Shots by " + player + " in points on response with the 1st in:\n",
          sinner_shots_1st_serve_response)
    print("Shots by " + player + " in points on response with the 2nd in:\n",
          get_shots_by_receiver(get_shots_in_2nd_serve_points(player_on_response)))

    min_support = 100  # Supporto minimo per le sequenze frequenti
    min_length = 3  # Lunghezza minima della sequenza

    # Trova tutte le sequenze frequenti
    freq_shot_seqs_1st_serve_response = seqmining.freq_seq_enum(sinner_shots_1st_serve_response.shots, min_support=min_support)

    # Calcola la percentuale di vittoria per ogni sequenza frequente
    results = []

    sinner_shots_1st_serve_response['shots'] = sinner_shots_1st_serve_response['shots'].apply(tuple)

    for seq, support in freq_shot_seqs_1st_serve_response:
        # Filtra i punti in cui la sequenza appare
        matches = sinner_shots_1st_serve_response[
            sinner_shots_1st_serve_response['shots'].apply(lambda x: x if is_subsequence(seq, x) else None).notnull()
        ]

        # Calcola la percentuale di vittoria
        win_percentage = matches['won_by_player'].mean() * 100  # Media dei valori booleani (True = 1, False = 0)

        # Aggiungi la sequenza, il supporto e la percentuale di vittoria ai risultati
        results.append((seq, support, win_percentage))

    # Filtra per lunghezza minima
    results = [(seq, support, win_percentage) for seq, support, win_percentage in results if len(seq) >= min_length]
    # Stampa i risultati
    for seq, support, win_percentage in results:
        print(f"Sequenza: {seq}, Supporto: {support}, Percentuale di vittoria: {win_percentage:.2f}%")


if __name__ == "__main__":
    main()
