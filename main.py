import pandas as pd
from attr.validators import min_len
from pymining import seqmining
import re
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from win32gui import UpdateLayeredWindow

from get_df_by_player import get_response_points_df, get_service_points_df
from get_freq_shots_seqs import get_freq_shots_seqs
from get_shots import get_shots_in_1st_serve_points, get_shots_in_2nd_serve_points
from get_shots_wo_opponent_shots import get_shots_by_server, get_shots_by_receiver


# todo: sistemare la colonna outcome, ora non tiene conto del fatto che il punto è vinto da player 1 o player 2
def main():
    """
    Main function to execute the script.
    """
    SHOT_LENGTH = 3
    surface = "hard"

    # vorrei applicare freq_seq_enum a un dataframe, dove gli itemset sono i colpi, codificati come nella funzione decode_point, e una sequenza è un punto intero.
    # Per ogni punto, decodifico i colpi e creo una sequenza di colpi
    df = pd.read_csv('charting-m-points-2020s.csv', low_memory=False)

    players = ("Jannik_Sinner", "Alexander_Zverev", "Carlos_Alcaraz")

    for player in players:
        # estraggo il dataframe con i punti in cui sinner è al servizio
        player_on_service = get_service_points_df(player, df, surface)
        # estraggo il dataframe con i punti in cui sinner è in risposta
        player_on_response = get_response_points_df(player, df, surface)

        player_shots_1st_service = get_shots_by_server(get_shots_in_1st_serve_points(player_on_service))
        player_shots_1st_service = player_shots_1st_service[player_shots_1st_service['shots'].apply(len) == SHOT_LENGTH]

        player_shots_2nd_service = get_shots_by_server(get_shots_in_2nd_serve_points(player_on_service))
        player_shots_2nd_service = player_shots_2nd_service[player_shots_2nd_service['shots'].apply(len) == SHOT_LENGTH]

        player_shots_1st_serve_response = get_shots_by_receiver(get_shots_in_1st_serve_points(player_on_response))
        player_shots_1st_serve_response = player_shots_1st_serve_response[
            player_shots_1st_serve_response['shots'].apply(len) == SHOT_LENGTH]

        player_shots_2nd_serve_response = get_shots_by_receiver(get_shots_in_2nd_serve_points(player_on_response))
        player_shots_2nd_serve_response = player_shots_2nd_serve_response[
            player_shots_2nd_serve_response['shots'].apply(len) == SHOT_LENGTH]

        list_of_shots = {player + " on serve with the 1st": player_shots_1st_service,
                         player + " on serve with the 2nd": player_shots_2nd_service,
                         player + " on response with the 1st": player_shots_1st_serve_response,
                         player + " on response with the 2nd": player_shots_2nd_serve_response}

        for point_name, shots in list_of_shots.items():
            # stampa la descrizione delle sequenze
            print("\n" + point_name + ":\n")
            # stampa i risultati
            for seq, support, win_percentage, most_frequent_outcome in get_freq_shots_seqs(shots):
                print(f"Sequence: {seq}, Support: {support}, Win percentage: {win_percentage:.2f}%, Most frequent outcome: {most_frequent_outcome}")


if __name__ == "__main__":
    main()
