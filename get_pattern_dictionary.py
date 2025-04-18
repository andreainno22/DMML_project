from costants import PLAYERS, SURFACE, SHOT_LENGTH
from get_df_by_player import get_service_points_df, get_response_points_df
from get_freq_shots_seqs import get_freq_shots_seqs
from get_shots import get_shots_in_2nd_serve_points, get_shots_in_1st_serve_points
from get_shots_wo_opponent_shots import get_shots_by_receiver, get_shots_by_server


# todo: fai il dizionario di pattern globali frequenti
def get_pattern_dictionary(df):
    all_pattern_lists = []
    for player in PLAYERS:
        # estraggo il dataframe con i punti in cui sinner è al servizio
        player_on_service = get_service_points_df(player, df, SURFACE)
        # estraggo il dataframe con i punti in cui sinner è in risposta
        player_on_response = get_response_points_df(player, df, SURFACE)

        # estraggo i punti in cui il giocatore è al servizio e ne ricavo i primi SHOT_LENGTH colpi
        player_shots_1st_service = get_shots_by_server(get_shots_in_1st_serve_points(player_on_service))

        # prima di estrarre i primi SHOT_LENGTH colpi, verifico che il punto sia lungo almeno SHOT_LENGTH colpi
        player_shots_1st_service = player_shots_1st_service[player_shots_1st_service['shots'].apply(len) >= SHOT_LENGTH]
        player_shots_1st_service["shots"] = player_shots_1st_service['shots'].apply(lambda x: x[:SHOT_LENGTH])

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
            pattern_list_in_context = []
            # stampa la descrizione delle sequenze
            print("\n" + point_name + ":\n")
            # stampa i risultati
            for seq, support, win_percentage, most_frequent_outcome in get_freq_shots_seqs(shots):
                pattern_list_in_context.append((point_name, seq, support, win_percentage, most_frequent_outcome))
                print(
                    f"Sequence: {seq}, Support: {support}, Win percentage: {win_percentage:.2f}%, Most frequent outcome: {most_frequent_outcome}")
            all_pattern_lists.append({point_name: pattern_list_in_context})
