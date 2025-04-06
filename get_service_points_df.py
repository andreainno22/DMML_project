import pandas as pd

""" It returns a dataframe with the points played by the player at service. """


def get_service_points_df(player, df):
    player_df = df[df['match_id'].str.contains(player)]

    # filtro per estrarre i punti in cui sinner gioca da 1st player
    df_1st_player = player_df[player_df['match_id'].str.contains(player + '-')]
    # filtro per estrarre i punti in cui sinner è al servizio da 1st player
    player_at_service_as_1st = df_1st_player[df_1st_player['Svr'] == 1]

    # filtro per estrarre i punti in cui sinner gioca da 2nd player
    df_2nd_player = player_df[player_df['match_id'].str.contains('-' + player)]
    # filtro per estrarre i punti in cui sinner è al servizio da 2nd player
    player_at_service_as_2nd = df_2nd_player[df_2nd_player['Svr'] == 2]

    # unisci i due dataframe
    return pd.concat([player_at_service_as_1st, player_at_service_as_2nd])
