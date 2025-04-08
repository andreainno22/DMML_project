import pandas as pd

""" It returns a dataframe with the points played by the player at service. """


def get_service_points_df(player, df):
    player_df = df[df['match_id'].str.contains(player)]

    # filtro per estrarre i punti in cui sinner gioca da 1st player
    df_1st_player = player_df[player_df['match_id'].str.contains(player + '-')]
    # filtro per estrarre i punti in cui sinner è al servizio da 1st player
    player_at_service_as_1st = df_1st_player[df_1st_player['Svr'] == 1]

    player_at_service_as_1st['PtWinner'] = player_at_service_as_1st['PtWinner'].apply(lambda x: True if x == 1 else False)


    # filtro per estrarre i punti in cui sinner gioca da 2nd player
    df_2nd_player = player_df[player_df['match_id'].str.contains('-' + player)]
    # filtro per estrarre i punti in cui sinner è al servizio da 2nd player
    player_at_service_as_2nd = df_2nd_player[df_2nd_player['Svr'] == 2]

    # modifica il campo 'PtWinner' con True se p2 ha vinto False altrimenti
    player_at_service_as_2nd['PtWinner'] = player_at_service_as_2nd['PtWinner'].apply(lambda x: True if x == 2 else False)

    # unisci i due dataframe
    return pd.concat([player_at_service_as_1st, player_at_service_as_2nd])


def get_response_points_df(player, df):
    player_df = df[df['match_id'].str.contains(player)]

    # filtro per estrarre i punti in cui player gioca da 1st player
    df_1st_player = player_df[player_df['match_id'].str.contains(player + '-')]
    # filtro per estrarre i punti in cui player è in risposta da 1st player
    player_on_response_as_1st = df_1st_player[df_1st_player['Svr'] == 2]

    player_on_response_as_1st['PtWinner'] = player_on_response_as_1st['PtWinner'].apply(lambda x: True if x == 1 else False)

    # filtro per estrarre i punti in cui player gioca da 2nd player
    df_2nd_player = player_df[player_df['match_id'].str.contains('-' + player)]
    # filtro per estrarre i punti in cui player è in risposta da 2nd player
    player_on_response_as_2nd = df_2nd_player[df_2nd_player['Svr'] == 1]

    # modifica il campo 'PtWinner' con False se è True e viceversa
    player_on_response_as_2nd['PtWinner'] = player_on_response_as_2nd['PtWinner'].apply(lambda x: True if x == 2 else False)

    # unisci i due dataframe
    return pd.concat([player_on_response_as_1st, player_on_response_as_2nd])
