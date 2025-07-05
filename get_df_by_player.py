import pandas as pd


def get_service_points_df(player, df, surface):
    """
    Filtra i punti di servizio di un giocatore su una superficie specifica.

    Args:
        player (str): Nome del giocatore.
        df (pd.DataFrame): DataFrame contenente i dati delle partite.
        surface (str): Tipo di superficie (es. "Clay", "Hard", ecc.).

    Returns:
        pd.DataFrame: DataFrame contenente i punti di servizio del giocatore,
                      con il campo 'PtWinner' modificato per indicare se il giocatore ha vinto il punto.
    """
    # Filtro per estrarre i punti in cui il giocatore gioca su una superficie specifica
    player_df = df[(df['match_id'].str.contains(player)) & (df['surface'] == surface)]
    if player_df is None:
        return None
    # Filtro per estrarre i punti in cui il giocatore gioca come 1st player
    df_1st_player = player_df[player_df['match_id'].str.contains(player + '-')]

    # Filtro per estrarre i punti in cui il giocatore è al servizio come 1st player
    player_at_service_as_1st = df_1st_player[df_1st_player['Svr'] == 1]

    # Modifica il campo 'PtWinner' con 1 (True) se il giocatore ha vinto il punto, 0 (False) altrimenti
    player_at_service_as_1st.loc[:, 'PtWinner'] = player_at_service_as_1st['PtWinner'].apply(
        lambda x: 1 if x == 1 else 0)

    # Filtro per estrarre i punti in cui il giocatore gioca come 2nd player
    df_2nd_player = player_df[player_df['match_id'].str.contains('-' + player)]

    # Filtro per estrarre i punti in cui il giocatore è al servizio come 2nd player
    player_at_service_as_2nd = df_2nd_player[df_2nd_player['Svr'] == 2]

    # Modifica il campo 'PtWinner' con 1 (True) se il giocatore ha vinto il punto, 0 (False) altrimenti
    player_at_service_as_2nd.loc[:, 'PtWinner'] = player_at_service_as_2nd['PtWinner'].apply(
        lambda x: 1 if x == 2 else 0)

    # Unisce i due DataFrame (1st player e 2nd player) e restituisce il risultato
    return pd.concat([player_at_service_as_1st, player_at_service_as_2nd])


def get_response_points_df(player, df, surface):
    """
    Filtra i punti di risposta di un giocatore su una surface specifica.

    Args:
        player (str): Nome del giocatore.
        df (pd.DataFrame): DataFrame contenente i dati delle partite.
        surface (str): Tipo di superficie (es. "Clay", "Hard", ecc.).

    Returns:
        pd.DataFrame: DataFrame contenente i punti di risposta del giocatore,
                      con il campo 'PtWinner' modificato per indicare se il giocatore ha vinto il punto.
    """
    # Filtro per estrarre i punti in cui il giocatore è coinvolto
    player_df = df[df['match_id'].str.contains(player) & (df['surface'] == surface)]
    if player_df is None:
        return None

    # Filtro per estrarre i punti in cui il giocatore gioca come 1st player
    df_1st_player = player_df[player_df['match_id'].str.contains(player + '-')]

    # Filtro per estrarre i punti in cui il giocatore è in risposta come 1st player
    player_on_response_as_1st = df_1st_player[df_1st_player['Svr'] == 2]

    # Modifica il campo 'PtWinner' con 1 (True) se il giocatore ha vinto il punto, 0 (False) altrimenti
    player_on_response_as_1st.loc[:, 'PtWinner'] = player_on_response_as_1st['PtWinner'].apply(
        lambda x: 1 if x == 1 else 0)

    # Filtro per estrarre i punti in cui il giocatore gioca come 2nd player
    df_2nd_player = player_df[player_df['match_id'].str.contains('-' + player)]

    # Filtro per estrarre i punti in cui il giocatore è in risposta come 2nd player
    player_on_response_as_2nd = df_2nd_player[df_2nd_player['Svr'] == 1]

    # Modifica il campo 'PtWinner' con 1 (True) se il giocatore ha vinto il punto, 0 (False) altrimenti
    player_on_response_as_2nd.loc[:, 'PtWinner'] = player_on_response_as_2nd['PtWinner'].apply(
        lambda x: 1 if x == 2 else 0)

    # Unisce i due DataFrame (1st player e 2nd player) e restituisce il risultato
    return pd.concat([player_on_response_as_1st, player_on_response_as_2nd])