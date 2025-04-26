import pandas as pd
import numpy as np
from to_seqeunce import to_sequence


def get_shots_in_1st_serve_points(shots_df):
    # trasforma i punti in cui player è al servizio con la prima in liste di colpi
    shots_sequence = shots_df['1st'].apply(lambda x: to_sequence(x))
    shots, outcome = zip(*shots_sequence)

    # aggiunge un flag che indica se il punto è vinto da player 1
    shots_sequence = pd.DataFrame({
        'shots': shots,
        'won_by_player': shots_df['PtWinner'] == 1,
        'outcome': np.where(shots_df['PtWinner'] == 1, [o + ", won" for o in outcome], [o + ", lost" for o in outcome])
    })

    # drop le righe in cui lo scambio non è partito (o ace, fault o errore in risposta)
    # shots_sequence = shots_sequence[shots_sequence.shots.apply(len) > 2]

    return shots_sequence


def get_shots_in_2nd_serve_points(shots_df):
    # drop delle righe in cui 2nd è NaN senza modificare gli indici di player_at_service
    shots_df_with_no_na = shots_df.dropna(subset=['2nd'])

    # trasforma i punti in cui sinner è al servizio con la seconda in liste di colpi
    shots_sequence = shots_df_with_no_na['2nd'].apply(lambda x: to_sequence(x))
    shots, outcome = zip(*shots_sequence)

    # aggiunge un flag che indica se il punto è vinto dal player
    shots_sequence = pd.DataFrame({
        'shots': shots,
        'won_by_player': shots_df_with_no_na['PtWinner'] == 1,
        'outcome': np.where(shots_df_with_no_na['PtWinner'] == 1, [o + ", won" for o in outcome],
                            [o + ", lost" for o in outcome])
    })

    shots_sequence.dropna(subset=['shots'], inplace=True)

    # drop le righe in cui lo scambio non è partito (o ace, fault o errore in risposta)
    # shots_sequence = shots_sequence[shots_sequence.shots.apply(len) > 2]

    return shots_sequence
