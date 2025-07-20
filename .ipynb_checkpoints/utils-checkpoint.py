from costants import SHOT_LENGTH


# Funzione per filtrare e tagliare i colpi a SHOT_LENGTH
def filter_and_trim_shots(shots_df):
    shots_df = shots_df[shots_df['shots'].apply(len) >= SHOT_LENGTH]
    shots_df["shots"] = shots_df['shots'].apply(lambda x: x[:SHOT_LENGTH])
    return shots_df
