import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display_functions import display

from decode_point import decode_point


def load_data():
    """
    Load data from a CSV file.
    """
    file_path = 'charting-m-points-2020s.csv'
    data = pd.read_csv(file_path)
    return data


def main():
    """
    Main function to execute the script.
    """
    df = pd.read_csv('charting-m-points-2020s.csv', low_memory=False)
    # dovrei riuscire a estrarre le righe che nel campo match id contengono il nome di un giocatore specifico
    zverev = df[df['match_id'].str.contains('Alexander_Zverev')]
    sinner = df[df['match_id'].str.contains(r'2024.*Jannik_Sinner')]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    # estrai solo alcune colonne
    sinner = sinner[['match_id', 'Pts', 'Svr', '1st', '2nd', 'PtWinner']]
    print(decode_point("cc0x"))
    # Esempio di utilizzo
    point_code = "5f-82f1f1v2n@"
    print(decode_point(point_code))
    #print(sinner.head(10))


if __name__ == "__main__":
    main()
