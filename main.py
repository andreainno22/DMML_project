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
from get_final_datasets import get_final_datasets
from get_shots import get_shots_in_1st_serve_points, get_shots_in_2nd_serve_points
from get_shots_wo_opponent_shots import get_shots_by_server, get_shots_by_receiver
from run_clustering import run_clustering

#todo: problema, per avere pattern giocati da tutti i giocatori devo abbassare parecchio il minsup, allora perde di significato
# todo: sennò potrei trovare i pattern frequenti per ogni giocatore, tutti della stessa lunghezza, poi posso calcolare nuove feature come: %forehand come primo colpo, %backhand come secondo colpo, etc

def main():
    """
    Main function to execute the script.
    """
    # vorrei applicare freq_seq_enum a un dataframe, dove gli itemset sono i colpi, codificati come nella funzione decode_point, e una sequenza è un punto intero.
    # Per ogni punto, decodifico i colpi e creo una sequenza di colpi
    df = pd.read_csv('charting-m-points-2020s.csv', low_memory=False)

    feature_dataset = get_final_datasets(df)
    pd.set_option('display.max_columns', None)
    print(feature_dataset)
    #print(run_clustering(feature_dataset))


if __name__ == "__main__":
    main()
