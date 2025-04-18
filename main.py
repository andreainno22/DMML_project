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
from get_pattern_dictionary import get_pattern_dictionary
from get_shots import get_shots_in_1st_serve_points, get_shots_in_2nd_serve_points
from get_shots_wo_opponent_shots import get_shots_by_server, get_shots_by_receiver
from run_clustering import run_clustering


def main():
    """
    Main function to execute the script.
    """
    # vorrei applicare freq_seq_enum a un dataframe, dove gli itemset sono i colpi, codificati come nella funzione decode_point, e una sequenza è un punto intero.
    # Per ogni punto, decodifico i colpi e creo una sequenza di colpi
    df = pd.read_csv('charting-m-points-2020s.csv', low_memory=False)

    feature_dataset = get_pattern_dictionary(df)
    print(run_clustering(feature_dataset))


if __name__ == "__main__":
    main()
