import pandas as pd
from attr.validators import min_len
from pymining import seqmining
import re
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from win32gui import UpdateLayeredWindow
import os
from get_df_by_player import get_response_points_df, get_service_points_df
from get_freq_shots_seqs import get_freq_shots_seqs
from get_final_datasets import get_final_datasets
from get_shots import get_shots_in_1st_serve_points, get_shots_in_2nd_serve_points
from get_shots_wo_opponent_shots import get_shots_by_server, get_shots_by_receiver
from run_clustering import run_kmeans_clustering


def main():
    """
    Main function to execute the script.
    """
    # vorrei applicare freq_seq_enum a un dataframe, dove gli itemset sono i colpi, codificati come nella funzione decode_point, e una sequenza è un punto intero.
    # Per ogni punto, decodifico i colpi e creo una sequenza di colpi
    df = pd.read_csv('points_datasets/charting-m-points-2020s.csv', low_memory=False)

    feature_datasets = get_final_datasets(df)
    pd.set_option('display.max_columns', None)
    print("numero di contesti:", len(feature_datasets))
    for name, df in feature_datasets.items():
        print(f"Nome: {name}")
        print(df)
        print("\n" + "=" * 50 + "\n")  # Separatore per migliorare la leggibilità

    # Crea una directory per salvare i file CSV, se non esiste
    output_dir = "feature_datasets_csv_reduced_contexts"
    os.makedirs(output_dir, exist_ok=True)

    # Itera su ogni elemento del dizionario e salva i DataFrame in file CSV
    for name, df in feature_datasets.items():
        # Sostituisci eventuali caratteri non validi nel nome del file
        safe_name = re.sub(r'[^\w\-_\. ]', '_', name)
        file_path = os.path.join(output_dir, f"{safe_name}.csv")

        # Salva il DataFrame in formato CSV
        df.to_csv(file_path, index=True)
        print(f"Salvato: {file_path}")

    #print(run_clustering(feature_dataset))


if __name__ == "__main__":
    main()
