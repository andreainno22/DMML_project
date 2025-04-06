import pandas as pd
from attr.validators import min_len
from pymining import seqmining
import re
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from win32gui import UpdateLayeredWindow

from decode_point import decode_point
from get_service_points_df import get_service_points_df
from to_seqeunce import to_sequence


# todo: per trovare sequenze di colpi contigue, potrei semplicemente filtrare per punti
# todo: capire perchè restituisce colpi senza la profondità
# di lunghezza n e poi impostare min length a n/2

def main():
    """
    Main function to execute the script.
    """

    # vorrei applicare freq_seq_enum a un dataframe, dove gli itemset sono i colpi, codificati come nella funzione decode_point, e una sequenza è un punto intero.
    # Per ogni punto, decodifico i colpi e creo una sequenza di colpi
    df = pd.read_csv('charting-m-points-2020s.csv', dtype={'2nd': str}, low_memory=False)

    player_at_service = get_service_points_df("Jannik_Sinner", df)

    # trasforma i punti in cui sinner è al servizio con la prima in liste di colpi
    player_at_service_with_1st = player_at_service['1st'].apply(lambda x: to_sequence(x))

    # drop le righe in cui 2nd è NaN
    player_at_service = player_at_service.dropna(subset=['2nd'])
    # trasforma i punti in cui sinner è al servizio con la seconda in liste di colpi
    sinner_at_service_with_2nd = player_at_service['2nd'].apply(lambda x: to_sequence(x))

    # drop le righe in cui lo scambio non è partito (o ace, fault o errore in risposta)
    player_at_service_with_1st = player_at_service_with_1st[player_at_service_with_1st.apply(len) > 2]
    # estraggo la lista dei colpi del servitore con la prima (escludendo i colpi dell'avversario)
    sinner_shots_1st_service = player_at_service_with_1st.apply(lambda x: [x[i] for i in range(len(x)) if i % 2 == 0])
    print(sinner_shots_1st_service)

    # drop le righe in cui lo scambio non è partito (o ace, fault o errore in risposta)
    sinner_at_service_with_2nd = sinner_at_service_with_2nd[sinner_at_service_with_2nd.apply(len) > 2]
    # estraggo la lista dei colpi del servitore con la seconda (escludendo i colpi dell'avversario)
    sinner_shots_2nd_service = sinner_at_service_with_2nd.apply(lambda x: [x[i] for i in range(len(x)) if i % 2 == 0])
    print(sinner_shots_2nd_service)

    min_support = 150
    min_length = 3  # Lunghezza minima della sequenza

    # Trova tutte le sequenze frequenti
    freq_shot_seqs_with_2nd = seqmining.freq_seq_enum(sinner_shots_2nd_service, min_support=min_support)

    # Filtra per lunghezza minima
    filtered_seqs = [(seq, supp) for seq, supp in freq_shot_seqs_with_2nd if len(seq) >= min_length]

    # una volta trovate le sequenze frequenti, ogni colpo che fa parte della sequenza vorrei avesse
    # delle labels che indichino le sue posizioni all'interno dei punti (es quante volte è stato giocato
    # nei primi 3, 6, 9 colpi e il conseguente tasso di vittoria del punto)
    print("filtered sequence: ", filtered_seqs)


if __name__ == "__main__":
    main()
