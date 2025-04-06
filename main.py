import pandas as pd
from attr.validators import min_len
from pymining import seqmining
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display_functions import display

from decode_point import decode_point
from to_seqeunce import to_sequence

#todo: fare seq patt però considerando solo i colpi del servitore, quindi estrarre dal dataset solo le righe in cui il giocatore serve e poi considerare solo i colpi dispari

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

    # vorrei applicare freq_seq_enum a un dataframe, dove gli itemset sono i colpi, codificati come nella funzione decode_point, e una sequenza è un punto intero.
    # Per ogni punto, decodifico i colpi e creo una sequenza di colpi
    df = pd.read_csv('charting-m-points-2020s.csv', dtype={'2nd': str}, low_memory=False)
    # estrai la tabella con i punti giocati da sinner
    sinner = df[df['match_id'].str.contains(r'Jannik_Sinner')]

    # filtro per estrarre i punti in cui sinner gioca da 1st player
    sinner_1st_player = sinner[sinner['match_id'].str.contains('Jannik_Sinner-')]
    # filtro per estrarre i punti in cui sinner è al servizio da 1st player
    sinner_at_service_as_1st = sinner_1st_player[sinner_1st_player['Svr'] == 1]

    # filtro per estrarre i punti in cui sinner gioca da 2nd player
    sinner_2nd_player = sinner[sinner['match_id'].str.contains('-Jannik_Sinner')]
    # filtro per estrarre i punti in cui sinner è al servizio da 2nd player
    sinner_at_service_as_2nd = sinner_2nd_player[sinner_2nd_player['Svr'] == 2]

    # trasforma i punti in cui sinner è al servizio con la prima in liste di colpi
    sinner_at_service_as_1st = sinner_at_service_as_1st['1st'].apply(lambda x: to_sequence(x))

    # drop le righe in cui 2nd è NaN
    sinner_at_service_as_2nd = sinner_at_service_as_2nd.dropna(subset=['2nd'])
    # trasforma i punti in cui sinner è al servizio con la seconda in liste di colpi
    sinner_at_service_as_2nd = sinner_at_service_as_2nd['2nd'].apply(lambda x: to_sequence(x))

    # estraggo la lista dei colpi del servitore con la prima
    sinner_shots_1st_service = sinner_at_service_as_1st.apply(lambda x: [x[i] for i in range(len(x)) if i % 2 == 0])
    print(sinner_shots_1st_service)

    # drop le righe in cui 2nd è NaN
    # estraggo la lista dei colpi del servitore con la seconda
    sinner_shots_2nd_service = sinner_at_service_as_2nd.apply(lambda x: [x[i] for i in range(len(x)) if i % 2 == 0])
    print(sinner_shots_2nd_service)

    min_support = 100
    min_length = 3  # Lunghezza minima della sequenza

    # Trova tutte le sequenze frequenti
    freq_seconds = seqmining.freq_seq_enum(sinner_shots_2nd_service, min_support=min_support)

    # Filtra per lunghezza minima
    filtered_seqs = [(seq, supp) for seq, supp in freq_seconds if len(seq) >= min_length]

    # una volta trovate le sequenze frequenti, ogni colpo che fa parte della sequenza vorrei avesse
    # delle labels che indichino le sue posizioni all'interno dei punti (es quante volte è stato giocato
    # nei primi 3, 6, 9 colpi e il conseguente tasso di vittoria del punto)
    print("filtered sequence: ", filtered_seqs)


if __name__ == "__main__":
    main()
