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
    # estrai la tabella con i punti giocati da sinner, proiettata solo sulla colonna 1st serve
    sinner = df[df['match_id'].str.contains(r'Jannik_Sinner')]
    # crea una nuova colonna "Server" con il nome e cognome del giocatore che ha servito. Sinner ha servito se appare per primo in match id e se Svr == 1 oppure se appare secondo in match id e Svr == 2
    sinner_at_service = sinner[sinner['match_id'].str.contains(r'Jannik_Sinner-(\w+),.+')]
    print(sinner_at_service)


    sinner = sinner.dropna(subset=['2nd'])
    sinner['1st'] = sinner['1st'].apply(lambda x: to_sequence(x))
    sinner['2nd'] = sinner['2nd'].apply(lambda x: to_sequence(x))
    # ritorna i record di 2nd di tipo float

    # estrai solo gli elementi dispari di 2nd, i pari sono dell'avversario
    sinner_shots = sinner['2nd'].apply(lambda x: [x[i] for i in range(len(x)) if i % 2 != 0])

    min_support = 200
    min_length = 5  # Lunghezza minima della sequenza

    # Trova tutte le sequenze frequenti
    freq_seconds = seqmining.freq_seq_enum(sinner['2nd'], min_support=min_support)

    # Filtra per lunghezza minima
    filtered_seqs = [(seq, supp) for seq, supp in freq_seconds if len(seq) >= min_length]

    # una volta trovate le sequenze frequenti, ogni colpo che fa parte della sequenza vorrei avesse delle labels che indichino le sue posizioni all'interno dei punti (es quante volte è stato giocato nei primi 3, 6, 9 colpi e il conseguente tasso di vittoria del punto)
    print(filtered_seqs)


if __name__ == "__main__":
    main()
