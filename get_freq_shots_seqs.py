from pymining import seqmining


# Verifica che gli elementi di seq siano in x nello stesso ordine
def is_subsequence(seq, x):
    it = iter(x)
    return all(item in it for item in seq)


def get_freq_shots_seqs(shots):
    """
    Count the frequency of each shot sequence in the list of shots.
    """

    min_support = len(shots["shots"])*0.1  # Supporto minimo per le sequenze frequenti
    min_length = 2  # Lunghezza minima della sequenza

    # Trova tutte le sequenze frequenti
    freq_shots_seqs = seqmining.freq_seq_enum(shots.shots, min_support=min_support)

    # Calcola la percentuale di vittoria per ogni sequenza frequente
    results = []

    shots['shots'] = shots['shots'].apply(tuple)

    for seq, support in freq_shots_seqs:
        # Filtra i punti in cui la sequenza appare
        matches = shots[shots['shots'].apply(lambda x: x if is_subsequence(seq, x) else None).notnull()]

        # Calcola la percentuale di vittoria
        win_percentage = matches['won_by_player'].mean() * 100  # Media dei valori booleani (True = 1, False = 0)
        most_frequent_outcome = matches['outcome'].mode()[0] if not matches['outcome'].empty else None

        # Aggiungi la sequenza, il supporto e la percentuale di vittoria ai risultati
        results.append((seq, support, win_percentage, most_frequent_outcome))

    # Filtra per lunghezza minima
    results = [(seq, support, win_percentage, most_frequent_outcome) for seq, support, win_percentage, most_frequent_outcome in results if len(seq) >= min_length]
    # Stampa i risultati
    for result in results:
        yield result
