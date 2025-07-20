def get_shots_by_server(shots):
    # estraggo la lista dei colpi del servitore (escludendo i colpi dell'avversario)
    shots.shots = shots.shots.apply(lambda x: [x[i] for i in range(len(x)) if i % 2 == 0])
    return shots


def get_shots_by_receiver(shots):
    # estraggo la lista dei colpi del ricevitore (escludendo i colpi del servitore)
    shots.shots = shots.shots.apply(lambda x: [x[i] for i in range(len(x)) if i % 2 == 1])
    return shots
