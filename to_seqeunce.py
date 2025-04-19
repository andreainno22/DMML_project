from get_mapping_dictionaries import get_mapping_dictionaries


def to_sequence(point_code):
    list_of_shots = []  # list of all the shots in point
    """
    Function to convert a point code into a sequence of shots.
    """
    fault_types = get_mapping_dictionaries("fault_types")
    serve_outcomes = get_mapping_dictionaries("serve_outcomes")
    serve_and_volley = get_mapping_dictionaries("serve_and_volley")
    shot_types = get_mapping_dictionaries("shot_types")
    shot_depth = get_mapping_dictionaries("shot_depth")
    court_positions = get_mapping_dictionaries("court_positions")
    shot_directions = get_mapping_dictionaries("shot_directions")
    error_types = get_mapping_dictionaries("error_types")
    point_outcomes = get_mapping_dictionaries("point_outcomes")
    unusual_situations = get_mapping_dictionaries("unusual_situations")

    serve = ""
    outcome = ""

    # detect the serve inside the rally
    char = point_code[0]
    if len(point_code) == 1:
        return point_code, ""
    serve += char  # serve direction
    if point_code[1] in serve_and_volley:
        serve += point_code[1]
        if 2 < len(point_code) and point_code[2] in serve_outcomes or point_code[2] in fault_types:
            serve += point_code[2]
            start_rally_index = 3
        else:
            start_rally_index = 2
    elif point_code[1] in serve_outcomes or point_code[1] in fault_types:
        serve += point_code[1]
        start_rally_index = 2
    else:
        start_rally_index = 1

    list_of_shots.append(serve)

    i = start_rally_index

    # transform a rally in a list of shots
    while i < len(point_code):
        char = point_code[i]
        shot = ""
        if char in shot_types:
            shot += char
            if i + 1 < len(point_code) and (point_code[i + 1] in unusual_situations or point_code[
                i + 1] in court_positions):
                shot += point_code[i + 1]
                i += 1
            if i + 1 < len(point_code) and (point_code[i + 1] in court_positions or point_code[
                i + 1] in unusual_situations):
                shot += point_code[i + 1]
                i += 1
            if i + 1 < len(point_code) and point_code[i + 1] in shot_directions:
                shot += point_code[i + 1]
                i += 1
            # aggiungo shot_depth solo alla risposta al servizio, lo ignoro altrimenti
            if i + 1 < len(point_code) and point_code[i + 1] in shot_depth and len(list_of_shots) == 1:
                # ignora la profondità della risposta al servizio
                #shot += point_code[i + 1]
                i += 1
            if i + 1 < len(point_code) and point_code[i + 1] in error_types:
                # il codice dell'error type non viene aggiunto all' outcome perchè non utile nell'analisi
                if i + 2 < len(point_code) and point_code[i + 2] in point_outcomes:
                    outcome += point_code[i + 2]
                    i += 2
                else:
                    i += 1
            elif i + 1 < len(point_code) and point_code[i + 1] in point_outcomes:  # a winner
                outcome = point_code[i + 1]
                i += 1
            list_of_shots.append(shot)
        i += 1

    return list_of_shots, outcome
