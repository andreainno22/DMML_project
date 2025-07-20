from get_mapping_dictionaries import get_mapping_dictionaries


def decode_point(point_code):
    """
    Decodes the entire point code into a human-readable description.
    """
    description = []
    i = 0
    serve_directions = get_mapping_dictionaries("serve_directions")
    fault_types = get_mapping_dictionaries("fault_types")
    serve_outcomes = get_mapping_dictionaries("serve_outcomes")
    shot_types = get_mapping_dictionaries("shot_types")
    shot_depth = get_mapping_dictionaries("shot_depth")
    court_positions = get_mapping_dictionaries("court_positions")
    shot_directions = get_mapping_dictionaries("shot_directions")
    error_types = get_mapping_dictionaries("error_types")
    point_outcomes = get_mapping_dictionaries("point_outcomes")

    # Decode serve
    while i < len(point_code):
        char = point_code[i]
        if char in serve_directions:
            description.append(f"Serve {serve_directions[char]}")
            if i + 1 < len(point_code) and point_code[i + 1] in serve_outcomes:
                description[-1] += f" ({serve_outcomes[point_code[i + 1]]})"
                i += 1
        elif char in fault_types:
            description.append(f"Fault ({fault_types[char]})")
        elif char == 'c':
            description.append("Let serve")
        elif char == 'V':
            description.append("Time violation")
        elif char == '+':
            description.append("Serve-and-volley attempt")
        else:
            break
        i += 1

    # Decode rally
    while i < len(point_code):
        char = point_code[i]
        if char in shot_types:
            shot_desc = shot_types[char] + ","
            if i + 1 < len(point_code) and point_code[i + 1] in shot_directions:
                shot_desc += f" {shot_directions[point_code[i + 1]]},"
                i += 1
            if i + 1 < len(point_code) and point_code[i + 1] in court_positions:
                shot_desc += f" {court_positions[point_code[i + 1]]},"
                i += 1
            if i + 1 < len(point_code) and point_code[i + 1] in shot_depth:
                shot_desc += f" {shot_depth[point_code[i + 1]]}"
                i += 1
            if i + 1 < len(point_code) and point_code[i + 1] in error_types:
                shot_desc += f" ({error_types[point_code[i + 1]]})"
                if i + 2 < len(point_code) and point_code[i + 2] in point_outcomes:
                    shot_desc += f" ({point_outcomes[point_code[i + 2]]})"
                    i += 2
                else:
                    i += 1
            description.append(shot_desc)
        i += 1

    return ';  '.join(description)
