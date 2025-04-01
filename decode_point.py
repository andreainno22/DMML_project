def decode_point(point_code):
    """
    Decodes the entire point code into a human-readable description.
    """
    serve_directions = {'4': 'out wide', '5': 'body', '6': 'down the T', '0': 'unknown direction'}
    fault_types = {'n': 'net', 'w': 'wide', 'd': 'deep', 'x': 'both wide and deep', 'g': 'foot fault',
                   'e': 'unknown fault', '!': 'shank'}
    serve_outcomes = {'*': 'ace', '#': 'forced error', '@': 'unforced error'}
    shot_types = {
        'f': 'forehand groundstroke', 'b': 'backhand groundstroke',
        'r': 'forehand slice', 's': 'backhand slice',
        'v': 'forehand volley', 'z': 'backhand volley',
        'o': 'overhead/smash', 'p': 'backhand overhead/smash',
        'u': 'forehand drop shot', 'y': 'backhand drop shot',
        'l': 'forehand lob', 'm': 'backhand lob',
        'h': 'forehand half-volley', 'i': 'backhand half-volley',
        'j': 'forehand swinging volley', 'k': 'backhand swinging volley',
        't': 'trick shot', 'q': 'unknown shot'
    }
    shot_depth = {'7': 'short depth', '8': 'medium depth', '9': 'near the baseline depth'}
    court_positions = {"-": "near the net", "=": "from the baseline"}
    shot_directions = {'1': 'cross court', '2': 'down the middle', '3': 'down the line',
                       '0': 'unknown direction'}
    error_types = {'n': 'net', 'w': 'wide', 'd': 'deep', 'x': 'both wide and deep', '!': 'shank', 'e': 'unknown'}
    point_outcomes = {'*': 'winner', '#': 'forced error', '@': 'unforced error'}

    description = []
    i = 0

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
                if i + 2 < len(point_code) and point_code[i + 2] in serve_outcomes:
                    shot_desc += f" ({point_outcomes[point_code[i + 2]]})"
                    i += 2
                else:
                    i += 1
            description.append(shot_desc)
        i += 1

    return ';  '.join(description)
