def get_mapping_dictionaries(map_id):
    """
    Function to get the mapping dictionaries for the point codes.
    These dictionaries map the codes to their respective descriptions.
    The dictionaries include:
    - serve_directions
    - fault_types
    - serve_outcomes
    - shot_types
    - shot_depth
    - court_positions
    - shot_directions
    - error_types
    - point_outcomes
    :return:
    """
    if map_id == "serve_directions":
        return {'4': 'out wide', '5': 'body', '6': 'down the T', '0': 'unknown direction'}
    if map_id == "fault_types":
        return {'n': 'net', 'w': 'wide', 'd': 'deep', 'x': 'both wide and deep', 'g': 'foot fault',
                'e': 'unknown fault', '!': 'shank'}
    if map_id == "serve_outcomes":
        return {'*': 'ace', '#': 'forced error', '@': 'unforced error'}
    if map_id == "serve_and_volley":
        return {'+': 'serve and volley'}
    if map_id == "shot_types":
        return {
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
    if map_id == "shot_depth":
        return {'7': 'short depth', '8': 'medium depth', '9': 'near the baseline depth'}
    if map_id == "court_positions":
        return {"-": "near the net", "=": "from the baseline", "+": "net approach"}
    if map_id == "shot_directions":
        return {'1': 'cross court', '2': 'down the middle', '3': 'down the line',
                '0': 'unknown direction'}
    if map_id == "error_types":
        return {'n': 'net', 'w': 'wide', 'd': 'deep', 'x': 'both wide and deep', '!': 'shank', 'e': 'unknown'}
    if map_id == "point_outcomes":
        return {'*': 'winner', '#': 'forced error', '@': 'unforced error'}
    if map_id == "unusual_situations":
        return {'^': 'stop_volley', ';': "let touch"}
