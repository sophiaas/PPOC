import numpy as np

def extract_map_features(game_map):
    board_size = np.shape(game_map)
    num_lights = 0
    height = []
    light_idx = []
    for i in range(board_size[0]):
        for j in range(board_size[1]):
            height.append(game_map[i][j]['h'])
            if game_map[i][j]['t'] == 'l':
                light_idx.append(num_lights)
                num_lights += 1
            else:
                light_idx.append(-1)
    x = np.tile(np.arange(board_size[1]), board_size[0])
    y = np.repeat(np.flipud(np.arange(board_size[0])), board_size[1])
    coords = list(zip(x, y))
    board_properties = {a: {"height": b, "light_idx": c} for a, b, c in zip(coords, height, light_idx)}
    max_height = np.max(height)
    return board_size, board_properties, num_lights, max_height
