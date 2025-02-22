maps = {
  "debug1": {
    "direction":0,
    "position": {"x": 0, "y": 3},
    "map":
      [[{'h': 0, 't': 'b'}],
       [{'h': 0, 't': 'l'}],
       [{'h': 1, 't': 'l'}],
       [{'h': 0, 't': 'l'}],
      ]
  # optimal number of steps: 6 steps
  # optimal reward: 0
  # [[' 0* ' ' 1* ' ' 0* ' '<0  ']]
  },
  "debug2": {
    "direction":1,
    "position": {"x": 2, "y": 2},
    "map":
          [[{'h': 0, 't': 'b'},
            {'h': 1, 't': 'l'},
            {'h': 0, 't': 'l'}],
           [{'h': 1, 't': 'b'},
            {'h': 2, 't': 'b'},
            {'h': 0, 't': 'b'}],
           [{'h': 1, 't': 'l'},
            {'h': 1, 't': 'l'},
            {'h': 0, 't': 'b'}]]
  # optimal number of steps: 12 steps
  # optimal reward: -4
  # [[' 1* ' ' 1  ' ' 0  ']
  #  [' 1* ' ' 2  ' ' 1* ']
  #  [' 0  ' ' 0  ' 'v0* ']]
  },
  # 0 tutorial
    'stairchunks': {"direction":0,"position": {"x": 0, "y": 0}, "map":[[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":2, "t":"b"},{"h":3, "t":"b"},{"h":4, "t":"l"},{"h":2, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":4, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":3, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":3, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":4, "t":"l"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":2, "t":"b"},{"h":4, "t":"l"},{"h":3, "t":"b"},{"h":2, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}]]},
    '2-3stripes':
    {"direction":0,"position": {"x": 0, "y": 0}, "map":[[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}]]},
    '2-3stripes_small':
    {"direction":0,"position": {"x": 0, "y": 0}, "map":[[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"}],[{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}]]},
    '2stripes':
    {"direction":0,
     "position": {"x": 0, "y": 0}, 
     "map":[[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"}],[{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":2, "t":"l"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}]]},
    '2square':
    {"direction":0,"position": 
     {"x": 0, "y": 0}, 
     "map":[[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}]]},
    '2tetris':
    {"direction":0,"position": {"x": 0, "y": 0}, "map":[[{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"}]]},
    '2zigzag':
    {"direction":0,"position": {"x": 0, "y": 0}, "map":[[{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"}]]},
    'fractal_cross_0': {
        "direction": 0, 
        "position": {"x": 1, "y": 2}, 
        "map":
        [[{"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"}],
         [{"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"}],
         [{"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"}]],
    },
    'fractal_cross_0-1': {
        "direction":0,
        "position": {"x": 5, "y": 1}, 
        "map":[[{"h":1, "t":"b"},{"h":2, "t":"l"},
                {"h":1, "t":"b"},{"h":1, "t":"b"},
                {"h":2, "t":"l"},{"h":1, "t":"b"}],
               [{"h":2, "t":"l"},{"h":2, "t":"l"},
                {"h":2, "t":"l"},{"h":2, "t":"l"},
                {"h":2, "t":"l"},{"h":2, "t":"l"}],
               [{"h":1, "t":"b"},{"h":2, "t":"l"},
                {"h":1, "t":"b"},{"h":1, "t":"b"},
                {"h":2, "t":"l"},{"h":1, "t":"b"}]],
    },
    'fractal_cross_0-2': {
        "direction":0,"position": 
        {"x": 1, "y": 5}, 
        "map":[[{"h":1, "t":"b"},
                {"h":2, "t":"l"},
                {"h":1, "t":"b"},
                {"h":1, "t":"b"},
                {"h":2, "t":"l"},
                {"h":1, "t":"b"}],
               [{"h":2, "t":"l"},
                {"h":2, "t":"l"},
                {"h":2, "t":"l"},
                {"h":2, "t":"l"},
                {"h":2, "t":"l"},
                {"h":2, "t":"l"}],
               [{"h":1, "t":"b"},
                {"h":2, "t":"l"},
                {"h":1, "t":"b"},
                {"h":1, "t":"b"},
                {"h":2, "t":"l"},
                {"h":1, "t":"b"}],
               [{"h":1, "t":"b"},
                {"h":2, "t":"l"},
                {"h":1, "t":"b"},
                {"h":1, "t":"b"},
                {"h":1, "t":"b"},
                {"h":1, "t":"b"}],
               [{"h":2, "t":"l"},
                {"h":2, "t":"l"},
                {"h":2, "t":"l"},
                {"h":1, "t":"b"},
                {"h":1, "t":"b"},
                {"h":1, "t":"b"}],
               [{"h":1, "t":"b"},
                {"h":2, "t":"l"},
                {"h":1, "t":"b"},
                {"h":1, "t":"b"},
                {"h":1, "t":"b"},
                {"h":1, "t":"b"}]],
                },
    'fractal_cross_0-3': {
        "direction":0,"position": 
        {"x": 0, "y": 0}, 
        "map":[[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"}],[{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"}],[{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"}],[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"}]]
    },
    'fractal_cross_1': {
        "direction": 0,
        "position": {"x": 4, "y": 8}, 
        "map":
        [[{"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"}],
         [{"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"}],
         [{"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"}],
         [{"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"}],
         [{"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"}],
         [{"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"}],
         [{"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"}],
         [{"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"}],
         [{"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":2, "t":"l"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"},
          {"h":1, "t":"b"}]],
        },
  'fractal_cross_2': 
    {"direction":0,
     "position": {"x": 0, "y": 0}, 
     "map":
     [[{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"}],
      [{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"}],
      [{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}],
      [{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":2, "t":"l"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"},{"h":1, "t":"b"}]],
  },
  '0_tutorial': {
    "direction":0,
    "position": {"x": 3, "y": 5}, 
    "map" :
      [[{"h":2, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"}],
       [{"h":2, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"}],
       [{"h":2, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"}],
       [{"h":2, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"}],
       [{"h":2, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"l"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"}],
       [{"h":2, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"}],
       [{"h":2, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"}]],
    "medals":{"gold":100,"silver":4,"bronze":5}
  # optimal number of steps:
  # optimal reward:
  # [[' 2  ' ' 2  ' ' 2  ' ' 2  ' ' 2  ' ' 2  ' ' 2  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1* ' ' 1  ' ' 1  ' '<1  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 2  ' ' 2  ' ' 2  ' ' 2  ' ' 2  ' ' 2  ' ' 2  ']]
  },
  # 1 tutorial
  '1_tutorial': {
    "direction":0,
    "position": {"x": 1, "y": 3}, 
    "map" :
      [[{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"}
        ,{"h":1, "t":"b"},
        {"h":3, "t":"b"},
        {"h":4, "t":"b"},
        {"h":4, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":3, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":2, "t":"b"},
        {"h":2, "t":"b"},
        {"h":3, "t":"l"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}]],
    "medals":{"gold":100,"silver":4,"bronze":5}
  # optimal number of steps:
  # optimal reward:
  # [[' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 2  ' ' 1  ' '<1  ' ' 1  ']
  #  [' 1  ' ' 2  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 3* ' ' 3  ' ' 3  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 4  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 4* ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']]
  },
  # 2 tutorial
  '2_tutorial': {
    "direction":3,
    "position": {"x": 3, "y": 3}, 
    "map" :
      [[{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":3, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"l"},
        {"h":2, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}]],
    "medals":{"gold":100,"silver":4,"bronze":5}
  # optimal number of steps:
  # optimal reward:
  # [[' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1* ' ' 2* ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 2* ' ' 3* ' '^1  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']]
  },
  # 3 stairs
  'stairs': {
    "direction":1,
    "position": {"x": 1, "y": 7}, 
    "map" :
      [[{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":3, "t":"l"},
        {"h":4, "t":"l"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":3, "t":"l"},
        {"h":4, "t":"l"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":3, "t":"l"},
        {"h":4, "t":"l"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":3, "t":"l"},
        {"h":4, "t":"l"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":3, "t":"l"},
        {"h":4, "t":"l"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":3, "t":"l"},
        {"h":4, "t":"l"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}]],
    "medals":{"gold":100,"silver":4,"bronze":5}
  # optimal number of steps: 46
  # optimal reward:
  # [[' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' 'v1  ' ' 1  ']
  #  [' 1  ' ' 2* ' ' 2* ' ' 2* ' ' 1  ' ' 2* ' ' 2* ' ' 2* ' ' 1  ']
  #  [' 1  ' ' 3* ' ' 3* ' ' 3* ' ' 1  ' ' 3* ' ' 3* ' ' 3* ' ' 1  ']
  #  [' 1  ' ' 4* ' ' 4* ' ' 4* ' ' 1  ' ' 4* ' ' 4* ' ' 4* ' ' 1  ']]
  },
  # 4 cross
  'cross': {
    "direction":3,
    "position": {"x": 2, "y": 0}, 
    "map" :
      [[{"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":2, "t":"b"},
        {"h":2, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":2, "t":"l"},
        {"h":2, "t":"l"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":2, "t":"l"}],
       [{"h":2, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"}],
       [{"h":2, "t":"l"},
        {"h":2, "t":"l"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":2, "t":"l"}],
       [{"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":2, "t":"b"},
        {"h":2, "t":"l"},
        {"h":1, "t":"b"}]],
    "medals":{"gold":100,"silver":4,"bronze":5}
  # optimal number of steps: 38
  # optimal reward:
  # [[' 1  ' ' 2* ' ' 2  ' ' 2* ' ' 1  ']
  #  [' 2* ' ' 2* ' ' 1  ' ' 2* ' ' 2* ']
  #  ['^2  ' ' 1  ' ' 1  ' ' 1  ' ' 2  ']
  #  [' 2* ' ' 2* ' ' 1  ' ' 2* ' ' 2* ']
  #  [' 1  ' ' 2* ' ' 2  ' ' 2* ' ' 1  ']]
  },
  # 5 monolith
  'monolith': {
    "direction":1,
    "position": {"x": 1, "y": 7}, 
    "map" :
      [[{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}]],
    "medals":{"gold":100,"silver":4,"bronze":5}
  # optimal number of steps: 46
  # optimal reward:
  # [[' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' 'v1* ' ' 1  ']
  #  [' 1  ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' ' 1  ']
  #  [' 1  ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' ' 1* ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']]
  },
  # 6 stairs_2
  'stairs_2': {
    "direction":1,
    "position": {"x": 0, "y": 4}, 
    "map" :
      [[{"h":2, "t":"b"},
        {"h":3, "t":"b"},
        {"h":4, "t":"l"},
        {"h":5, "t":"b"},
        {"h":5, "t":"b"},
        {"h":6, "t":"b"},
        {"h":7, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":4, "t":"b"},
        {"h":5, "t":"b"},
        {"h":6, "t":"l"},
        {"h":6, "t":"b"},
        {"h":7, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":6, "t":"b"},
        {"h":7, "t":"b"},
        {"h":8, "t":"l"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}]],
    "medals":{"gold":100,"silver":4,"bronze":5}
  # optimal number of steps: 13
  # optimal reward:
  # [[' 1  ' ' 1  ' ' 1  ' ' 1  ' 'v2  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 3  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 4  ' ' 4* ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 5  ' ' 5  ']
  #  [' 1  ' ' 1  ' ' 6  ' ' 6* ' ' 5  ']
  #  [' 1  ' ' 1  ' ' 7  ' ' 6  ' ' 6  ']
  #  [' 1  ' ' 1  ' ' 8* ' ' 7  ' ' 7  ']]
  },
  # 7 little_l
  'little_l': {
    "direction":2,
    "position": {"x": 1, "y": 3}, 
    "map" :
      [[{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":2, "t":"b"},
        {"h":2, "t":"l"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":2, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"},
        {"h":2, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"},
        {"h":2, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
        [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}]],
    "medals":{"gold":100,"silver":4,"bronze":5}
  # optimal number of steps: 17
  # optimal reward:
  # [[' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' '>1  ' ' 2  ' ' 2  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 2* ' ' 1  ']
  #  [' 1  ' ' 2  ' ' 2* ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 2  ' ' 1  ' ' 1  ' ' 2  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 2* ' ' 2  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']]
  },
  # 8 zigzag
  'zigzag': {
    "direction":0,
    "position": {"x": 1, "y": 4}, 
    "map" :
      [[{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"l"},
        {"h":3, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"},
        {"h":3, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":2, "t":"b"},
        {"h":3, "t":"b"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"l"},
        {"h":2, "t":"b"},
        {"h":3, "t":"l"},
        {"h":1, "t":"b"}],
       [{"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"},
        {"h":1, "t":"b"}]],
    "medals":{"gold":100,"silver":4,"bronze":5}
  # optimal number of steps: 13
  # optimal reward:
  # [[' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']
  #  [' 1  ' ' 1* ' ' 1  ' ' 1  ' '<1  ' ' 1  ']
  #  [' 1  ' ' 2  ' ' 2  ' ' 2  ' ' 2* ' ' 1  ']
  #  [' 1  ' ' 3* ' ' 3  ' ' 3  ' ' 3  ' ' 1  ']
  #  [' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ' ' 1  ']]
  }
}

