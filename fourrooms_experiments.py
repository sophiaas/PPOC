commands = [
    
    'python run_lightbot.py --env fourrooms --num_options 4 --max_count 200 --dc 0.1 --wsaves'
]


import os
for command in commands:
    print(command)

    os.system(command)
