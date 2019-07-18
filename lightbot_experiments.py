commands = [
    
#     'python run_lightbot.py --env lightbot_minigrid --puzzle_name fractal_cross_0-0 --num_options 8 --max_count 300 --dc 0.05  --wsaves --seed 8 --load_dir experiments/lightbot_minigrid_fractal_cross_0_s8_mc100_me20000_no8_dc0.05/ --epoch 70 --name sparse_'
    
    
    'python run_lightbot.py --env lightbot_minigrid --puzzle_name fractal_cross_0-1 --num_options 8 --max_count 500 --dc 0.05  --wsaves --seed 21 --name sparse_transfer --load_dir experiments/sparse_from_scratch_lightbot_minigrid_fractal_cross_0_s12_mc100_me20000_no8_dc0.05_lr1e-05/lightbot_minigridseed12_epoch_170.ckpt --epoch 170'
    
]


import os
for command in commands:
    print(command)

    os.system(command)
