# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import os.path as osp
import os
import gym, logging
import pdb
import pickle

from baselines import logger
import sys

def set_env(params):
    
    if params.env == 'hanoi':
        from hanoi_env.env import HanoiEnv
        params.model_type = 'rnn'
        env = HanoiEnv()
        env.set_env_parameters(max_count=params.max_count, num_disks=params.num_disks, 
                                    num_pegs=params.num_pegs, allow_impossible=params.allow_impossible, 
                                    continual=params.continual, initial_peg=params.initial_peg)
        
    elif params.env == 'lightbot_minigrid':
        from gym_minigrid.envs import LightbotEnv as LightbotMinigridEnv
        from gym_minigrid.wrappers import ImgObsWrapper, AgentViewWrapper
        params.model_type = 'cnn'
        env = LightbotMinigridEnv(params.puzzle_name, reward_fn=params.rewards, 
                                  max_steps=params.max_count, toggle_ontop=False)
        env = ImgObsWrapper(AgentViewWrapper(env, agent_view_size=9))
        
    elif params.env == 'lightbot':
        from lightbot_env.env import LightbotEnv
        params.model_type = 'mlp'
        env = LightbotEnv(params.puzzle_name)
        env.set_env_parameters(max_count=params.max_count, testing=params.testing, 
                                reward_fn=params.rewards, random_init=params.random_init,
                                allow_impossible=params.allow_impossible)
        
    elif params.env == 'fourrooms':
        from fourrooms.fourrooms import Fourrooms
        params.model_type = 'mlp'
        env = Fourrooms(max_count=params.max_count)
        
    elif params.env == 'fourrooms_minigrid':
        params.model_type = 'cnn'
        raise NotImplementedError
    return env, params

def train(env_id, args):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env, args = set_env(args)
    if args.env == 'lightbot':
        board_properties = {'board_size': [int(x) for x in env.board_size], 
                            'num_lights': int(env.num_lights), 
                            'max_height': int(env.max_height)}

        with open(args.save_dir + '/board_properties.p', 'wb') as file:
            pickle.dump(board_properties, file)

    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=args.num_options, dc=args.dc, 
                                    head=args.model_type)
#     env = bench.Monitor(env, logger.get_dir() and 
#         osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
#     gym.logger.setLevel(logging.WARN)

    if args.num_options == 1:
        optimsize = 64
    elif args.num_options == 2:
        optimsize = 32
    else:
        optimsize = 32
#     else:
#         print("Only two options or primitive actions is currently supported.")
#         sys.exit()

    pposgd_simple.learn(env, policy_fn, 
            max_episodes=args.max_episodes,
            timesteps_per_batch=args.max_count*50,
            clip_param=0.1, entcoeff=0.0,
            optim_epochs=5, optim_stepsize=3e-4, optim_batchsize=optimsize,
            gamma=1.0, lam=0.95, schedule='linear', num_options=args.num_options,
            app=args.app, saves=args.saves, wsaves=args.wsaves, epoch=args.epoch, 
            seed=args.seed,dc=args.dc, save_dir=args.save_dir, 
            load_dir=args.load_dir, init_lr=args.init_lr)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment name', default='lightbot')
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--max_count', help='max steps per episode', type=int, default=100)
    parser.add_argument('--max_episodes', help='max episodes', type=int, default=20000)
#     parser.add_argument('--opt', help='number of options', type=int, default=2) 
    parser.add_argument('--app', help='Append to folder name', type=str, default='')        
    parser.add_argument('--saves', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', dest='wsaves', action='store_true', default=False)    
    parser.add_argument('--epoch', help='Epoch', type=int, default=-1) 
    parser.add_argument('--dc', type=float, default=0.)
    parser.add_argument('--init_lr', type=float, default=1e-5)
    parser.add_argument('--num_options', type=int, default=2)
    parser.add_argument('--load_dir', type=str, default=None)
    parser.add_argument('--name', type=str, default=None)
    
    #hanoi args
    parser.add_argument('--num_disks', help='num disks for hanoi', type=int, default=3)
    parser.add_argument('--continual', action='store_true', default=False)
    parser.add_argument('--num_pegs', type=int, default=3)
    parser.add_argument('--initial_peg', type=int, default=None)
    #lightbot args
    parser.add_argument('--puzzle_name', type=str, default='cross')
    parser.add_argument('--random_init', action='store_true', default=False)
    parser.add_argument('--allow_impossible', action='store_true', default=True)
    parser.add_argument('--rewards', type=str, default='1000,-1,-1,-1')
    parser.add_argument('--testing', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if args.name:
        args.save_dir = 'experiments/{}_{}_{}_s{}_mc{}_me{}_no{}_dc{}_lr{}'.format(args.name, args.env, args.puzzle_name, args.seed, args.max_count, args.max_episodes, args.num_options, args.dc, args.init_lr)
    else:
        args.save_dir = 'experiments/{}_{}_s{}_mc{}_me{}_no{}_dc{}'.format(args.env, args.puzzle_name, args.seed, args.max_count, args.max_episodes, args.num_options, args.dc)
    
    if args.epoch > 0:
        args.save_dir += '_checkpoint{}'.format(args.load_dir)
    
    args.save_dir += '/'

    os.makedirs(args.save_dir)
    
    train(args.env, args)
#         seed=args.seed, num_options=args.opt, app=args.app, saves=args.saves, wsaves=args.wsaves, epoch=args.epoch, dc=args.dc)


if __name__ == '__main__':
    main()
