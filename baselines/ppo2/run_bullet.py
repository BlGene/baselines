#!/usr/bin/env python3
import numpy as np
from baselines.common.cmd_util import atari_arg_parser
import sys
from baselines import logger
from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.ppo2 import ppo2
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy, MlpPolicy
import multiprocessing
import tensorflow as tf

import os, logging, gym
from baselines import bench
from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
import gym_grasping


def train(env_id, num_timesteps, seed, policy, save_interval):

    num_env = 16
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    tf.Session(config=config).__enter__()

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            gym.logger.setLevel(logging.WARN)
            return env

        return _thunk
    env = SubprocVecEnv([make_env(i) for i in range(num_env)])
    #env = VecFrameStack(make_atari_env(env_id, 8, seed), 4)

    policy = {'cnn' : CnnPolicy, 'lstm' : LstmPolicy, 'lnlstm' : LnLstmPolicy, 'mlp': MlpPolicy}[policy]
    ppo2.learn(policy=policy, env=env, nsteps=128//2, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=4, log_interval=1,
        ent_coef=.01,
        lr=lambda f : f * 2.5e-4,
        cliprange=lambda f : f * 0.1,
        total_timesteps=int(num_timesteps * 1.1),
        save_interval=save_interval)

def main():
    parser = atari_arg_parser()
    logger.configure()

    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm', 'mlp'], default='cnn')
    parser.add_argument('--save-interval',type=int, default=None)
    parser.add_argument('--load-path',type=str, default='')
    args = parser.parse_args()


    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, save_interval=args.save_interval)

if __name__ == '__main__':
    main()
