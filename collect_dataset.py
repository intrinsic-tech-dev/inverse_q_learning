import os
import argparse
import yaml

import numpy as np
from mdp.objectworld import Objectworld
from mdp.value_iteration import find_policy


def get_arg_parser():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-conf_file", "--conf_file", help="Configuration file path",default="conf.yaml")

    return arg_parser


def get_conf(conf_path):
    conf_file = open(conf_path)
    conf = yaml.safe_load(conf_file)
    conf_file.close()
    return conf



def generate_demos(env, num_traj, traj_length):

    ground_rewards = np.array([env.reward(s) for s in range(env.n_states)])
    transition_probabilities = env.transition_probability
    policy = find_policy(env.n_states, env.n_actions, transition_probabilities, ground_rewards, env.discount, stochastic=False)
    
    trajectories = env.generate_trajectories(num_traj, traj_length, lambda s: policy[s])

    return trajectories, transition_probabilities, ground_rewards


def create_dataset(env, conf):

    traj_length = conf['dataset']['traj_length']
    num_traj = conf['dataset']['num_traj']

    feature_matrix = env.feature_matrix(discrete=False)

    trajectories, transition_probabilities, ground_rewards= generate_demos(env, num_traj, traj_length)

    # Saving collected data
    np.save(os.path.join(data_dir, "trajectories.npy"), trajectories)
    np.save(os.path.join(data_dir, "transition_probabilities.npy"), transition_probabilities)
    np.save(os.path.join(data_dir, "ground_reward.npy"), ground_rewards)
    np.save(os.path.join(data_dir, "feature_matrix.npy"), feature_matrix)


if __name__=="__main__":

    # Arguiments
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()
    # Configuration file
    conf_file = args.conf_file
    conf = get_conf(conf_file)

    grid_size = conf['env']['grid_size']
    num_obj = conf['env']['num_obj']
    num_colours = conf['env']['num_colours']
    wind = conf['env']['wind']
    gamma = conf['env']['gamma']

    # Creating folders
    data_dir = conf['dataset']['dataset_path']
    os.makedirs(data_dir, exist_ok = True)

    # Creating environment
    env = Objectworld(grid_size, num_obj, num_colours, wind, gamma)

    # Creating dataset
    create_dataset(env, conf)

    print("Data collection is done!")
