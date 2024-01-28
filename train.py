import numpy as np

import os
import argparse
import yaml
import time

import numpy as np
from algorithms import iavi, iql


def get_arg_parser():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-conf_file", "--conf_file", help="Configuration file path",default="conf.yaml")
    arg_parser.add_argument("-algo", "--algo", help="Algorithm name",default="iavi")

    return arg_parser


def get_conf(conf_path):
    conf_file = open(conf_path)
    conf = yaml.safe_load(conf_file)
    conf_file.close()
    return conf

def action_probs_from_trajectories(trajectories, nS, nA):

    aps = np.zeros((nS, nA))
    for traj in trajectories:
        for (s, a, _, _) in traj:
            aps[s][a] += 1

    aps = aps + 1e-5 # To remove zero elements
    sum_of_state = aps.sum(axis=1).reshape(nS, 1)
    aps = aps / sum_of_state

    return aps


def load_dataset(data_dir):

    trajectories = np.load(os.path.join(data_dir, "trajectories.npy"))
    transition_probabilities = np.load(os.path.join(data_dir, "transition_probabilities.npy"))
    ground_reward = np.load(os.path.join(data_dir, "ground_reward.npy"))
    feature_matrix = np.load(os.path.join(data_dir, "feature_matrix.npy"))

    (nS, nA, _) = transition_probabilities.shape
    action_probabilities = action_probs_from_trajectories(trajectories, nS, nA)

    return feature_matrix, transition_probabilities, action_probabilities, ground_reward, trajectories


def save_results(q, evd_list, boltzman_distribution, algo, result_folder):
    run_id = 1
    while True:
        save_folder = os.path.join(result_folder, algo, f'run_{run_id}')
        if not os.path.exists(save_folder):
            break
        run_id +=1
    os.makedirs(save_folder, exist_ok = True)
    np.save(os.path.join(save_folder, 'q.npy'), q)
    np.save(os.path.join(save_folder, 'evd_list.npy'), evd_list)
    np.save(os.path.join(save_folder, 'boltzman_distribution.npy'), boltzman_distribution)


if __name__=="__main__":

    # Arguiments
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    conf_file = args.conf_file
    algo = args.algo

    # Configuration file
    conf = get_conf(conf_file)

    # directories
    data_dir = conf['dataset']['dataset_path']
    result_folder = conf['train']['result_folder']

    # Creating folders
    os.makedirs(result_folder, exist_ok = True)

    # Loading the dataset
    feature_matrix, transition_probabilities, action_probabilities, ground_reward, trajectories = load_dataset(data_dir)
    print(f"Dataset found with trajectories shape of: {trajectories.shape}")

    # Training
    print(f"Started training {algo} algorithm")
    if algo == 'iavi':
        t1 = time.time()
        q, evd_list, boltzman_distribution = iavi(feature_matrix, transition_probabilities, action_probabilities, trajectories, conf)
        t2 = time.time()
        print("time taken for iavi:", t2-t1)
    elif algo == 'iql':
        t1 = time.time()
        q, evd_list, boltzman_distribution = iql(trajectories, conf)
        t2 = time.time()
        print("time taken for iql:", t2-t1)

    else:
        print("Unknown algorithm")

    save_results(q, evd_list, boltzman_distribution, algo, result_folder)