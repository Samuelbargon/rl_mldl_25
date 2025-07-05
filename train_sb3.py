"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from dropo import Dropo
import numpy as np
import glob
import os

def main():
    # 1. Prepare your sim_env and dataset for DROPO
    sim_env = gym.make('CustomHopper-source-v0')
    
    # Load your offline dataset here 
    dataset_dir = "datasets/hopper10000" 

    observations = np.load(glob.glob(os.path.join(dataset_dir, '*_observations.npy'))[0])
    next_observations = np.load(glob.glob(os.path.join(dataset_dir, '*_nextobservations.npy'))[0])
    actions = np.load(glob.glob(os.path.join(dataset_dir, '*_actions.npy'))[0])
    terminals = np.load(glob.glob(os.path.join(dataset_dir, '*_terminals.npy'))[0])
    T = {'observations': observations, 'next_observations': next_observations, 'actions': actions, 'terminals': terminals}

    # 2. Run DROPO optimization
    dropo = Dropo(sim_env=sim_env, t_length=1, scaling=1.0, seed=42, sync_parall=True)
    dropo.set_offline_dataset(T, n=10, sparse_mode=False)
    best_bounds, best_score, elapsed, learned_epsilon = dropo.optimize_dynamics_distribution(
        opt='adam', budget=1000, additive_variance=False, epsilon=1e-5, sample_size=100, now=10,
        learn_epsilon=False, normalize=True, logstdevs=False
    )
    means = dropo.get_means(best_bounds)
    stds = dropo.get_stdevs(best_bounds)

    # 3. Pass means and stds to your environments
    train_env = Monitor(CustomHopper(domain='source', mass_means=means, mass_stds=stds))
    eval_env = Monitor(CustomHopper(domain='source', mass_means=means, mass_stds=stds))

    n_cycles = 1  # Number of train-test cycles
    train_steps_per_cycle = 1e6 #2e6
    n_test_episodes = 50

    cycle_avg_rewards = []

    for cycle in range(n_cycles):
        print(f"\n=== Training Cycle {cycle+1} ===")

        # Load previous best model if exists, else create new
        try:
            model = PPO.load("./best_model/best_model", env=train_env)
            print("Loaded previous best model.")
        except FileNotFoundError:
            model = PPO("MlpPolicy", train_env, verbose=1)
            print("Created new model.")

        # Callbacks
        checkpoint_callback = CheckpointCallback(save_freq=20_000, save_path='./checkpoints/', name_prefix='ppo_hopper')
        eval_callback = EvalCallback(eval_env, best_model_save_path='./best_model/', log_path='./logs/', eval_freq=16_000, deterministic=True, render=False)

        # Train
        model.learn(total_timesteps=train_steps_per_cycle, callback=[checkpoint_callback, eval_callback])

        # Test the best model
        model = PPO.load("./best_model/best_model", env=eval_env)
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=n_test_episodes, deterministic=True
        )
        print(f"Cycle {cycle+1} - Average test reward: {mean_reward} +/- {std_reward}")
        cycle_avg_rewards.append(mean_reward)
        

    print("\n=== Recap of Average Rewards per Cycle ===")
    for i, avg in enumerate(cycle_avg_rewards):
        print(f"Cycle {i+1}: {avg}")
    print(f"Final average over all cycles: {sum(cycle_avg_rewards)/len(cycle_avg_rewards)}")

if __name__ == '__main__':
    main()