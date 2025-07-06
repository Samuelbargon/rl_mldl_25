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
from argparse import ArgumentParser

def parse_args_dropo():
	parser = ArgumentParser()

	# RECOMMENDED FLAGS
	parser.add_argument("--additive_variance", "-av", default=False, action='store_true', help="RECOMMENDED. Add value --epsilon to the diagonal of the cov_matrix to regularize the next-state distribution inference (default: False)")
	parser.add_argument("--normalize", default=False, action='store_true', help="RECOMMENDED. Normalize dynamics search space to [0,4] as a regularization for CMA-ES.")
	parser.add_argument("--logstdevs", default=False, action='store_true', help="RECOMMENDED. Optimize stdevs in log space. (Default: false)")

	# Hyperparameters
	parser.add_argument("--n-trajectories", "-n", type=int, default=None, help="Number of target trajectories for running DROPO. if --sparse-mode is selected, this parameter refers to the number of single TRANSITIONS instead.")
	parser.add_argument("-l", type=int, default=1, help="Lambda hyperparameter.")
	parser.add_argument("--epsilon", "-eps", type=float, default=1e-3, help="RECOMMENDED. Epsilon hyperparameter. Valid only when --additive_variance is set (default: 1e-3)")
	parser.add_argument('--env', default='CustomHopper-source-v0', type=str, help='Gym-registered environment.')
	parser.add_argument("--output-dir", type=str, default='output', help="Output directory for results")
	parser.add_argument("--scaling", default=False, action='store_true', help="Scaling each state dimension (Default: False)")
	parser.add_argument("--now", type=int, default=1, help="Number of workers for parallelization (Default: 1 => no parallelization)")
	parser.add_argument("--seed", type=int, default=0, help="Set a specific seed")
	parser.add_argument("--opt", type=str, default='cma', help="nevergrad optimizer [oneplusone, bayesian, twopointsde, pso, tbpsa, random, meta, cma (default)]")
	parser.add_argument("--no-output", "-no", default=False, action='store_true', help="If set, DO NOT save the output of optimization problem to --output-dir")
	parser.add_argument("--budget", type=int, default=1000, help="Number of evaluations in the opt. problem (Default: 1000)")
	parser.add_argument("--sample_size", "-ss", type=int, default=100, help="Number of observations to sample to estimate the next-state distribution (Default: 100)")
	parser.add_argument("--dataset", type=str, default='datasets/hopper10000', help="Specify directory containing a custom dataset to use.")
	parser.add_argument("--sparse-mode", "-sm", default=False, action='store_true', help="Whether to use sparse transitions for running DROPO than reproducing full episodes. (Default: False)")
	parser.add_argument("--no-sync-parall", default=False, action='store_true', help="If set, avoids asking `popsize` values before telling their values during parallelization.")
	
	# Not officially supported
	parser.add_argument("--learn-epsilon", default=False, action='store_true', help="(Not recommended) Whether to learn the hyperparameter --epsilon (default: False)")

	args = parser.parse_args()

	return args

def main():
    # 1. Prepare your sim_env and dataset for DROPO
    args = parse_args_dropo()
    
    # sim_env = gym.make('CustomHopper-source-v0')    
    # sim_env = Monitor(gym.make('CustomHopper-source-v0'))
    sim_env = gym.make(args.env)
    
    # Load your offline dataset here 
    # dataset_dir = "datasets/hopper10000" 

    observations = np.load(glob.glob(os.path.join(args.dataset, '*_observations.npy'))[0])
    next_observations = np.load(glob.glob(os.path.join(args.dataset, '*_nextobservations.npy'))[0])
    actions = np.load(glob.glob(os.path.join(args.dataset, '*_actions.npy'))[0])
    terminals = np.load(glob.glob(os.path.join(args.dataset, '*_terminals.npy'))[0])
    T = {'observations': observations, 'next_observations': next_observations, 'actions': actions, 'terminals': terminals}

    # 2. Run DROPO optimization
    # dropo = Dropo(sim_env=sim_env, t_length=1, scaling=True, seed=42, sync_parall=True)
    dropo = Dropo(sim_env=sim_env,
				  t_length=args.l,
				  scaling=args.scaling,
				  seed=args.seed,
				  sync_parall=(not args.no_sync_parall))
    
    # dropo.set_offline_dataset(T, n=10, sparse_mode=False)
    dropo.set_offline_dataset(T, n=args.n_trajectories, sparse_mode=args.sparse_mode)

    # best_bounds, best_score, elapsed, learned_epsilon = dropo.optimize_dynamics_distribution(
    #     opt='cma', budget=1000, additive_variance=False, epsilon=1e-5, sample_size=100, now=10,
    #     learn_epsilon=False, normalize=True, logstdevs=False
    # )
    best_bounds, best_score, elapsed, learned_epsilon = dropo.optimize_dynamics_distribution(
        opt=args.opt,
        budget=args.budget,
        additive_variance=args.additive_variance,
        epsilon=args.epsilon,
        sample_size=args.sample_size,
        now=args.now,
        learn_epsilon=args.learn_epsilon,
        normalize=args.normalize,
        logstdevs=args.logstdevs
    )
    means = dropo.get_means(best_bounds)
    stds = dropo.get_stdevs(best_bounds)
    
    print('Best means and st.devs:\n---------------')
    print(dropo.pretty_print_bounds(best_bounds),'\n')

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