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

def main():
    train_env = Monitor(gym.make('CustomHopper-source-v0'))
    eval_env = Monitor(gym.make('CustomHopper-source-v0'))

    n_cycles = 5  # Number of train-test cycles
    train_steps_per_cycle = 160_000
    n_test_episodes = 10

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
        rewards = []
        for _ in range(n_test_episodes):
            obs = eval_env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                total_reward += reward
            rewards.append(total_reward)
        avg_reward = sum(rewards) / len(rewards)
        cycle_avg_rewards.append(avg_reward)
        print(f"Cycle {cycle+1} - Average test reward: {avg_reward}")

    print("\n=== Recap of Average Rewards per Cycle ===")
    for i, avg in enumerate(cycle_avg_rewards):
        print(f"Cycle {i+1}: {avg}")
    print(f"Final average over all cycles: {sum(cycle_avg_rewards)/len(cycle_avg_rewards)}")

if __name__ == '__main__':
    main()