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
    train_env = Monitor(gym.make('CustomHopper-source-v0'))  # Use a Monitor wrapper to log training episodes
    
    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

    eval_env = Monitor(gym.make('CustomHopper-source-v0'))  # Use a separate env for evaluation

    model = PPO("MlpPolicy", train_env, verbose=1)
    # model = PPO.load("./best_model/best_model", env=train_env) # Uncomment to load a pre-trained model


    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=5_000, save_path='./checkpoints/',
                                             name_prefix='ppo_hopper')
    eval_callback = EvalCallback(eval_env, best_model_save_path='./best_model/',
                                 log_path='./logs/', eval_freq=3_000,
                                 deterministic=True, render=False)

    # Train the agent with callbacks
    model.learn(total_timesteps=40_000, callback=[checkpoint_callback, eval_callback])

    # After training, load and test the best model:
    model = PPO.load("./best_model/best_model")

    n_episodes = 10
    rewards = []
    for _ in range(n_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    print("Average test reward:", sum(rewards)/len(rewards))

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

if __name__ == '__main__':
    main()