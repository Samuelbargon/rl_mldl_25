import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def get_state_values(policy, states, next_states, done, device):
    with torch.no_grad():  #It tells PyTorch not to compute or store gradients for any operations within this block.
        _, state_values = policy(states) #The _ (underscore) in _, state_values indicates that the first output of self.policy (likely the action logits) is being ignored here
        _, next_state_values = policy(next_states)
        state_values = state_values.squeeze(-1)
        next_state_values = next_state_values.squeeze(-1)
        # For terminal states, next_state_value = 0
        next_state_values = next_state_values * (~done.bool().to(device)) # ~done.bool(): it is a not on the boolean value of done. If done is true ~done is false. And if done is True, next_state_value should be 0 (False has as value 0), otherwise it should be the value predicted by the critic network
    return state_values, next_state_values

class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        
        # Learned standard deviation for exploration at training time 
        self.sigma_activation = F.softplus
        init_sigma = 0.5
        self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic neural network for actor-critic algorithm , it takes a state as input
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic = torch.nn.Linear(self.hidden, 1)  # Output a single value(single scalar value) for the state value function, which is its current estimate of the expected return from that state

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic neural network
        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic(x_critic)
        
        
        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        
        # self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3) #Use it for TASK 2
        
        # Separate parameters for actor and critic #TODO da valutare se giusto
        self.optimizer_actor = torch.optim.Adam(
            list(self.policy.fc1_actor.parameters()) +
            list(self.policy.fc2_actor.parameters()) +
            list(self.policy.fc3_actor_mean.parameters()) +
            [self.policy.sigma], lr=1e-3)
        self.optimizer_critic = torch.optim.Adam(
            list(self.policy.fc1_critic.parameters()) +
            list(self.policy.fc2_critic.parameters()) +
            list(self.policy.fc3_critic.parameters()), lr=1e-3)
        

        # self.gamma = 0.99
        # self.states = []
        # self.next_states = []
        # self.action_log_probs = []
        # self.rewards = []
        # self.done = []

        self.gamma = 0.99
        self.lambda_actor = lambda_actor
        self.lambda_critic = lambda_critic
        
        # Eligibility traces for actor and critic
        self.actor_params = (
            list(self.policy.fc1_actor.parameters()) +
            list(self.policy.fc2_actor.parameters()) +
            list(self.policy.fc3_actor_mean.parameters()) +
            [self.policy.sigma]
        )
        self.critic_params = (
            list(self.policy.fc1_critic.parameters()) +
            list(self.policy.fc2_critic.parameters()) +
            list(self.policy.fc3_critic.parameters())
        )
        self.actor_eligibility = [torch.zeros_like(p, device=self.train_device) for p in self.actor_params]
        self.critic_eligibility = [torch.zeros_like(p, device=self.train_device) for p in self.critic_params]
         
             
    def step(self, state, action, reward, next_state, done):
        # Convert to tensors
        state = torch.from_numpy(state).float().to(self.train_device)
        next_state = torch.from_numpy(next_state).float().to(self.train_device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.train_device)
        done = torch.tensor(done, dtype=torch.float32, device=self.train_device)

        # Get value estimates
        _, value = self.policy(state)
        _, next_value = self.policy(next_state)
        value = value.squeeze()
        next_value = next_value.squeeze() * (1.0 - done)  # 0 if terminal

        # TD error
        delta = reward + self.gamma * next_value - value # equivalent to "td_targets " in task 3 One-step Actor–Critic

        # Critic: compute gradients
        self.optimizer_critic.zero_grad()
        value.backward(retain_graph=True)
        critic_grads = [p.grad.clone() for p in self.policy.fc1_critic.parameters()]
        # Update eligibility traces for critic
        self.critic_eligibility = [self.gamma * self.lambda_critic * e + g for e, g in zip(self.critic_eligibility, critic_grads)]
        # Update critic parameters
        for p, e in zip(self.policy.fc1_critic.parameters(), self.critic_eligibility):
            p.data += self.optimizer_critic.param_groups[0]['lr'] * delta * e

        # Actor: compute gradients
        self.optimizer_actor.zero_grad()
        normal_dist, _ = self.policy(state)
        log_prob = normal_dist.log_prob(action).sum()
        log_prob.backward()
        actor_grads = [p.grad.clone() for p in self.policy.parameters()]
        # Update eligibility traces for actor
        self.actor_eligibility = [self.gamma * self.lambda_actor * e + g for e, g in zip(self.actor_eligibility, actor_grads)]
        # Update actor parameters
        for p, e in zip(self.policy.parameters(), self.actor_eligibility):
            p.data += self.optimizer_actor.param_groups[0]['lr'] * delta * e
    
    
    
    
    def reset_traces(self):
        self.actor_eligibility = [torch.zeros_like(p, device=self.train_device) for p in self.actor_params]
        self.critic_eligibility = [torch.zeros_like(p, device=self.train_device) for p in self.critic_params]
        

    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _  = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)


# About Actor–Critic with Eligibility Traces
# It allows the agent to learn from sequences of actions and rewards, rather than just the immediate next state.
# Eligibility traces are a way to assign credit to past states and actions based on their contribution to future rewards.
# They help in learning from sequences of actions and rewards, rather than just the immediate next state.

# Simple analogy:
# One-step Critic: You taste the dish, and if it's bad, you only blame the last ingredient you added. You'll eventually learn, but it will be slow because a bad ingredient added early might not be identified for many "episodes" (meals).
# Critic with Eligibility Traces: You taste the dish, and if it's bad, you blame the last ingredient a lot, the second-to-last ingredient a bit less, and so on. You assign partial blame to all recent ingredients. This allows you to pinpoint the problematic ingredient much faster.