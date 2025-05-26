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
        

        self.gamma = 0.99
        self.states = []
        self.next_states = []
        self.action_log_probs = []
        self.rewards = []
        self.done = []


    def update_policy(self):
        action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        done = torch.Tensor(self.done).to(self.train_device)

        self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []


        # #
        # # TASK 2: REINFORCE (Vanilla Policy Gradient)
        # #   - compute discounted returns, rewards are usually discounted over time, meaning immediate rewards are valued more than future rewards.
        # returns = discount_rewards(rewards, self.gamma) # For TASK 2
        # # Baseline, does not change the final policy, but can help with variance reduction which help us to achive the final policy faster, better performance. It can be any constant value, or even any function, as long as it does not depend on the actions taken.
        # baseline = 0 # For TASK 2.a
        # baseline = 20 # For TASK 2.b
        
        # #   - compute policy gradient loss function given actions and returns
        # policy_loss = - (action_log_probs * (returns - baseline)).mean() #sum() or mean()? # the expression -(action_log_probs * returns).sum() is directly the negative of the REINFORCE objective for a sampled trajectory. When you minimize this, you are performing gradient ascent on the actual objective.
        
        # #   - compute gradients and step the optimizer
        # # Backpropagation (we are minimizing the negative log likelihood of the actions taken, weighted by the returns)
        # self.optimizer.zero_grad()      # 1. Clear old gradients
        # policy_loss.backward()          # 2. Compute new gradients (backprop)
        # self.optimizer.step()           # 3. Update the policy parameters
        


        #
        # TASK 3: Actor Critic, To assist the policy update, by reducing gradient variance. In other words, to provide a baseline for the policy gradient.
        # Critic is when the state-value function is used to avaluate actions
        # Once the critic network has learned to provide reasonable state value estimates, these estimates are used as the baseline in the policy gradient update.
        
        # Compute state values and next state values using the critic, the state values (which is used as baseline) are used to compute the advantage terms for the policy update.
        # The advantage term estimate tells the actor whether a particular action in a particular state performed better or worse than expected from that state, relative to the average performance from that state.
        # The advantage is the difference between what actually happened (or the bootstrapped target) and what was expected (the baseline).
        state_values, next_state_values = get_state_values(self.policy, states, next_states, done, self.train_device)


        #   - compute boostrapped discounted return estimates (TD targets = Temporal Difference targets which is the target used in TD learning methods, like the Actor Critic one)  
        td_targets = rewards + self.gamma * next_state_values # equivalent to "returns" in task 2
        
        #   - compute advantage terms
        advantages = td_targets - state_values # state_values is used as baseline
        
        #   - compute actor loss and critic loss
        actor_loss = - (action_log_probs * advantages.detach()).mean() # equivalent to "policy_loss" in task 2
        
        _, state_values_for_update = self.policy(states)
        state_values_for_update = state_values_for_update.squeeze(-1)
        critic_loss = torch.nn.functional.mse_loss(state_values_for_update, td_targets.detach())

        
        #   - compute gradients and step the optimizer
        # Update actor (policy network)
        self.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)  # retain_graph=True tells PyTorch: "Don't free the computation graph after this backward pass, because I'm going to need it again for another backward pass (the critic's loss) before the next forward pass."
        self.optimizer.step()

        # Update critic (value network)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        

        return        


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist = self.policy(x)

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

