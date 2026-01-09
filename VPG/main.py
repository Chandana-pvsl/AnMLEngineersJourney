import torch
import torch.nn as nn
import gym
import time
from tqdm import tqdm
    
# A dummy Policy network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, n_actions, hidden_sizes=(128,)):
        super().__init__()
        layers = []
        self.input_dim = input_dim
        hidden_input_dim = input_dim
        for hidden_output_dim in hidden_sizes:
            layers += [nn.Linear(hidden_input_dim, hidden_output_dim), nn.ReLU()]
            hidden_input_dim = hidden_output_dim
        layers += [nn.Linear(hidden_input_dim, n_actions), nn.Softmax(dim=-1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Return computation
def compute_intermediate_returns(reward_per_step_list, discount_factor):
    return_per_step = []
    G = 0
    for reward in reward_per_step_list[::-1]:
        G = reward + G * discount_factor
        return_per_step.append(G)
    return return_per_step[::-1]


# Training Loop (REINFORCE + Mini-batch)
env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy_net = PolicyNetwork(input_dim=4, n_actions=2)

lr = 1e-2
optimizer = torch.optim.AdamW(policy_net.parameters(), lr = lr)

discount_factor = 0.99
batch_size = 300  # number of timesteps collected before each update

def collect_trajectories(policy_net, batch_size):
        batch_states = []
        batch_actions = []
        batch_returns = []
        batch_logprobs = []
        total_returns = []
        steps_collected = 0
        max_steps = 500

        # Collect trajectories until we reach batch_size steps
        while steps_collected < batch_size:

            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_logprobs = []

            state, _ = env.reset()
            done = False

            # while not done:
            for _ in range(max_steps):
                s = torch.tensor(state, dtype=torch.float32)
                probs = policy_net(s)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()

                next_state, reward, term, trunc, _ = env.step(action.item())

                episode_states.append(s)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_logprobs.append(dist.log_prob(action))

                state = next_state
                done = term or trunc
                if done:
                    break
            
            total_returns.append(sum(episode_rewards))
            # Compute returns for this trajectory
            returns = compute_intermediate_returns(episode_rewards, discount_factor)

            # Add to batch
            batch_states.extend(episode_states)
            batch_actions.extend(episode_actions)
            batch_returns.extend(returns)
            batch_logprobs.extend(episode_logprobs)

            steps_collected += len(episode_rewards)
        return batch_returns, batch_logprobs, total_returns


num_steps = 200
losses = []
start_time = time.time()
returns = []
for iteration in tqdm(range(num_steps)):
    # Convert batch to tensors
    batch_returns, batch_logprobs, total_returns = collect_trajectories(policy_net, batch_size)
    returns.extend(total_returns)
    batch_returns = torch.tensor(batch_returns, dtype=torch.float32)
    batch_logprobs = torch.stack(batch_logprobs)

    # Normalize returns (optional but stabilizes training)
    batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-8)

    # Compute policy loss
    loss = -(batch_logprobs * batch_returns).mean()

    optimizer.zero_grad()
    loss.backward()
    losses.append(loss.item())
    optimizer.step()

print("Time taken ", time.time()-start_time)
