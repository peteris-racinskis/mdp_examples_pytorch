import torch
from torch.nn import Module, Linear, Sequential, ReLU, Softmax
from torch.distributions import Categorical
from torch.optim.sgd import SGD
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np

import lovely_tensors as lt

lt.monkey_patch()

ENV="CartPole-v1"
# TRAIN=False
TRAIN=True
OUTNAME="actor_critic_models/ActorCritic_adam"
INNAME="actor_critic_models/ActorCritic_ep10000.pth"

SEED=543

torch.manual_seed(SEED)

class ActorCriticResult():
    
    def __init__(self, logits, probs, value):
        self.logits = logits
        self.probs = probs
        self.value = value

    def sample(self) -> torch.Tensor:
        distribution = Categorical(self.probs)
        s = distribution.sample()
        return s, distribution.log_prob(s)


class ActorCriticModel(Module):

    EP_LIMIT=200

    def __init__(self, state_dim=4, action_dim=2, critic_dim=1, hidden_dim=128, device=torch.device('cpu')):
        super().__init__()
        self.hidden = Sequential(
            Linear(state_dim, hidden_dim),
            ReLU(),
            # Linear(hidden_dim, hidden_dim),
            # ReLU()
        )
        self.actor_head = Linear(hidden_dim, action_dim)
        self.critic_head = Linear(hidden_dim, critic_dim)
        self.softmax = Softmax(dim=-1)
        self.huber_loss = torch.nn.HuberLoss(reduction="mean")
        self._device = device
        self._cpu = torch.device('cpu')
        self.to(device)
        self.losses = []
        self.died_ats = []
    
    def forward(self, x) -> ActorCriticResult:
        self.last_input = x
        hidden = self.hidden(x)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        probs = self.softmax(logits)
        return ActorCriticResult(logits, probs, value)
    
    def run_episode(self, env, render=False):

        obs, _ = env.reset()
        done = False
        step = 0

        action_log_probs = []
        actual_rewards = []
        estimated_rewards = []

        while not done and step < self.EP_LIMIT:
            step += 1

            result = self.forward(torch.tensor(obs).to(self._device))
            action, log_prob = result.sample()

            obs, actual_reward, done, _, __ = env.step(action.item())

            actual_rewards.append(torch.tensor(actual_reward))
            estimated_rewards.append(result.value.squeeze())
            action_log_probs.append(log_prob)

            if render:
                env.render()

        return [
            torch.stack(x).to(self._device) for x in (action_log_probs, estimated_rewards, actual_rewards)
            ], step

    def actor_critic_loss(self, log_probs, estimated_rewards, actual_rewards, opt):
        advantage = actual_rewards - estimated_rewards
        actor_loss = torch.sum(advantage * -log_probs)
        # critic_loss = self.huber_loss(estimated_rewards, actual_rewards)
        critic_loss = F.smooth_l1_loss(estimated_rewards, actual_rewards,reduction="sum")
        opt.zero_grad()
        return actor_loss + critic_loss

    def train_episode(self, env, opt: torch.optim.Optimizer, ep, gamma=0.99):
        
        
        (action_probs, est_rewards, act_rewards), died_at = self.run_episode(env)
        cumulative_rewards = torch.zeros_like(act_rewards)

        accumulator = torch.tensor(0)
        for i, r in enumerate(act_rewards.flip([0])):
            accumulator = gamma * accumulator + r
            cumulative_rewards[i] = accumulator

        # cumulative_rewards = torch.nn.functional.normalize(cumulative_rewards.flip([0]), dim=0)
        # est_rewards = torch.nn.functional.normalize(est_rewards, dim=0)
        cumulative_rewards = (cumulative_rewards - cumulative_rewards.mean()) / (cumulative_rewards.std() + np.finfo(np.float32).eps.item())
        
        # opt.zero_grad()
        loss = self.actor_critic_loss(action_probs, est_rewards, cumulative_rewards.flip([0]), opt)
        loss.backward()
        
        opt.step()

        self.losses.append(loss.detach().to(self._cpu).numpy())
        self.died_ats.append(died_at)

        if ep != 0 and ep % 50 == 0:
            avg_died_at = np.mean(self.died_ats)
            avg_loss = np.mean(self.losses)
            print(f"Episode {ep} died at {avg_died_at} loss {avg_loss}")


def train_model(model, env, outname=OUTNAME):
    model.train()
    opt = Adam(model.parameters(), lr=3e-2)
    # opt = SGD(model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.9)
    for ep in range(100000):
        model.train_episode(env, opt, ep)
        if ep % 10000 == 0:
            print(f"Saving model at epoch {ep}...")
            torch.save(model.state_dict(), f"{outname}_ep{ep}.pth")

def evaluate_model(model: ActorCriticModel, env, inname=INNAME):
    model.eval()
    model.load_state_dict(torch.load(inname))
    for ep in range(1000):
        _, died_at = model.run_episode(env, True)
        print(f"Died at {died_at}")

    

if __name__ == "__main__":
    import gym
    env = gym.make(ENV)
    env.reset(seed=SEED)
    # dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dev = torch.device('cpu')
    model = ActorCriticModel(
        env.observation_space.shape[0],
        env.action_space.n,
        device=dev
    )

    if TRAIN:
        train_model(model, env)
    else:
        env = gym.make(ENV, render_mode="human")
        env.reset(seed=SEED)
        evaluate_model(model, env)