import torch
from torch.nn import Module, Sequential, Linear, ReLU
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as tf
from ActorCritic import ActorCriticModel, INNAME

ENV="CartPole-v1"
EPS_COLLECT_TRAIN=10000
EPS_COLLECT_TEST=1000
MAX_EPS_COLLECT=10000
PROB_RANDOM=0.01
TRAIN_EPOCHS=20
TRAIN=True
# TRAIN=False
OUTNAME_BASE="bc_models/BehavioralClone_"
EVAL_NAME=""


class BehavioralClone(Module):

    def __init__(self, state_dim=4, action_dim=2, hidden_features=128):
        super().__init__()
        self.hidden = Sequential(
            Linear(state_dim, hidden_features),
            ReLU()
        )
        self.action_head = Linear(hidden_features, action_dim)
        self.name_modifier = f"{state_dim}-{hidden_features}-{action_dim}"
    
    def forward(self, x):
        x = torch.tensor(x)
        return tf.softmax(self.action_head(self.hidden(x)))

class StateActionDataset(Dataset):

    def __init__(self, state_tensor, action_tensor):
        self.states = state_tensor
        self.actions = action_tensor
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


def collect_episode(env, model):

    states = []
    actions = []

    state, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < 500:
        action, was_random = model.randomized_action(state)
        if not was_random:
            states.append(state)
            actions.append(action)
        state, _, done, __, ___ = env.step(action)
    
    return (torch.stack(x) for x in [states, actions])

def collect_dataset(env, model, eps):

    state_list = []
    action_list = []

    for _ in range(eps):
        states, actions = collect_episode(env, model)
        state_list.append(states)
        action_list.append(actions)
    
    state_tensor = torch.concat(state_list)
    action_tensor = torch.concat(action_list)

    return StateActionDataset(state_tensor, action_tensor)

def test_model(model, dataloader):

    correct_sum = 0
    total_length = 0

    for state_batch, action_batch in dataloader:
        infer_batch = model.forward(state_batch)
        correct_sum += torch.where(infer_batch != action_batch, 1, 0).sum().item()
        total_length += len(infer_batch)
    
    return correct_sum / total_length

def clone_agent(env, expert):

    train_ds = collect_dataset(env, expert, EPS_COLLECT_TRAIN)
    test_ds = collect_dataset(env, expert, EPS_COLLECT_TEST)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=True)

    model = BehavioralClone(env.observation_space.shape[0], env.action_space.n)
    model.train()

    opt = Adam(model.parameters(), lr=3e-2)

    for epoch in range(TRAIN_EPOCHS):
        print(f"\n\nStarting epoch {epoch}")

        for i, (state_batch, action_batch) in enumerate(train_dl):
            infer_batch = model.forward(state_batch)
            loss = tf.binary_cross_entropy(action_batch, infer_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i != 0 and i % 100 == 0:
                print(f"Batch {i} loss {loss}")
            
        correct_percent = test_model(model, test_dl) * 100
        print(f"Epoch {epoch} test accuracy {correct_percent}%")
    
    return model


if __name__ == "__main__":
    import gym

    if TRAIN:
        env = gym.make(ENV)

        expert = ActorCriticModel(env.observation_space.shape[0], env.action_space.n)
        expert.load_state_dict(torch.load(INNAME))

        cloned_model = clone_agent(env, expert)

        torch.save(cloned_model.state_dict, f"{OUTNAME_BASE}{cloned_model.name_modifier}.pth")
    else:
        env = gym.make(ENV, render_mode = "human")

        cloned_model = BehavioralClone()
        cloned_model.load_state_dict(torch.load(f"{EVAL_NAME}"))
        cloned_model.eval()

        state, _ = env.reset()
        done = False

        for _ in range(100):

            steps = 0

            while not done:
                action = cloned_model.forward(state)
                state, _, done, __, ___ = env.step(state)

            print(f"died at {steps}")
            


    

