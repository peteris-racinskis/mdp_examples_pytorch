import torch
import numpy as np
import torch.nn.functional as tf
import pickle
from os.path import exists
from torch.nn import Module, Sequential, Linear, ReLU
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from ActorCritic import ActorCriticModel, INNAME
from utils import visualize_model


ENV="CartPole-v1"
EPS_COLLECT_TRAIN=1000
EPS_COLLECT_TEST=100
MAX_EPS_COLLECT=10000
PROB_RANDOM=0.01
TRAIN_EPOCHS=2
# TRAIN=True
TRAIN=False
TRAIN_DATASET_FNAME=f"expert_datasets/train_{EPS_COLLECT_TRAIN}_expl_{PROB_RANDOM}"
TEST_DATASET_FNAME=f"expert_datasets/test_{EPS_COLLECT_TEST}_expl_{PROB_RANDOM}"
OUTNAME_BASE=f"bc_models/BehavioralClone_ds_{EPS_COLLECT_TRAIN}-{PROB_RANDOM}_epochs_{TRAIN_EPOCHS}_layers_"
EVAL_NAME="bc_models/BehavioralClone_4-128-2.pth"


class BehavioralClone(Module):

    def __init__(self, device, state_dim=4, action_dim=2, hidden_features=128):
        super().__init__()
        self.device = device
        self.hidden = Sequential(
            Linear(state_dim, hidden_features),
            ReLU(),
        )
        self.action_head = Linear(hidden_features, action_dim)
        self.name_modifier = f"{state_dim}-{hidden_features}-{action_dim}"
        self.to(self.device)
    
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        return self.action_head(self.hidden(x.to(self.device)))
    
    def get_action(self, state):
        return self.forward(state).argmax().item()

class StateActionDataset(Dataset):

    def __init__(self, state_tensor, action_tensor):
        self.states = state_tensor
        self.actions = action_tensor
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

    def save_to_file(self, name):
        with open(name, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_from_file(cls, name):
        with open(name, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, cls):
            return obj
        return None

def collect_episode(env, model, exploration_rate=PROB_RANDOM):

    states = []
    actions = []

    state, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < 1000:
        steps += 1
        action, was_random = model.randomized_action(env.action_space, torch.tensor(state), exploration_rate)
        if not was_random:
            states.append(state)
            actions.append(action)
        state, _, done, __, ___ = env.step(action)
    
    return torch.tensor(np.array(states)), torch.tensor(actions)

def collect_dataset(env, model, eps, name):

    if exists(name):
        print(f"Loading dataset from file {name}")
        return StateActionDataset.load_from_file(name)

    print(f"Collecting new dataset for {name}")

    state_list = []
    action_list = []

    for ep in range(eps):
        states, actions = collect_episode(env, model)
        state_list.append(states)
        action_list.append(actions)

        if ep % 50 == 0:
            print(f"Processed episode {ep} of {eps}")
    
    state_tensor = torch.concat(state_list)
    action_tensor = torch.concat(action_list)
    
    ds = StateActionDataset(state_tensor, action_tensor)
    print(f"Saving dataset to {name}")
    ds.save_to_file(name)

    return ds

def test_model(model, dataloader, device):

    correct_sum = torch.tensor(0).to(device)
    total_length = 0

    for state_batch, action_batch in dataloader:
        infer_batch = model.forward(state_batch)
        correct_sum += torch.where(infer_batch.argmax(dim=-1) == action_batch.to(device), 1, 0).sum()
        total_length += len(infer_batch)
    
    return (correct_sum / total_length).item()

def clone_agent(env, expert):

    train_ds = collect_dataset(env, expert, EPS_COLLECT_TRAIN, TRAIN_DATASET_FNAME)
    test_ds = collect_dataset(env, expert, EPS_COLLECT_TEST, TEST_DATASET_FNAME)

    train_dl = DataLoader(train_ds, batch_size=2048, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=True)

    cpu = torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else cpu
    model = BehavioralClone(device, env.observation_space.shape[0], env.action_space.n)
    model.train()

    opt = Adam(model.parameters(), lr=1e-2)
    scd = MultiStepLR(opt, [1,2,5,10])
    losses = []
    for epoch in range(TRAIN_EPOCHS):
        print(f"\n\nStarting epoch {epoch} with lr {scd.get_last_lr()}")

        for i, (state_batch, action_batch) in enumerate(train_dl):
            infer_batch = model.forward(state_batch)
            loss = tf.cross_entropy(infer_batch, action_batch.to(device), reduction='sum')
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss)
            if i != 0 and i % 2000 == 0:
                print(f"Batch {i} mean loss {torch.stack(losses).mean()}")
                losses = []
        scd.step()
        correct_percent_train = test_model(model, train_dl, device) * 100
        correct_percent = test_model(model, test_dl, device) * 100
        print(f"Epoch {epoch} test accuracy {correct_percent}% train accuracy {correct_percent_train}%")
    
    return model


if __name__ == "__main__":
    import gym

    if TRAIN:
        env = gym.make(ENV)

        expert = ActorCriticModel(env.observation_space.shape[0], env.action_space.n)
        expert.load_state_dict(torch.load(INNAME))

        cloned_model = clone_agent(env, expert)

        torch.save(cloned_model.state_dict(), f"{OUTNAME_BASE}{cloned_model.name_modifier}.pth")
    else:
        env = gym.make(ENV, render_mode = "human")

        cloned_model = BehavioralClone(torch.device('cuda'))
        cloned_model.load_state_dict(torch.load(f"{EVAL_NAME}"))
        cloned_model.eval()
        
        visualize_model(cloned_model)
            


    

