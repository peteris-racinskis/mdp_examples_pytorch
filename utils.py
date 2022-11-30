import gym

def visualize_model(model, envname="CartPole-v1", max_steps = 2000, episodes = 100):
    env = gym.make(envname, render_mode = "human")

    for ep in range(episodes):    
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = model.get_action(state)
            state, _, done, __, __ = env.step(action)
            steps += 1
        print(f"episode {ep} died at {steps}")

    