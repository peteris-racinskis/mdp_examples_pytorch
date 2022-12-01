import gym

def visualize_model(model, envname="CartPole-v1", max_steps = 300, episodes = 100, notebook = False):
    if notebook:
        import matplotlib.pyplot as plt
        from IPython.display import clear_output
        episodes = 1
        env = gym.make(envname, render_mode = "rgb_array")
    else:
        env = gym.make(envname, render_mode = "human")

    for ep in range(episodes):    
        state, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = model.get_action(state)
            state, _, done, __, __ = env.step(action)
            steps += 1
            if notebook:
                clear_output(wait=True)
                plt.imshow(env.render())
                plt.show()
        print(f"episode {ep} died at {steps}")

    