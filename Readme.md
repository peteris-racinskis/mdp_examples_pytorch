## MDP reference - simple examples of agents in a Markov Decision Process

### QMatrix - an implementation of the classic Bellman equation through a fully enumerated reward matrix

The simplest, "brute force" approach to reinforcement learning. Construct a tensor the dimensions of which correspond to discretized versions of the states and discrete action set. Every value corresponds to the expected reward. Output an action based on maximum expected reward at any given state. Iteratively optimize the values in the tensor by repeatedly sampling the environment and using Bellman's update rule. 

### ActorCritic - a simple deep learning approach using the Advantage Actor-Critic architecture

### BehavioralClone - an imitation learning approach