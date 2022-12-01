## MDP reference - simple examples of agents in a Markov Decision Process

For content, refer to *mdp_examples.ipynb*.

### Installation:

Preferably set up a virtual environment and run

```{bash}
pip install -r requirements.txt
pip install torch --extra-index-url https://download.pytorch.org/whl/cu116 # if using a specific version of cuda
```

### Execution:

To train or evaluate a model individually, run

```{bash}
python QMatrix.py
# or
python ActorCritic.py
# or
python BehaviouralClone.py
```

Configuration parameters are simply set at the top of every script.

To run the jupyter notebook:

```{bash}
jupyter notebook
```