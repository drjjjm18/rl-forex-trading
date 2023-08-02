# rl-forex-trading

An environment to train RL trading models initially targeting forex

## To Do
- Environment:
    - Why am I buying using the normalised close value? Should use the actual price, but give the agent the norm value
- Datasets: 
    - set appropriate rolling average metrics for different intervals in data.py
    - explore other signals
- Training:
    - Configure train.py to loop through multiple datasets in training run given a list in config
- Hyperparams:
    - enable tweaking of multiple hyper parameters over different training runs