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
- Evaluation:
    - Get some basic performance metrics calculated:
        - max drawdown
        
## How to run

get data:
```
python data.py -c <path to config yaml> -s <dir to save output files>
```

train an agent:
```
python train.py -s <dir to save models to> -f <path to training data> -c <path to train config yaml>
```

evaluate an agent:
```
python evaluated_agent.py -m <dir model is saved in> -f <path to evaluation data> -n <number of evaluation steps>
```