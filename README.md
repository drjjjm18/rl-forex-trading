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
    - Get some better basic performance metrics to use in the evaluate_agent.py:
        - max drawdown
        
## How to run

*Recommended* 
Consider creating a virtual environment:
```
# create python venv
python -m venv trade_env
# activate venv - note this varies between windows, bash and mac
trade_env\Scripts\activate # linux/bash: source trade_env/Scripts/activate # mac: source trade_env/bin/activate
```

### Install dependencies:
```
python -m pip install -r requirements.txt
```

### download some data:
`data.py` handles downloading datasets to train agents with, using yahoo finance. The `config/data_config.yaml` contains the config for this code:
- tickers: a list of valid yahoo tickers to download, e.g. `'GBPUSD=X'`
- interval: the interval for price data, e.g. `'60m'` for hourly data
- period: the period to get data for, e.g. `'730d` for 730 days (this is the max for hourly data (and any data in general))
- moving_averages: a list of valid moving averages to calculate for the `Close` value, e.g. `'1D'` for a daily moving average
- calculate_rsi: whether to calculate RSI

# Post processing options
drop_close_equals_0: False
normalise_columns: True
drop_volume: True

```
python data.py -c <path to data_config.yaml> -s <dir to save output files>
```

train an agent on some data:
```
python train.py -s <dir to save models to> -f <path to training data> -c <path to train config yaml>
```

evaluate an agent:
```
python evaluated_agent.py -m <dir model is saved in> -f <path to evaluation data> -n <number of evaluation steps>
```
