from argparse import ArgumentParser
import time
import os
import pandas as pd
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from environment import TradeEnvironment
from eval_fn import init, agent_predict


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-m', '--model_path', required=False, default='model/', help='Directory to save the file')
    parser.add_argument('-f', '--file_path', required=True, help='Path to the data')
    parser.add_argument('-n', '--num_steps', default=1000, type=int)


    args = parser.parse_args()
    model_path = args.model_path
    file_path = args.file_path
    df = pd.read_csv(file_path)

    init(model_path)
    env = TradeEnvironment(df)

    log_interval = 100
    rewards = 0
    port = []

    is_done = False
    eps_step = 0

    rewards = 0

    time_step = env.reset()
  

    for i in range(args.num_steps):
        action = agent_predict(time_step)
        time_step, _, is_done, _ = env.step(action)
        eps_step += 1
        rewards += time_step[1]
        port.append(env.current_value)

        if is_done:
            break
        if i % log_interval == 0:
                print('step = {0}: \t value={1}'.format(i, env.current_value))

    avg_return = rewards / eps_step    
    
    print(f'Eval complete. Steps = {i+1}, value = {env.current_value}, avg reward = {avg_return}')
    with open('eval_port.pkl', 'wb') as f:
         pickle.dump(port, f)
