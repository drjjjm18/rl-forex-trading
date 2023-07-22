from argparse import ArgumentParser
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from environment import TradeEnvironment
from user_eval_fn import init, agent_predict


def median(arr):
    arr = sorted(arr)
    return arr[len(arr)//2]

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('modelpath')
    parser.add_argument('-n', '--num-episodes', default=100, type=int)
    args = parser.parse_args()

    init(args.modelpath)
    env = TradeEnvironment()

    scores = []
    pred_times = []
    step_times = []
    eps_times = []
    eps_lengths = []

    start_time = time.time()

    for i in range(args.num_episodes):
        obs = env.reset()
        is_done = False
        eps_steps = 0
        
        start_eps = time.time()
        while not is_done:
            start_pred = time.time()
            action = agent_predict(obs)
            pred_times.append(time.time() - start_pred)
            start_step = time.time()
            obs, _, is_done, _ = env.step(action)
            step_times.append(time.time() - start_step)
            eps_steps += 1

        scores.append(env.current_score)
        eps_times.append(time.time() - start_eps)
        eps_lengths.append(eps_steps)

    print(f"n_episodes: {args.num_episodes}")
    print(f"avg_score: {float(sum(scores)) / len(scores):.2f}")
    print(f"median_score: {median(scores)}")
    print(f"avg_pred_time: {int(sum(pred_times) * 1e6)  // len(pred_times)}ns")
    print(f"avg_step_time: {int(sum(step_times) * 1e6) // len(step_times)}ns")
    print(f"avg_eps_time: {int(sum(eps_times) * 1e3) // len(eps_times)}ms")
    print(f"avg_eps_steps: {int(sum(eps_lengths) // len(eps_lengths))}")
    print(f"tot_time: {time.time() - start_time:.2f}s")
