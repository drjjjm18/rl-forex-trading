import os
from argparse import ArgumentParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.policies import random_tf_policy, py_tf_eager_policy
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver

from environment import TradeEnvironment
from user_agent import TradeAgent

import pandas as pd


def compute_avg_return(environment, policy, num_steps=None):

    rewards = 0

    if not num_steps:
        num_steps = len(environment.envs[0].df)

    time_step = environment.reset()
    eps_step = 0

    for i in range(num_steps):
        action_step = policy.action(time_step)
        time_step = environment.step(action_step.action)
        eps_step += 1
        rewards += time_step[1]

        if time_step.is_last():
            break

    avg_return = rewards / num_steps
#   avg_length = episode_length / num_episodes
    
    return environment.envs[0].current_value, avg_return


if __name__ == "__main__":

    ## ----------------------- Read parameters ------------------------- ##

    parser = ArgumentParser()

    parser.add_argument('-s', '--save_dir', required=False, default='model/', help='Directory to save the file')
    parser.add_argument('-f', '--file_path', required=True, help='Path to the file')

    args = parser.parse_args()
    save_dir = args.save_dir
    file_path = args.file_path


    ## ------------------------ Configuration -------------------------- ##
    df = pd.read_csv(file_path)

    # RL algorithm options
    num_iterations = len(df) #
    initial_collect_steps = 100
    collect_steps_per_iteration = 1 
    replay_buffer_max_length = 10000 # memory. Batch size chooses x random samples from this (?)

    batch_size = 42 # 
    learning_rate = 1e-5

    # Logging and evaluation options
    log_interval = 1000
    num_eval_episodes = 1000
    eval_interval = None
    # QNet options
    fc_layer_params = (256,128)
    # fc_layer_params = (1024, )
    # fc_layer_params = (32, 16, 8)
    start_epsilon = 0.3 # 
    end_epsilon = 0. 
    qnet_target_update_tau = 0.09
    qnet_target_update_period = 1
    discount_factor = 0.95



    ## ---------------- Create train and test environment ---------------- ##
    train_py_env = GymWrapper(TradeEnvironment(df))
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_py_env = GymWrapper(TradeEnvironment(df))
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    # Create random policy for environment
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

    print(f"Observation spec: {train_env.observation_spec()}")
    print(f"Action spec: {train_env.action_spec()}")


    ## ----------------------- Create DQN Agent -------------------------- ##

    # Decaying epsilon greedy
    train_step_counter = tf.Variable(0)
    epsilon = tf.compat.v1.train.polynomial_decay(
        start_epsilon,
        train_step_counter,
        num_iterations,
        end_learning_rate=end_epsilon
    )


    agent = TradeAgent(train_env, learning_rate, fc_layer_params, discount_factor, epsilon, qnet_target_update_tau, qnet_target_update_period, train_step_counter)
    agent.initialize()

    print("DQN agent network summary:")
    print(agent._q_network.summary())


    ## ------------------ Create replay buffer and prefill ----------------- ##
    # Create replay buffer
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    # Create and run driver to collect sampes from random policy
    dynamic_step_driver.DynamicStepDriver(
        train_env,
        py_tf_eager_policy.PyTFEagerPolicy(
        random_policy, use_tf_function=True),
        [replay_buffer.add_batch],
        num_steps=initial_collect_steps).run(train_env.reset())

    # Create iterator to sample from dataset
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)


    ## -------------------- Run training algorithm ------------------------- ##
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate random poicy
    val, avg_return = compute_avg_return(eval_env, random_policy, 100)
    print(f"Random policy avg return is: {avg_return}, final value is {val}")

    # Evaluate the agent's policy once before training.
    val, avg_return = compute_avg_return(eval_env, agent.policy, 100)
    print(f"Initial avg return is: {avg_return}, final value is {val}")
    returns = [avg_return]

    # Reset the environment.
    time_step = train_env.reset()

    # Create a driver to collect experience.
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        [replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration
        )
    

    for i in range(num_iterations):
        # Collect a few steps and save to the replay buffer.
        time_step, _ = collect_driver.run(time_step)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: \t loss = {1} \t value={2}'.format(step, train_loss, train_env.envs[0].current_value))

        if eval_interval and (step % eval_interval == 0):
            portfolio, avg_rew = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print(f'Evaluating model: portfolio {portfolio}, avg_rew {avg_rew}')
            # If best average return, save the model
            if len(returns) == 0 or avg_rew > max(returns):
                agent.save(save_dir)
                print(f"Saved model to {save_dir}")
            returns.append(avg_rew)
    print('training complete')
    print('value of portfolio: ', train_env.envs[0].current_value)
