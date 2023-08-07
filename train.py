import os
from argparse import ArgumentParser
import pandas as pd
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.policies import random_tf_policy, py_tf_eager_policy
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
import yaml

from environment import TradeEnvironment
from trade_agent import TradeAgent
from utils import compute_avg_return


if __name__ == "__main__":

    ## ----------------------- Read parameters and config ------------------------- ##

    parser = ArgumentParser()

    parser.add_argument('-s', '--save_dir', required=False, default='model/', help='Directory to save the file')
    parser.add_argument('-f', '--file_path', required=True, help='Path to the file')
    parser.add_argument('-t', '--train_config', default='config/train_config.yaml', help='Path to config yaml file')

    args = parser.parse_args()
    save_dir = args.save_dir
    file_path = args.file_path
    train_config_path = args.train_config

    try:
        with open(train_config_path, 'r') as f:
            train_config = yaml.safe_load(f)
    except Exception as e:
        raise(f'Failed to load train config, does yaml file {train_config_path} exist? \r\r Traceback: ', e)
    
    print('Loaded train config: ', train_config)
    
    ## ------------------------ Dataset -------------------------- ##

    df = pd.read_csv(file_path)

    ## ------------------------ Configuration -------------------------- ##

    # RL algorithm options
    num_iterations = len(df)
    initial_collect_steps = train_config['initial_collect_steps']
    collect_steps_per_iteration = train_config['collect_steps_per_iteration']
    replay_buffer_max_length = train_config['replay_buffer_max_length'] # memory. Batch size chooses x random samples from this (?)

    batch_size = train_config['replay_buffer_max_length']
    learning_rate = train_config['learning_rate']

    # Logging and evaluation options
    log_interval = train_config['log_interval']
    num_eval_episodes = train_config['num_eval_episodes']
    eval_interval = train_config['eval_interval']
    # QNet options
    # fc_layer_params = (256,)
    # fc_layer_params = (1024, )
    fc_layer_params = train_config['fc_layer_params']
    start_epsilon = train_config['start_epsilon']
    end_epsilon = train_config['end_epsilon']
    qnet_target_update_tau = train_config['qnet_target_update_tau']
    qnet_target_update_period = train_config['qnet_target_update_period']
    discount_factor = train_config['discount_factor']
    epochs = train_config['epochs']
    total_steps = epochs * num_iterations



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
        total_steps,
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

    losses = []

    saved_model = False
    

    for epoch in range(epochs):

        # Reset the train step.
        agent.train_step_counter.assign(0)

        # Evaluate random poicy
        val, avg_return = compute_avg_return(eval_env, random_policy, 100)
        print(f"Random policy avg return is: {avg_return}, final value is {val}")

        # Evaluate the agent's policy once before training.
        val, avg_return = compute_avg_return(eval_env, agent.policy, 100)
        print(f"Initial agent avg return is: {avg_return}, final value is {val}")
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

            losses.append(train_loss.numpy())

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
                    saved_model = True
                returns.append(avg_rew)
        print('Epoch complete')
        print(f'Epoch: {epoch+1}, value of portfolio: {train_env.envs[0].current_value}')
    
    print('Training complete.')
    if not saved_model:
        agent.save(save_dir)
        print(f"Saved model to {save_dir}")
    with open('training_loss.pkl', 'wb') as f:
        pickle.dump(losses, f)