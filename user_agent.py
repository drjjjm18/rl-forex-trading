import os
import shutil
from typing import Optional, Tuple

import tensorflow as tf
from tf_agents.environments import TFEnvironment
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.agents.dqn.dqn_agent import DqnAgent


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units: int):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.selu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))


class TradeAgent(DqnAgent):
    def __init__(self,
                 env: TFEnvironment,
                 learning_rate: float = 1e-4,
                 fc_layer_params: Tuple[int] = (64, 32, 16),
                 discount_factor: float = 0.99,
                 epsilon: float = 0.3,
                 target_update_tau: float = 0.01,
                 target_update_period: int = 1,
                 train_step_counter: Optional[tf.Variable] = None):
        action_tensor_spec = tensor_spec.from_spec(env.action_spec())
        num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1
        # QNetwork consists of a sequence of Dense layers followed by a dense layer
        # with `num_actions` units to generate one q_value per available action as
        # its output.
        dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))

        q_net = sequential.Sequential(dense_layers + [q_values_layer])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if train_step_counter is None:
            train_step_counter = tf.Variable(0)

        super().__init__(
            env.time_step_spec(),
            env.action_spec(),
            q_network=q_net,
            epsilon_greedy=epsilon,
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_huber_loss,
            gamma=discount_factor,
            train_step_counter=train_step_counter,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period)

        self._saver = policy_saver.PolicySaver(self.policy)


    def save(self, dir: str, overwrite=True):
        if os.path.exists(dir):
            if overwrite:
                shutil.rmtree(dir)
            else:
                raise RuntimeError("Directory exists, set overwrite to True")
        os.makedirs(dir)

        self._saver.save(dir)

