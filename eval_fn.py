from tensorflow import saved_model
import numpy as np
from tf_agents.agents.dqn import dqn_agent
from tf_agents.trajectories import time_step as ts

policy = None


def init(model_path: str):
    global policy
    # Load saved policy
    policy = saved_model.load(model_path)


def agent_predict(observation: np.array) -> int:
    global policy
    # Agent takes an observation and returns an action
    step = ts.transition(observation[None, ...], np.array([0.]), discount=np.array([1.]))
    return policy.action(step).action.numpy()[0]
