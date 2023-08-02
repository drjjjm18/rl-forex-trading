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

    avg_return = rewards / eps_step    
    return environment.envs[0].current_value, avg_return

def calculate_rsi(data, window=14):
    close_prices = data['Close']
    delta = close_prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi