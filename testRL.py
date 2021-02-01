import numpy as np
import gym
import GOAABenv
from gym import spaces
from gym.utils import seeding
from stable_baselines3 import PPO, A2C # DQN coming soon
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


# validate Environment
env = gym.make('GOAABenv-v0')
check_env(env, warn=False)
env = gym.make('GOAABenv-v0')

obs = env.reset()
print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())

# Hardcoded best agent: PP and CLF
print("\n----------HARDCODED AGENT----------")
print("Step {}".format(1))
obs, reward, done, info = env.step(0)
env.render()
print('reward=', reward, 'done=', done)

print("Step {}".format(2))
obs, reward, done, info = env.step(10)
env.render()
print('reward=', reward, 'done=', done)

if done:
    print("Goal reached!", "reward=", reward)
# Plug into an algorithm from stablebaselines

# Instantiate the env
env = gym.make('GOAABenv-v0')

# wrap it
env = make_vec_env(lambda: env, n_envs=1)

# Train the agent
print("\n----------TRAINING AGENT----------")
model = A2C('MlpPolicy', env, verbose=1).learn(1000)

# Test the trained agent
print("\n-------TESTING TRAINED AGENT--------")
obs = env.reset()
n_steps = 10
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print(info[0]['Steps Taken'])
    print(info[0]['Results'])
    print('reward=', reward, 'done=', done)
    if done:
        print("Goal reached!", "reward=", reward)
        break
