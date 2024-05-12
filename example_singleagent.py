from gym_battlesnake_pytorch.gymbattlesnake import BattlesnakeEnv

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy


env = BattlesnakeEnv(n_threads=4, n_envs=32)

model = PPO(ActorCriticCnnPolicy, env, verbose=1, learning_rate=1e-3, tensorboard_log="./tensorboard_logs/")
model.learn(total_timesteps=100000)
model.save('ppo2_trainedmodel')

del model

model = PPO.load('ppo2_trainedmodel')

obs = env.reset()
for _ in range(10000):
    action,_ = model.predict(obs)
    obs,_,_,_ = env.step(action)
    env.render()