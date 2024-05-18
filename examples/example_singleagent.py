from gym_battlesnake_pytorch.gymbattlesnake import BattlesnakeEnv

from stable_baselines3 import PPO

N_ENVS = 16
N_STEPS = 32

env = BattlesnakeEnv(n_threads=4, n_envs=N_ENVS)

model = PPO("MlpPolicy", env, verbose=1, device="auto", n_steps=N_STEPS)
model.learn(total_timesteps=N_ENVS * N_STEPS * 20)
model.save('ppo2_trainedmodel')

del model

model = PPO.load('ppo2_trainedmodel')

obs = env.reset()
for _ in range(10000):
    action,_ = model.predict(obs)
    obs,_,_,_ = env.step(action)
    env.render()