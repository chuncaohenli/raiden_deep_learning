from matplotlib import pyplot as plt
from raiden_env import *
env = Raiden_Env()
env.reset()
cnt = 1
while cnt < 1000:
    action = [0] * 9
    action[1] = 1
    observation, reward, terminated, hp, live = env.step(action)
    # plt.imshow(observation)
    # plt.show()
    cnt += 1
env.reset()
cnt = 1
while cnt < 100:
    action = [0] * 9
    action[4] = 1
    env.step(action)
    cnt += 1