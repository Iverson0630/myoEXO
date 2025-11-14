
from myosuite.utils import gym
import deprl
import time
# we can also change the reset_type of the environment here
env = gym.make('fullBodyWalk-v0', reset_type='init')
path = "./baselines_DEPRL/myofullbody/251114.003628/"
pt_name = "step_57000000.pt"
policy = deprl.load_baseline(env, path, pt_name)

for ep in range(100):
    obs = env.reset()
    for i in range(1000):
      
        action = policy(obs)
  
        next_obs, reward, done, terminate, info = env.step(action)
        env.sim.renderer.render_to_window()
        obs = next_obs
        time.sleep(0.01)
        if done:
            break