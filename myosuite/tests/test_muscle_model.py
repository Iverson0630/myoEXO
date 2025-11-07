from myosuite.utils import gym
import time
import mujoco

# model = mujoco.MjModel.from_xml_path("myosuite/simhive/myo_sim/body/myofullbody.xml")
# print("nq:", model.nq)
# print("joint names:", [model.joint(i).name for i in range(model.njnt)])


env = gym.make('myoLegWalk-v0')
env.reset()
for _ in range(1000):
    env.mj_render()
    env.step(env.action_space.sample()) # take a random action
    time.sleep(0.1)