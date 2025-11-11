from myosuite.utils import gym
import time
import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("../simhive/myo_sim/leg/myolegs.xml")
print("nq:", model.nq)
print("joint names:", [model.joint(i).name for i in range(model.njnt)])


data = mujoco.MjData(model)

# 打开可交互窗口
with viewer.launch_passive(model, data) as v:

    while v.is_running():
        v.sync()
# env = gym.make('myoLegWalk-v0')
# env.reset()
# for _ in range(1000):
#     env.mj_render()
#     env.step(env.action_space.sample()) # take a random action
# env.close()