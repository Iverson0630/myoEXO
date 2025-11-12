from myosuite.utils import gym
import time
import mujoco
from mujoco import viewer

# model = mujoco.MjModel.from_xml_path("../simhive/myo_sim/leg/myolegs.xml")
# print("nq:", model.nq)
# print("joint names:", [model.joint(i).name for i in range(model.njnt)])


# data = mujoco.MjData(model)

# # 打开可交互窗口
# with viewer.launch_passive(model, data) as v:

#     while v.is_running():
#         v.sync()

def log_muscle_joint():
    env = gym.make('myoLegWalk-v0')
    env.reset()
    muscle_names = [env.sim.model.id2name(i, "actuator") for i in range(env.sim.model.nu)]


    forces = env.sim.data.actuator_force
    activations = env.sim.data.act

    for name, a, f in zip(muscle_names, activations, forces):
        print(f"{name:20s} | activation={a:.3f} | force={f:.1f}")

    for i in range(env.sim.model.njnt):
        name = env.sim.model.id2name(i, "joint")
        addr = env.sim.model.jnt_qposadr[i]
        print(f"{i:02d}  {name:20s}  qpos index: {addr}")



    print("🔹 关节数量:", env.sim.model.njnt)
    print("🔹 自由度数量:", env.sim.model.nq)

    for i in range(env.sim.model.njnt):
        name = env.sim.model.id2name(i,'joint')
        joint_type = env.sim.model.jnt_type[i]
        dof_start = env.sim.model.jnt_dofadr[i]
        print(f"{i:2d}: {name:25s}  type={joint_type}  dof_index={dof_start}")



log_muscle_joint()