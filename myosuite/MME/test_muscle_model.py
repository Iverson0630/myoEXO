from myosuite.utils import gym
import time
import mujoco
from mujoco import viewer
import numpy as np
paused = False   # 是否暂停
def key_callback(key):
    """在viewer中按空格暂停/继续"""

    if key == " ":
        paused = not paused
        print("⏸️ 暂停" if paused else "▶️ 继续")

def load_muscle_file():
    model = mujoco.MjModel.from_xml_path("../simhive/myo_sim/body/myofullbodyarms.xml")
    print("nq:", model.nq)
    print("joint names:", [model.joint(i).name for i in range(model.njnt)])



    data = mujoco.MjData(model)

    # 列出所有关节名称
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


    # 每个关节逐个演示角度变化
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            for j in range(model.njnt):
                
                name = joint_names[j]
                print(f"展示关节: {name}")
                addr = model.jnt_qposadr[j]
                while paused and v.is_running():
                    time.sleep(0.05)
                # 清零所有角度
                data.qpos[7:] = 0.0

                # 只转当前这个关节
                if name =='root':
                    continue
                    for idx in range(7):
                        for t in np.linspace(0, 2*np.pi, 100):
                            if not v.is_running():
                                break
                            angle = 30 * np.sin(t)  # ±30度摆动
                            
                            data.qpos[addr+idx] = np.deg2rad(angle)
                            mujoco.mj_forward(model, data)
                            v.sync()
                            time.sleep(0.05)
                else:
                    for t in np.linspace(0, 2*np.pi, 100):
                        if not v.is_running():
                            break
                        angle = 30 * np.sin(t)  # ±30度摆动
                        
                        data.qpos[addr] = np.deg2rad(angle)
                        mujoco.mj_forward(model, data)
                        v.sync()
                        time.sleep(0.05)
                time.sleep(2.0)
            
            break  # 播完所有关节退出
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


load_muscle_file()
#log_muscle_joint()