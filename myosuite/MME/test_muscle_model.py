from myosuite.utils import gym
import time
import mujoco
from mujoco import viewer
import numpy as np
from register_env import register_mme
paused = False   # 是否暂停
def key_callback(key):
    """在viewer中按空格暂停/继续"""

    if key == " ":
        paused = not paused
        print("⏸️ 暂停" if paused else "▶️ 继续")

def load_muscle_sinwave():
    model = mujoco.MjModel.from_xml_path("../simhive/myo_sim/body/myofullbodyarms_muscle_A.xml")
    print("nq:", model.nq)
    print("joint names:", [model.joint(i).name for i in range(model.njnt)])



    data = mujoco.MjData(model)

    # 列出所有关节名称
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


    # 每个关节逐个演示角度变化
    with viewer.launch_passive(model, data) as v:
        while v.is_running():

            time.sleep(20.0)
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
def load_zero_action():
    register_mme()
    env = gym.make('fullBodyWalk-v0')
    env.reset()
    muscle_names = [env.sim.model.id2name(i, "actuator") for i in range(env.sim.model.nu)]
    muscle_act_mask = env.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    muscle_act_count = int(muscle_act_mask.sum())

    zero_action = np.zeros(env.sim.model.nu, dtype=np.float32)
    for _ in range(2000):
        env.step(zero_action)
        env.mj_render()
        time.sleep(0.1)
    forces = env.sim.data.actuator_force
    activations = env.sim.data.act

    # for name, a, f in zip(muscle_names, activations, forces):
    #     print(f"{name:20s} | activation={a:.3f} | force={f:.1f}")

    # for i in range(env.sim.model.njnt):
    #     name = env.sim.model.id2name(i, "joint")
    #     addr = env.sim.model.jnt_qposadr[i]
    #     print(f"{i:02d}  {name:20s}  qpos index: {addr}")



    print("🔹 关节数量:", env.sim.model.njnt)
    print("🔹 自由度数量:", env.sim.model.nq)
    print("🔹 肌肉执行器数量:", muscle_act_count)

    for i in range(env.sim.model.njnt):
        name = env.sim.model.id2name(i,'joint')
        joint_type = env.sim.model.jnt_type[i]
        dof_start = env.sim.model.jnt_dofadr[i]
        #print(f"{i:2d}: {name:25s}  type={joint_type}  dof_index={dof_start}")


def test1():
    import mujoco, numpy as np

    model = mujoco.MjModel.from_xml_path("../simhive/myo_sim/body/myofullbodyarms_muscle_A.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def body_xmat(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xmat[bid].reshape(3,3).copy()

    def body_pos(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[bid].copy()

    def site_pos(name):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        return data.site_xpos[sid].copy()

    print("arm_align_r xmat:\\n", body_xmat("arm_align_r"))
    print("arm_align_l xmat:\\n", body_xmat("arm_align_l"))
    print("arm_align_r pos:", body_pos("arm_align_r"))
    print("arm_align_l pos:", body_pos("arm_align_l"))

    print("MFtip_r:", site_pos("MFtip_r"))
    print("RFtip_r:", site_pos("RFtip_r"))
    print("MFtip_l:", site_pos("MFtip_l"))
    print("RFtip_l:", site_pos("RFtip_l"))



# test1()
# load_muscle_sinwave()
load_zero_action()
