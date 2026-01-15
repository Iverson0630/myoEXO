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
    
    #model = mujoco.MjModel.from_xml_path("../simhive/myo_sim/arm/myoarm.xml")
    model = mujoco.MjModel.from_xml_path("../simhive/myo_sim/body/myofullbodyarms_muscle.xml")
    print("nq:", model.nq)
    print("joint names:", [model.joint(i).name for i in range(model.njnt)])



    data = mujoco.MjData(model)

    # 列出所有关节名称
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]


    # 只展示肩部与手臂相关关节
    arm_keywords = (
        "shoulder", "elv", "arm", "elbow", "wrist", "pro_sup",
        "flexion", "deviation", "sternoclavicular", "acromioclavicular",
        "unrotscap", "unrothum",
    )
    arm_joint_indices = []
    for j, name in enumerate(joint_names):
        name_l = (name or "").lower()
        if name_l == "root":
            continue
        if any(key in name_l for key in arm_keywords):
            arm_joint_indices.append(j)

    # 每个关节逐个演示角度变化
    with viewer.launch_passive(model, data) as v:
        while v.is_running():

    
            for j in arm_joint_indices:
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
                        time.sleep(0.01)
                time.sleep(1.0)
            
            break  # 播完所有关节退出
def load_rand_action():
    register_mme()
    env = gym.make('fullBodyWalk-v0')
    env.reset()
    joint_names = [env.sim.model.id2name(i, "joint") for i in range(env.sim.model.njnt)]
    muscle_names = [env.sim.model.id2name(i, "actuator") for i in range(env.sim.model.nu)]
    muscle_act_mask = env.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    muscle_act_count = int(muscle_act_mask.sum())
    print("🔹 关节数量:", env.sim.model.njnt)
    print("🔹 自由度数量:", env.sim.model.nq)
    print("🔹 肌肉执行器数量:", muscle_act_count)
    print("🔹 action 维度:", env.sim.model.nu)




    print("🔹 关节名称:")
    for name in joint_names:
        print(name)
    print("🔹 肌肉/执行器名称:")
    for name in muscle_names:
        print(name)

    for _ in range(2000):
        action = np.random.uniform(low=-1.0, high=1.0, size=env.sim.model.nu).astype(np.float32)
        env.step(action)
        env.mj_render()
        if env.sim.data.ncon:
            print("🔹 接触对:")
            for i in range(env.sim.data.ncon):
                c = env.sim.data.contact[i]
                geom1 = int(c.geom1)
                geom2 = int(c.geom2)
                # Prefer model.id2name to handle wrapper model types.
                if hasattr(env.sim.model, "id2name"):
                    try:
                        g1 = env.sim.model.id2name(geom1, "geom")
                        g2 = env.sim.model.id2name(geom2, "geom")
                    except TypeError:
                        g1 = env.sim.model.id2name(geom1, mujoco.mjtObj.mjOBJ_GEOM)
                        g2 = env.sim.model.id2name(geom2, mujoco.mjtObj.mjOBJ_GEOM)
                else:
                    g1 = mujoco.mj_id2name(env.sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
                    g2 = mujoco.mj_id2name(env.sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
                print(f"  {g1} <-> {g2}")
        time.sleep(0.1)
      
    forces = env.sim.data.actuator_force
    activations = env.sim.data.act

 

def load_single_muscle_max():
    register_mme()
    env = gym.make('fullBodyWalk-v0')
    env.reset()
    muscle_names = [env.sim.model.id2name(i, "actuator") for i in range(env.sim.model.nu)]
    muscle_act_mask = env.sim.model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    muscle_indices = [i for i, is_muscle in enumerate(muscle_act_mask) if is_muscle]
    # Filter out trunk/back-related muscles; keep arm/leg-focused actuators.
    trunk_keywords = (
        "abd", "obliq", "rect", "spine", "lumbar", "thorax", "back", "iliocost",
        "multif", "erector", "ql_", "il_", "ltpt", "ltpl", "mf_", "ps_", "io",
        "eo", "psoas", "lat_", "pecm", "pec", "serr", "trap", "rhom", "dia",
    )
    filtered_indices = []
    for idx in muscle_indices:
        name = (muscle_names[idx] or "").lower()
        if not any(key in name for key in trunk_keywords):
            filtered_indices.append(idx)
    muscle_indices = filtered_indices

    for idx in muscle_indices:
        print(f"🔹 肌肉名称: {muscle_names[idx]}")
        print(f"🔹 激活肌肉: {muscle_names[idx]}")
        action = np.zeros(env.sim.model.nu, dtype=np.float32)
        action[idx] = 1.0
        for _ in range(50):
            env.step(action)
            env.mj_render()
            time.sleep(0.02)






# load_muscle_sinwave()
load_rand_action()
# load_single_muscle_max()
