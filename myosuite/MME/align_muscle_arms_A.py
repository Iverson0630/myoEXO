import mujoco
import numpy as np
import math
import re
from pathlib import Path

# Compute alignment quats for A model and patch file in-place.
model_simple = mujoco.MjModel.from_xml_path('../simhive/myo_sim/body/myofullbodyarms.xml')
model_muscle = mujoco.MjModel.from_xml_path('../simhive/myo_sim/body/myofullbodyarms_muscle_A.xml')

data_simple = mujoco.MjData(model_simple)
data_muscle = mujoco.MjData(model_muscle)
mujoco.mj_forward(model_simple, data_simple)
mujoco.mj_forward(model_muscle, data_muscle)

def xmat(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xmat[bid].reshape(3,3).copy()

R_simple_r = xmat(model_simple, data_simple, 'humerus_r')
R_simple_l = xmat(model_simple, data_simple, 'humerus_l')
R_muscle_r = xmat(model_muscle, data_muscle, 'humerus_r')
R_muscle_l = xmat(model_muscle, data_muscle, 'humerus_l')

R_delta_r = R_simple_r @ R_muscle_r.T
R_delta_l = R_simple_l @ R_muscle_l.T

def mat_to_quat(m):
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2,1] - m[1,2]) / s
        y = (m[0,2] - m[2,0]) / s
        z = (m[1,0] - m[0,1]) / s
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = math.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2]) * 2
        w = (m[2,1] - m[1,2]) / s
        x = 0.25 * s
        y = (m[0,1] + m[1,0]) / s
        z = (m[0,2] + m[2,0]) / s
    elif m[1,1] > m[2,2]:
        s = math.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2]) * 2
        w = (m[0,2] - m[2,0]) / s
        x = (m[0,1] + m[1,0]) / s
        y = 0.25 * s
        z = (m[1,2] + m[2,1]) / s
    else:
        s = math.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1]) * 2
        w = (m[1,0] - m[0,1]) / s
        x = (m[0,2] + m[2,0]) / s
        y = (m[1,2] + m[2,1]) / s
        z = 0.25 * s
    q = np.array([w,x,y,z])
    return q/np.linalg.norm(q)

q_r = mat_to_quat(R_delta_r)
q_l = mat_to_quat(R_delta_l)

chain_path = Path('../simhive/myo_sim/body/myotorso_witharms_muscle_chain_A.xml')
text = chain_path.read_text(encoding='utf-8')

# Patch arm_align quats
text = re.sub(r'(body name="arm_align_r"[^>]*quat=")[^"]+(")', 
              r'\g<1>' + ' '.join(f'{v:.8f}' for v in q_r) + r'\g<2>', text)
text = re.sub(r'(body name="arm_align_l"[^>]*quat=")[^"]+(")', 
              r'\g<1>' + ' '.join(f'{v:.8f}' for v in q_l) + r'\g<2>', text)

chain_path.write_text(text, encoding='utf-8')
print('Updated arm_align quats in', chain_path)
