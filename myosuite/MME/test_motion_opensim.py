import os
import requests
import pandas as pd
from urllib.parse import urlparse
import mujoco
import numpy as np
import skvideo
import skvideo.io

def read_mot(url_or_path):
    """
    读取 OpenSim .mot 文件：
    - 如果传入的是 URL，会自动下载
    - 如果传入的是本地路径，会直接读取
    返回 pandas.DataFrame
    """
    # 判断是否是 URL
    if url_or_path.startswith("http"):
        parsed_url = urlparse(url_or_path)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "downloaded_file.mot"

        # 若文件不存在则下载
        if not os.path.exists(filename):
            print(f"⬇️ Downloading from {url_or_path} ...")
            try:
                req = requests.get(url_or_path, allow_redirects=True, stream=True)
                req.raise_for_status()
                with open(filename, "wb") as f:
                    for chunk in req.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"✅ File downloaded successfully to: {filename}")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"❌ Error downloading file: {e}")
        else:
            print(f"✅ File already exists: {filename}")
    else:
        filename = url_or_path
        if not os.path.exists(filename):
            raise FileNotFoundError(f"❌ Local file not found: {filename}")

    # 找到 "endheader" 行
    skiprows = 0
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip().lower() == "endheader":
                break
            skiprows += 1

    # 读取数据（自动识别制表符或空格分隔）
    df = pd.read_csv(filename, sep=r"\s+", skiprows=skiprows + 1)
    print(f"📊 Loaded {len(df)} rows × {len(df.columns)} columns")
    return df
from IPython.display import HTML
from base64 import b64encode

def show_video(video_path, video_width = 400):

  video_file = open(video_path, "r+b").read()

  video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
  return HTML(f"""<video autoplay width={video_width} controls><source src="{video_url}"></video>""")
url = "https://raw.githubusercontent.com/opensim-org/opensim-models/refs/heads/master/Pipelines/Gait2392_Simbody/OutputReference/subject01_walk1_ik.mot"
df = read_mot(url)

mj_model = mujoco.MjModel.from_xml_path(
    '../simhive/myo_sim/leg/myolegs.xml'
)
mj_data = mujoco.MjData(mj_model)

joint_names = [mj_model.joint(jn).name for jn in range(mj_model.njnt)]
subc = [c for c in df.columns if c in joint_names]

print(
    f"Joints in the Mot files that are not present in the MJC model: {set(subc) - set(joint_names)}"
)

# ---- camera settings
camera = mujoco.MjvCamera()
camera.azimuth = 90
camera.distance = 3
camera.elevation = -45.0
camera.lookat = [0,0,.75]
options_ref = mujoco.MjvOption()
options_ref.flags[:] = 0
options_ref.geomgroup[1:] = 0
renderer_ref = mujoco.Renderer(mj_model)
renderer_ref.scene.flags[:] = 0
frames=[]
from tqdm import tqdm
for t in tqdm(range(len(df)), desc="Rendering frames"):
    for jn in subc:
        mjc_j_idx = mj_model.joint(joint_names.index(jn)).qposadr
        mj_data.qpos[mjc_j_idx] = np.deg2rad(df[jn].loc[t])
        if "knee_angle" in jn:  # knee joints have negative sign in myosuite
            mj_data.qpos[mjc_j_idx] *= -1

    mujoco.mj_forward(mj_model, mj_data)
    renderer_ref.update_scene(mj_data, camera=camera)#, scene_option=options_ref)
    frame = renderer_ref.render()
    frames.append(frame)

os.makedirs('videos', exist_ok = True)
output_name = 'videos/playback_mot.mp4'
skvideo.io.vwrite(output_name, np.asarray(frames),outputdict={"-pix_fmt": "yuv420p"})
# show in the notebook
show_video('videos/playback_mot.mp4')