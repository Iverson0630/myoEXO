import os

import numpy as np
import matplotlib.pyplot as plt
from bvh import Bvh


ARM_JOINTS = [
    "Character1_RightShoulder",
    "Character1_RightArm",
    "Character1_RightForeArm",
    "Character1_RightHand",
    "Character1_LeftShoulder",
    "Character1_LeftArm",
    "Character1_LeftForeArm",
    "Character1_LeftHand",
]


def load_bvh(path):
    with open(path) as f:
        return Bvh(f.read())


def extract_joint_rotation_series(mocap, joint_name):
    channels = mocap.joint_channels(joint_name)
    rot_channels = [ch for ch in channels if "rotation" in ch.lower()]
    if not rot_channels:
        return {}, []

    series = {ch[0].upper(): [] for ch in rot_channels}
    for f_idx in range(mocap.nframes):
        for ch in rot_channels:
            axis = ch[0].upper()
            val = float(mocap.frame_joint_channel(f_idx, joint_name, ch))
            series[axis].append(val)
    return series, rot_channels


def plot_arm_joints(bvh_path):
    mocap = load_bvh(bvh_path)
    time = np.arange(mocap.nframes) * mocap.frame_time

    rows = len(ARM_JOINTS)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.2 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    for idx, joint_name in enumerate(ARM_JOINTS):
        ax = axes[idx]
        series, rot_channels = extract_joint_rotation_series(mocap, joint_name)
        if not series:
            ax.text(0.5, 0.5, f"{joint_name}: no rotation channels", ha="center", va="center")
            ax.set_axis_off()
            continue

        for axis in sorted(series.keys()):
            ax.plot(time, series[axis], label=f"{axis} rot")
        ax.set_ylabel(joint_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    bvh_path = os.path.join(os.path.dirname(__file__), "motion", "walk.bvh")
    plot_arm_joints(bvh_path)
