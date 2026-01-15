from dataclasses import dataclass

@dataclass
class Config:
    class model:
        num_env: int = 8
        use_exo: bool = False
        model_path: str = "../simhive/myo_sim/body/myofullbodyarms_muscle.xml"
        bvh_path : str = 'motion/walk.bvh'
        env : str = 'fullBodyWalk-v0' #myoLegWalk
        sim_hz: int = 1000
        ctl_hz: int = 100
    class train:
        n_steps: int = 2048
        n_epochs:int = 10
        max_iteration: int = 20000000
        buffer_size:int = 2048
        batch_size:int = 256
        learning_rate:float = 2E-4
        clip_ratio:float = 0.2
        gamma: float = 0.99
        gae_lambda: float = 0.95
        ent_coef:float = 0.01
        save_freq: int = 1
        save_step:int = 200000

    class save_dir:
        checkpoints:str = 'None'
        nn_dir: str = 'nn/human/walk'
        wandb_project:str = 'MME'
        wandb_dir : str= 'walk'