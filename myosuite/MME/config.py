from dataclasses import dataclass

@dataclass
class Config:
    class model:
        num_env: int = 16
        use_exo: bool = False
        model_path: str = "../simhive/myo_sim/body/myofullbodyarms_muscle.xml"
        bvh_path : str = 'motion/walk.bvh'
        env : str = 'myoLegWalk-v1' #myoLegWalk fullBodyBalance
        sim_hz: int = 1000
        ctl_hz: int = 100
    class train:
        algo: str = "SAC"
        n_steps: int = 2048
        n_epochs:int = 10
        max_iteration: int = 2e7
        buffer_size:int = 3_000_000
        batch_size:int = 256
        learning_rate:float = 3e-4
        lr_schedule: str = "linear"
        clip_ratio:float = 0.2
        gamma: float = 0.99
        gae_lambda: float = 0.95
        ent_coef: str = "auto"
        train_freq: int = 5
        gradient_steps: int = 4
        learning_starts: int = 20_000
        target_update_interval: int = 1
        tau: float = 0.002
        target_entropy: str = "auto"
        policy_hiddens: tuple = (256, 256)
        q_hiddens: tuple = (256, 256)
        save_freq: int = 1
        save_step:int = 200_000

    class save_dir:
        checkpoints:str = 'None'
        nn_dir: str = 'nn/human/myoLegWalk_SAC'
        wandb_project:str = 'MME'
        wandb_dir : str= 'myoLegWalk_SAC'
