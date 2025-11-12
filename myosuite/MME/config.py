from dataclasses import dataclass

@dataclass
class Config:
    class model:
        num_env: int = 32
        use_exo: bool = False
        model_path: str = "../simhive/myo_sim/leg/myolegs.xml"
        bvh_path : str = 'motion/walk.bvh'
    class train:
        num_epochs: int = 10
        max_iteration: int = 10000
        buffer_size:int = 2048
        batch_size:int = 128
        default_learning_rate:float = 1E-4
        default_clip_ratio:float = 0.2

    class save_dir:
        checkpoints:str = 'None'
        nn_dir: str = 'nn/human/walk'
        wandb_project:str = 'MME'
        wandb_dir : str= 'human_walk'