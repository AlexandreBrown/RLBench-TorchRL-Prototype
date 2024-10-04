import gymnasium as gym
import rlbench  # noqa: F401 MUST BE KEPT FOR GYM TO FIND RLBench ENVs
import torch
from torchrl.collectors.collectors import SyncDataCollector
from torchrl.envs.utils import RandomPolicy
from torchrl.record.loggers.csv import CSVLogger
from torchrl.record import VideoRecorder
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs import GymEnv


if __name__ == "__main__":

    print("All RLBench envs : ")
    envs = gym.envs.registry
    rlbench_envs = [
        env_spec.id for env_spec in envs.values() if "rlbench/" in env_spec.id
    ]

    for env in rlbench_envs:
        print(env)

    video_logger = CSVLogger(
        exp_name="reach_target", log_dir="videos", video_format="mp4", video_fps=30
    )
    recorder = VideoRecorder(logger=video_logger, tag="iteration", skip=2)

    env = TransformedEnv(
        GymEnv(
            env_name="rlbench/reach_target-vision-v0",
            from_pixels=True,
            pixels_only=True,
        )
    )
    env.append_transform(recorder)

    policy = RandomPolicy(action_spec=env.action_spec)

    device = torch.device("cpu")

    collector = SyncDataCollector(
        create_env_fn=env,
        policy=policy,
        total_frames=200,
        max_frames_per_traj=50,
        frames_per_batch=200,
        device=device,
        storing_device=device,
    )

    for _ in collector:
        continue

    env.transform.dump()
    env.close()

    print("Success!")
