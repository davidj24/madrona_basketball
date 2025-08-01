import sys
import time
from env import EnvWrapper


def main():
    env = EnvWrapper(num_worlds, use_gpu=True, gpu_id=0)
    env.set_agent_idx(0)

    n_steps = 1000
    start_time = time.time()
    for i in range(n_steps):
        actions = env.get_blank_actions()
        obs, rew, done = env.step(actions)

    end_time = time.time()
    elapsed_frames = n_steps * num_worlds
    print(
        f"Average FPS: {elapsed_frames / (end_time - start_time):.2f} frames/sec")


if __name__ == "__main__":
    num_worlds = int(sys.argv[1])
    main()
