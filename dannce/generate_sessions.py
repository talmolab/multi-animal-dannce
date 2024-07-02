import os
import random
from typing import List, Optional

SEED = 0
BASE_PATH = "/home/jovyan/talmolab-smb/eric/slap_2m/"
DAY = ["2022-10-30", "2022-10-21", "2022-10-20", "2022-10-19", "2022-10-07"]
DAY_DEBUG = ["2022-10-30"]

random.seed(SEED)

NUM_FRAMES = 1000


def generate_instance_index(num_frames: int, num_instances: int):
    use_instances = [random.randint(0, num_instances - 1) for _ in range(num_frames)]
    return use_instances


def create_use_frames(num_frames: int):
    use_frames = [random.randint(0, 18000) for _ in range(num_frames)]
    return use_frames


def generate_day(debug: Optional[bool] = False):
    if debug:
        random_day = DAY_DEBUG[0]
    else:
        random_day = random.choice(DAY)
    return os.path.join(BASE_PATH, random_day)


def generate_session(debug: Optional[bool] = False):
    # day_path = os.path.join(BASE_PATH, DAY)
    day_path = generate_day(debug=debug)
    session_list = sorted(os.listdir(day_path))
    if debug:
        random_session = session_list[0]
    else:
        random_session = random.choice(session_list)
    path = os.path.join(day_path, random_session)
    return path


if __name__ == "__main__":
    p = generate_session(debug=True)
    print(f"path: {p}")
