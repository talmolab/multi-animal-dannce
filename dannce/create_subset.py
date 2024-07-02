import os
import json
import glob
import toml
import h5py
import pickle
import cv2 as cv
import numpy as np
from tqdm import tqdm
from typing import Optional

from dannce.generate_sessions import (
    create_use_frames,
    generate_session,
    generate_instance_index,
)

N_TARGETS = 15  # number of keypoints
NUM_FRAMES = 1000
# NUM_FRAMES = 100
NUM_INSTANCES = 2
SAVE_DIR = (
    "/home/jovyan/vast/ckapoor/keypoint-tracking/slap_2m_sample_cage_aligned_full"
)
FRAME_DIR = f"{SAVE_DIR}/frames"
POSE_DIR = f"{SAVE_DIR}/poses"
INTRINSICS_DIR = f"{SAVE_DIR}/intrinsics"
CAM_VIEWS = ["back", "backL", "mid", "midL", "side", "sideL", "top", "topL"]

# some useful variables
USE_FRAMES = create_use_frames(num_frames=NUM_FRAMES)
USE_INSTANCE = generate_instance_index(
    num_frames=NUM_FRAMES, num_instances=NUM_INSTANCES
)


def grab_frames(path: str, frame: int):
    # grab frames with specified frame index and video path
    # instantiate video handler
    vid_base_dirs = [os.path.join(path, view) for view in CAM_VIEWS]
    frame_dict = {}

    for path in vid_base_dirs:
        # vid_path = glob.glob(path + '*.mp4')
        search_pattern = os.path.join(path, "*.mp4")
        splits = path.split("/")
        view, session = splits[-1], splits[-2]

        vpath = os.path.join(FRAME_DIR, f"{session}_{frame}_{view}.jpg")
        if not os.path.exists(vpath):
            vid_path = glob.glob(search_pattern, recursive=True)[
                0
            ]  # we know that there is a single mp4 a priori
            video_capture = cv.VideoCapture(vid_path)
            # total frames in specified video
            total_frames = int(video_capture.get(cv.CAP_PROP_FRAME_COUNT))
            # grab specified frame
            video_capture.set(cv.CAP_PROP_POS_FRAMES, frame)
            # read specified frame
            _, frame_sel = video_capture.read()
            # save frames
            save_frame(
                save_dir=FRAME_DIR,
                view=view,
                session=session,
                frame_idx=frame,
                frame=frame_sel,
            )
        else:
            print(f"{vpath} exists, skipping redundant processing")


def save_frame(
    save_dir: str, view: str, session: str, frame_idx: int, frame: np.ndarray
):
    # save frame to specified path as jpg
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{session}_{frame_idx}_{view}.jpg")
    cv.imwrite(save_path, frame)


def grab_poses(
    path: str, frame: int, instance: int, cage_aligned: Optional[bool] = True
):
    # grab ground truth 2D + 3D poses in a specified path
    # for multi-instance case, select a random animal instance
    label_dict = {}
    label_2d_path = os.path.join(path, "reprojections.h5")
    if cage_aligned:
        label_3d_path = os.path.join(path, "aligned_points3d.h5")
    else:
        label_3d_path = os.path.join(path, "points3d.h5")

    label_dict["2D_keypoints"] = {}

    f_2d, f_3d = h5py.File(label_2d_path, "r"), h5py.File(label_3d_path, "r")

    # choose 3D points
    allcams_3d = np.array(f_3d["tracks"][:])
    # handle multi-instance cas
    if allcams_3d.shape[1] == 1:
        selected_3d = allcams_3d[frame, 0, ...].reshape(N_TARGETS, -1)
    else:
        selected_3d = allcams_3d[frame, instance, ...]

    for cam in CAM_VIEWS:
        allcams_2d = np.array(f_2d[cam][:])
        if allcams_2d.shape[1] == 1:
            selected_2d = allcams_2d[frame, 0, ...].reshape(N_TARGETS, -1)
        else:
            selected_2d = allcams_2d[frame, instance, ...].reshape(N_TARGETS, -1)
        label_dict["2D_keypoints"][cam] = selected_2d.tolist()
        label_dict["3D_keypoints"] = selected_3d.tolist()
        label_dict["frame_idx"] = frame

    return label_dict


def save_poses(label_dict: dict, session: str, frame: int):
    # save poses in JSON format to specified path
    fname = f"{session}_{frame}.json"

    if not os.path.exists(POSE_DIR):
        os.makedirs(POSE_DIR)

    fpath = os.path.join(POSE_DIR, fname)

    with open(fpath, "w") as f:
        json.dump(label_dict, f)
        # json.dump(save_dict, f)


def grab_intrinsics(path: str):
    # grab camera intrinsics from specified path
    calib = toml.load(os.path.join(path, "calibration.toml"))
    param_dict = {}
    cams = [key for key in calib.keys() if key.startswith("cam")]

    for cam in cams:
        camname = calib[cam]["name"]
        param_dict[camname] = {}
        all_params = calib[cam]
        # get all intrinsics
        K = np.array(all_params["matrix"])
        RDistort = np.array(all_params["distortions"][0:3])  # very hacky fix
        TDistort = np.array(all_params["distortions"][1:3])
        r = np.array(all_params["rotation"])
        R, _ = cv.Rodrigues(r)  # convert rotation vector to 3x3 rotation matrix
        t = np.array(all_params["translation"])
        t = np.expand_dims(t, axis=0)
        # populate dictionary
        param_dict[camname]["K"] = K.tolist()
        param_dict[camname]["RDistort"] = RDistort.tolist()
        param_dict[camname]["TDistort"] = TDistort.tolist()
        param_dict[camname]["r"] = R.tolist()
        param_dict[camname]["t"] = t.tolist()
        param_dict[camname]["R"] = R.tolist()
        # add to params list
    return param_dict


def save_intrinsics(intrinsics: dict, session: str):
    # save intrinsics to specified path from a random session
    # save as JSON

    fname = f"{session}.json"

    if not os.path.exists(INTRINSICS_DIR):
        os.makedirs(INTRINSICS_DIR)

    fpath = os.path.join(INTRINSICS_DIR, fname)

    with open(fpath, "w") as f:
        json.dump(intrinsics, f)


def create_subset(save_dir: str, cage_aligned: Optional[bool] = True):
    # helper function to run everything else
    gt_attempts = 0
    pbar = tqdm(total=NUM_FRAMES)
    # check for sessions without GT
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    while gt_attempts != NUM_FRAMES:
        try:
            # generate session paths
            session_path = generate_session(debug=False)
            session_name = session_path[session_path.rfind("/") + 1 :]
            # handle intrinsics
            calib_path = os.path.join(save_dir, session_name + ".toml")

            # handle poses
            label_dict = grab_poses(
                path=session_path,
                frame=USE_FRAMES[gt_attempts],
                instance=USE_INSTANCE[gt_attempts],
                cage_aligned=True,
            )
            save_poses(
                label_dict=label_dict,
                session=session_name,
                frame=USE_FRAMES[gt_attempts],
            )

            # if not os.path.exists(calib_path):
            calib = grab_intrinsics(session_path)

            # handle cage-aligned coordinates
            # recompute rotation + translation for cage-aligned coords for every view
            if cage_aligned:
                for view in CAM_VIEWS:
                    pts_2d = np.array(label_dict["2D_keypoints"][view])
                    pts_3d = np.array(label_dict["3D_keypoints"])
                    K = np.array(calib[view]["K"])
                    Rdistort = np.array(calib[view]["RDistort"])
                    TDistort = np.array(calib[view]["TDistort"])
                    # construct distortion coefficients
                    dist_coeffs = np.hstack([Rdistort, TDistort])
                    _, rvecs, tvecs, _ = cv.solvePnPRansac(
                        pts_3d, pts_2d, K, distCoeffs=dist_coeffs
                    )
                    R, _ = cv.Rodrigues(rvecs)
                    # rewrite rotation + translation
                    calib[view]["R"] = R.tolist()
                    calib[view]["t"] = tvecs.tolist()

            # save intrinsics labeled by session
            save_intrinsics(intrinsics=calib, session=session_name)

            # handle frames
            grab_frames(path=session_path, frame=USE_FRAMES[gt_attempts])

            gt_attempts += 1
            pbar.update(1)
        except:
            print(
                f"session: {session_path} does not have GT, searching for next session"
            )


def save_all_poses():

    annot_dict = {}
    annot_dict["camera_names"] = CAM_VIEWS
    annot_dict["all_poses"] = []

    root = "/home/jovyan/talmolab-smb/eric/slap_2m/"
    poses = "consolidated_poses.pkl"

    days = ["2022-10-30", "2022-10-21", "2022-10-20", "2022-10-19", "2022-10-07"]
    all_exps = [
        os.path.join(root, day, session)
        for day in days
        if os.path.isdir(os.path.join(root, day))
        for session in os.listdir(os.path.join(root, day))
        if os.path.isdir(os.path.join(root, day, session))
    ]

    # all_poses = []
    save_dir = "/home/jovyan/vast/ckapoor/keypoint-tracking/slap_2m_sample/"
    file_path = os.path.join(save_dir, poses)
    with open(file_path, "wb") as f:
        pickle.dump(annot_dict, f)

    # for exp in tqdm(all_exps):


if __name__ == "__main__":
    create_subset(save_dir=SAVE_DIR)
    # save_all_poses()
    # load specific video, save frames
    # vid_path = "/home/jovyan/talmolab-smb/eric/slap_2m/2022-10-07/10072022173152/side/"

    # video_capture = cv.VideoCapture(vid_path)
