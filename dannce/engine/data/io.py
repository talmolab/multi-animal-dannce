"""Data loading and saving operations."""
import toml
import random
import h5py
import cv2 as cv
import numpy as np
import scipy.io as sio
from typing import List, Dict, Text, Union
import mat73

# TODO (CK): make modular
# currently hardcoded frames, pts
n_frames = 18000
n_cams = 8
n_kpts = 15
n_targets = 81
label3d_path = "./points3d.h5"
label2d_path = "./reprojections.h5"


def load_camera_params(path: Text) -> List[Dict]:
    """
    Load camera params from toml file
    """
    # do something
    calib = toml.load(path)
    params = []
    cams = [key for key in calib.keys() if key.startswith("cam")]

    for cam in cams:
        param_dict = {}
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
        param_dict["K"] = K
        param_dict["RDistort"] = RDistort
        param_dict["TDistort"] = TDistort
        param_dict["r"] = R
        param_dict["t"] = t
        param_dict["R"] = R
        # add to params list
        params.append(param_dict)

    return params


def load_sync(path: Text) -> List[Dict]:
    """
    Assume all cameras are syncronised
    """
    # generate synchronization matrices
    sync = []
    data_2d_mat = np.zeros((n_frames, n_kpts * 2))
    data_3d_mat = np.zeros((n_frames, n_kpts * 3))
    data_frames_mat = np.array([[frame] for frame in range(n_frames)])
    sample_id_mat = np.array([[(10 * idx) + 1] for idx in range(n_frames)])

    for cam in range(n_cams):
        cam_dict = {}
        cam_dict["data_2d"] = data_2d_mat
        cam_dict["data_3d"] = data_3d_mat
        cam_dict["data_frame"] = data_frames_mat
        cam_dict["data_sampleID"] = sample_id_mat

        # populate list
        sync.append(cam_dict)

    return sync


def load_labels(path: Text) -> List[Dict]:
    """
    Load keypoint labels from points3d file
    """
    labels = []
    f1 = h5py.File(label2d_path, "r")
    f2 = h5py.File(label3d_path, "r")
    calib = toml.load(path)

    # get camnmes (in a consistent order)
    cams = [key for key in calib.keys() if key.startswith("cam")]

    # generate a list of target frames
    # random selection increases the probability of temporally inconsequential frames
    frame_idx = sorted([random.randint(0, n_frames) for _ in range(n_targets)])

    # generate corresponding sample IDs (sid)
    frame_idx_id = [(10 * sid) + 1 for sid in frame_idx]

    # choose corresponding 3d points
    allcams_3d = np.array(f2["tracks"][:])
    allcams_3d = np.squeeze(allcams_3d, axis=1)
    selected_3d = allcams_3d[frame_idx].reshape(n_targets, -1)  # flatten 3d points

    for cam in cams:
        label_dict = {}
        cam_name = calib[cam]["name"]
        allcams_2d = np.array(
            f1[cam_name][:]
        )  # list of all 2d points for a particular view
        allcams_2d = np.squeeze(allcams_2d, axis=1)  # squeeze along animal dim
        selected_2d = allcams_2d[frame_idx].reshape(n_targets, -1)  # flatten 2d points
        # populate label dictionary
        label_dict["data_2d"] = selected_2d
        label_dict["data_3d"] = selected_3d
        label_dict["data_frame"] = np.array(frame_idx)
        label_dict["data_sampleID"] = np.array(frame_idx_id)
        # populate label list
        labels.append(label_dict)

    return labels


def load_camnames(path: Text) -> Union[List, None]:
    """
    Use default camera names
    """
    calib = toml.load(path)
    cams = [
        key for key in calib.keys() if key.startswith("cam")
    ]  # don't use metadata key
    # get corresponding camnames
    camnames = [calib[cam]["name"] for cam in cams]

    return camnames


def load_com(path: Text) -> Dict:
    """
    Compute COM from 3d points
    """
    com = {}
    f2 = h5py.File(label3d_path, "r")
    threeD_pts = np.array(f2["tracks"][:])
    threeD_pts = np.squeeze(threeD_pts, axis=1)
    data_frames_mat = np.array([[frame] for frame in range(n_frames)])
    sample_id_mat = np.array([[(10 * idx) + 1] for idx in range(n_frames)])
    # for each frame, compute COM of all keypoints
    com_coord = np.mean(threeD_pts, axis=1)
    com["com3d"] = com_coord
    com["sampleID"] = sample_id_mat

    return com


'''
def load_label3d_data(path: Text, key: Text):
    """Load Label3D data

    Args:
        path (Text): Path to Label3D file
        key (Text): Field to access

    Returns:
        TYPE: Data from field
    """
    try:
        d = sio.loadmat(path)[key]
        dataset = [f[0] for f in d]

        # Data are loaded in this annoying structure where the array
        # we want is at dataset[i][key][0,0], as a nested array of arrays.
        # Simplify this structure (a numpy record array) here.
        # Additionally, cannot use views here because of shape mismatches. Define
        # new dict and return.
        data = []
        for d in dataset:
            d_ = {}
            for key in d.dtype.names:
                d_[key] = d[key][0, 0]
            data.append(d_)
    except:
        d = mat73.loadmat(path)[key]
        data = [f[0] for f in d]
    return data


def load_camera_params(path: Text) -> List[Dict]:
    """Load camera parameters from Label3D file.

    Args:
        path (Text): Path to Label3D file

    Returns:
        List[Dict]: List of camera parameter dictionaries.
    """
    params = load_label3d_data(path, "params")
    for p in params:
        if "r" in p:
            p["R"] = p["r"]
        if len(p["t"].shape) == 1:
            p["t"] = p["t"][np.newaxis, ...]
    return params


def load_sync(path: Text) -> List[Dict]:
    """Load synchronization data from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of synchronization dictionaries.
    """
    dataset = load_label3d_data(path, "sync")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_labels(path: Text) -> List[Dict]:
    """Load labelData from Label3D file.

    Args:
        path (Text): Path to Label3D file.

    Returns:
        List[Dict]: List of labelData dictionaries.
    """
    dataset = load_label3d_data(path, "labelData")
    for d in dataset:
        d["data_frame"] = d["data_frame"].astype(int)
        d["data_sampleID"] = d["data_sampleID"].astype(int)
    return dataset


def load_com(path: Text) -> Dict:
    """Load COM from .mat file.

    Args:
        path (Text): Path to .mat file with "com" field

    Returns:
        Dict: Dictionary with com data
    """
    try:
        d = sio.loadmat(path)["com"]
        data = {}
        data["com3d"] = d["com3d"][0, 0]
        data["sampleID"] = d["sampleID"][0, 0].astype(int)
    except:
        data = mat73.loadmat(path)["com"]
        data["sampleID"] = data["sampleID"].astype(int)
    return data


def load_camnames(path: Text) -> Union[List, None]:
    """Load camera names from .mat file.

    Args:
        path (Text): Path to .mat file with "camnames" field

    Returns:
        Union[List, None]: List of cameranames
    """
    try:
        label_3d_file = sio.loadmat(path)
        if "camnames" in label_3d_file:
            names = label_3d_file["camnames"][:]
            if len(names) != len(label_3d_file["labelData"]):
                camnames = [name[0] for name in names[0]]
            else:
                camnames = [name[0][0] for name in names]
        else:
            camnames = None
    except:
        label_3d_file = mat73.loadmat(path)
        if "camnames" in label_3d_file:
            camnames = [name[0] for name in label_3d_file["camnames"]]
    return camnames
'''
