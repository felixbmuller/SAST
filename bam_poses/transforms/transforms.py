import numpy as np
import numpy.linalg as la
import torch

import bam_poses.transforms.rotation as rot
from bam_poses.data.utils import unknown_pose_shape_to_known_shape


def normalize_at_each_frame(
    seq: np.ndarray,
    jid_left=13,
    jid_right=14,
    allow_zero_z=False,
):
    """
    :param seq: {n_frames x 18 x 3}
    """
    left3d = seq[:, jid_left]
    right3d = seq[:, jid_right]

    if not allow_zero_z:
        leftz = left3d[:, 2]
        rightz = right3d[:, 2]
        if np.mean((leftz < 0.1) * 1) > 0.001:
            raise ValueError("Lefts with zeros...")
        if np.mean((rightz < 0.1) * 1) > 0.001:
            raise ValueError("Rights with zeros...")

    left3d = torch.from_numpy(left3d)
    right3d = torch.from_numpy(right3d)
    seq = torch.from_numpy(seq)
    R, mu = tn_get_normalization(left3d=left3d, right3d=right3d)
    return tn_framewise_transform(poses=seq, mu=mu, R=R).numpy()


def normalize(
    seq,
    frame: int,
    jid_left=13,
    jid_right=14,
    return_transform=False,
    allow_zero_z=False,
):
    """
    :param seq: {n_frames x 18 x 3}
    :param frame:
    """
    #  0  1  2  3  4
    #  5  6  7  8  9 10
    # 11 12 13 14 15 16
    seq = unknown_pose_shape_to_known_shape(seq)
    assert len(seq) > frame
    left3d = seq[frame, jid_left]
    right3d = seq[frame, jid_right]

    if not allow_zero_z:
        if np.isclose(left3d[2], 0.0):
            raise ValueError(f"Left seems to be zero! -> {left3d}")
        if np.isclose(right3d[2], 0.0):
            raise ValueError(f"Right seems to be zero! -> {right3d}")

    mu, R = get_normalization(left3d, right3d)
    if return_transform:
        return apply_normalization_to_seq(seq, mu, R), (mu, R)
    else:
        return apply_normalization_to_seq(seq, mu, R)


def undo_normalization_to_seq(seq, mu, R):
    """
    :param seq: {n_frames x 18 x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    seq = unknown_pose_shape_to_known_shape(seq)
    mu = np.expand_dims(np.expand_dims(np.squeeze(mu), axis=0), axis=0)
    R_T = np.transpose(R)
    seq = rot.apply_rotation_to_seq(seq, R_T)
    seq = seq + mu
    return seq


def apply_normalization_to_points3d(pts3d, mu, R):
    """
    :param pts3d: {n_points x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    mu = np.expand_dims(np.squeeze(mu), axis=0)
    pts3d = pts3d - mu
    return np.ascontiguousarray(pts3d @ R)


def apply_normalization_to_seq(seq, mu, R):
    """
    :param seq: {n_frames x 18 x 3}
    :param mu: {3}
    :param R: {3x3}
    """
    mu = np.expand_dims(np.expand_dims(np.squeeze(mu), axis=0), axis=0)
    seq = unknown_pose_shape_to_known_shape(seq)
    seq = seq - mu
    return rot.apply_rotation_to_seq(seq, R)


def get_normalization(left3d, right3d):
    """
    Get rotation + translation to center and face along the x-axis
    """
    mu = (left3d + right3d) / 2
    mu[2] = 0
    left2d = left3d[:2]
    right2d = right3d[:2]
    y = right2d - left2d
    y = y / (la.norm(y) + 0.00000001)
    angle = np.arctan2(y[1], y[0])
    R = rot.rot3d(0, 0, angle)
    return mu, R


def tn_framewise_transform(poses, mu, R):
    """
    :param poses: {n_batch x 17 x 3}
    :param mu: {n_batch x 3}
    :param R: {n_batch x 3x3}
    """
    n_batch = poses.size(0)
    is_flat = False
    if len(poses.shape) == 2:
        is_flat = True
        poses = poses.reshape(n_batch, -1, 3)
    mu = mu.unsqueeze(1)
    poses = poses - mu
    output_poses = poses @ R

    if is_flat:
        output_poses = output_poses.reshape((n_batch, -1))

    return output_poses


def tn_get_normalization(left3d, right3d):
    """
    :param left3d: {n_batch x 3}
    :param right3d: {n_batch x 3}
    Get rotation + translation to center and face along the x-axis
    """
    mu = (left3d + right3d) / 2
    mu[:, 2] = 0

    left2d = left3d[:, :2]
    right2d = right3d[:, :2]
    y = right2d - left2d
    y = y / (torch.linalg.norm(y, dim=1, keepdims=True) + 0.0000001)
    a = y[:, 0]
    b = y[:, 1]
    angle = torch.atan2(b, a)

    R = rot.tn_rot3d_c(angle)
    return R, mu

