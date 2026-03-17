import random
import os
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from copy import deepcopy
from scipy.stats import wasserstein_distance
import re


def check_cell_collisions(skiers):
    """
    Checks whether any two skiers occupy the same cell (i, j)
    at the same global timestep, respecting staggered start times.
    """
    # Compute the maximum global step among all skiers
    max_step = max(s.start_step + len(s.traj) for s in skiers)
    collision_found = False

    for t in range(max_step):
        occupied = {}  # (i, j) -> skier index

        for k, skier in enumerate(skiers):
            # Only check if skier has started by this global step
            if t >= skier.start_step:
                local_idx = t - skier.start_step  # convert global step to local trajectory index
                if local_idx < len(skier.traj):
                    i, j, _, _ = skier.traj[local_idx]
                    cell = (i, j)

                    if cell in occupied:
                        print(
                            f"COLLISION at global step {t}: "
                            f"Skier {occupied[cell] + 1} and Skier {k + 1} "
                            f"both at cell {cell}"
                        )
                        collision_found = True
                    else:
                        occupied[cell] = k

    if not collision_found:
        print("No same-cell collisions detected.")

def generate_unique_positions(M, num_skiers):
    # Start at top random (can get the same random number)
    rng = np.random.default_rng(42)
    positions = np.arange(5, M-5)
    rng.shuffle(positions)
    return positions[:num_skiers]

def save_all_boxes_mot(
    all_boxes,
    out_txt_path,
    frame_base=1,   # MOT is typically 1-based
    id_base=1,      # track IDs start at 1
    img_w=1920,
    img_h=1080,
    float_fmt="{:.3f}"
):
    """
    all_boxes: list of arrays OR np.ndarray
      - expected per-track arrays shaped (T,4) with columns [cx, cy, w, h]
      - NaNs indicate missing detections; those frames are skipped

    Saves MOT format:
      frame,id,x,y,w,h,1,1,1
    """
    # Normalize input to list of per-track arrays
    if isinstance(all_boxes, np.ndarray):
        if all_boxes.ndim == 3 and all_boxes.shape[-1] == 4:
            tracks = [all_boxes[i] for i in range(all_boxes.shape[0])]
        elif all_boxes.ndim == 2 and all_boxes.shape[-1] == 4:
            tracks = [all_boxes]
        else:
            raise ValueError(f"Unexpected ndarray shape for all_boxes: {all_boxes.shape}")
    else:
        tracks = list(all_boxes)

    os.makedirs(os.path.dirname(out_txt_path), exist_ok=True)

    lines = []
    for t_idx, trk in enumerate(tracks):
        if trk is None:
            continue
        trk = np.asarray(trk)
        if trk.size == 0:
            continue
        if trk.ndim != 2 or trk.shape[1] != 4:
            raise ValueError(f"Track {t_idx} has invalid shape {trk.shape}, expected (T,4)")

        track_id = id_base + t_idx

        # For each frame row in this track
        for f_idx in range(trk.shape[0]):
            cx, cy, w, h = trk[f_idx]

            # Skip any frame with NaN/Inf (missing detection)
            if not np.isfinite([cx, cy, w, h]).all():
                continue

            # Convert center to top-left
            # center → top-left
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0

            clipped = clip_bbox_xywh(x1, y1, w, h, img_w, img_h)

            if clipped is None:
                continue

            x1, y1, w, h = clipped

            frame_num = frame_base + f_idx

            # MOT: frame,id,x,y,w,h,conf,cls,vis  (you asked 1,1,1)
            lines.append(
                f"{frame_num},{track_id},"
                f"{float_fmt.format(x1)},{float_fmt.format(y1)},"
                f"{float_fmt.format(w)},{float_fmt.format(h)},"
                f"1,1,1"
            )

    # MOT files are commonly sorted by frame then id
    def _key(line):
        parts = line.split(",")
        return (int(parts[0]), int(parts[1]))
    lines.sort(key=_key)

    with open(out_txt_path, "w", newline="\n") as f:
        f.write("\n".join(lines))

    return out_txt_path

def clip_bbox_xywh(x, y, w, h, img_w, img_h, min_size=1.0):
    """
    Clip bbox to image frame.
    Returns clipped (x, y, w, h) or None if fully outside.
    """

    # original bottom-right
    x2 = x + w
    y2 = y + h

    # clip to image
    x1_c = np.clip(x, 0, img_w)
    y1_c = np.clip(y, 0, img_h)
    x2_c = np.clip(x2, 0, img_w)
    y2_c = np.clip(y2, 0, img_h)

    w_c = x2_c - x1_c
    h_c = y2_c - y1_c

    # reject if box vanished or too small
    if (w_c < min_size) or (h_c < min_size):
        return None

    return x1_c, y1_c, w_c, h_c

def sample_generation_config(img_sizes, alpha_range, fps_list, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    img_w, img_h = random.choice(img_sizes)
    alpha = rng.integers(alpha_range[0], alpha_range[1] + 1)
    fps = random.choice(fps_list)

    return img_w, img_h, int(alpha), fps

def write_seqinfo_ini(seq_dir, seq_name, fps, seqlength, img_w, img_h, imdir="img1", imext=".jpg"):
    """
    Writes MOTChallenge seqinfo.ini into seq_dir.
    """
    os.makedirs(seq_dir, exist_ok=True)
    ini_path = os.path.join(seq_dir, "seqinfo.ini")
    content = (
        "[Sequence]\n"
        f"name={seq_name}\n"
        f"imdir={imdir}\n"
        f"framerate={int(fps)}\n"
        f"seqlength={int(seqlength)}\n"
        f"imwidth={int(img_w)}\n"
        f"imheight={int(img_h)}\n"
        f"imext={imext}\n"
    )
    with open(ini_path, "w", newline="\n") as f:
        f.write(content)
    return ini_path

def choose_split(u, train=0.7, val=0.2, test=0.1):
    # assumes train+val+test == 1.0
    if u < train:
        return "train"
    if u < train + val:
        return "val"
    return "test"


def project_trajectories_pinhole(
        skiers,
        img_w, img_h,

        # Slope geometry
        slope_length_m=80.0,
        slope_width_top_m=40.0,
        slope_width_bottom_m=20.0,
        slope_angle_deg=18.0,  # slope inclination

        # Camera intrinsics
        fx=1500.0, fy=1500.0,  # 1000
        cx=None, cy=None,

        # Camera extrinsics (world → camera)
        R=np.eye(3),
        t=np.zeros(3),

        # Timing
        ca_dt=0.04,
        video_fps=30,

        # Skier physical size (meters)
        person_height_m=1.7,
        person_width_m=1.0,
        view=None,

        smooth_sigma=2.0
):
    """
    Project CA trajectories onto image space using a pinhole camera model.

    Parameters
    ----------
    skiers : list
        List of skiers with .traj = [(i,j,v,last_pos), ...] and attributes M, J
    img_w, img_h : int
        Image width and height (pixels)
    slope_length_m : float
        Downhill slope length in meters
    slope_width_top_m : float
        Top width of trapezoid (uphill) in meters
    slope_width_bottom_m : float
        Bottom width of trapezoid (downhill) in meters
    slope_angle_deg : float
        Inclination of slope in degrees
    fx, fy, cx, cy : float
        Camera intrinsics
    R, t : np.ndarray
        Camera extrinsics (world → camera)
    ca_dt : float
        Time per CA step
    video_fps : float
        Output frame rate
    person_height_m, person_width_m : float
        Skier physical dimensions
    smooth_sigma : float
        Trajectory smoothing parameter

    Returns
    -------
    all_bboxes : list of np.ndarray
        Bounding boxes per skier per frame [u, v, w, h]
    proj_trajs : list of np.ndarray
        2D projected trajectories per skier [u, v]
    """

    # Grid size
    M = skiers[0].M
    J = skiers[0].J
    cx = img_w / 2
    cy = img_h / 2
    # cx = 0.0
    # cy =0.0
    all_bboxes, proj_trajs, all_times, all_speeds, z_vals = [], [], [], [], []

    theta = np.deg2rad(slope_angle_deg)

    origin = "top_left"
    use_cell_centers = True

    # Choose origin in (i0, j0)
    if origin == "top_left":
        i0, j0 = 0, J - 1
    elif origin == "top_center":
        i0, j0 = (M - 1) / 2, J - 1,
    elif origin == "top_right":
        i0, j0 = M - 1, J - 1
    else:
        raise ValueError("origin must be: top_left, top_center, top_right")

    world_trajs = []

    global_total_time = max(
        (sk.start_step + len(sk.traj)) * ca_dt
        for sk in skiers
    )

    n_frames = int(np.ceil(global_total_time * video_fps))
    frame_times_global = np.linspace(0.0, global_total_time, n_frames)

    for skier in skiers:
        traj = skier.traj
        if len(traj) > 1:
            traj = np.asarray(skier.traj)

            start_step = skier.start_step * ca_dt
            i_s = gaussian_filter1d(traj[:, 0], sigma=9)  # 35
            j_s = gaussian_filter1d(traj[:, 1], sigma=9)  # 35
            world_pts = []
            for i_ca, j_ca in zip(i_s, j_s):  # traj:
                ci = i_ca + 0.5 if use_cell_centers else i_ca
                cj = j_ca + 0.5 if use_cell_centers else j_ca
                ci0 = i0 + 0.5 if use_cell_centers else i0
                cj0 = j0 + 0.5 if use_cell_centers else j0

                # Across-slope (width)
                X = (ci - ci0) * 0.5

                # Along-slope surface distance from origin (positive downhill)
                s = (cj0 - cj) * 0.5  # (cj- cj0 ) * 0.5 #

                # Downslope horizontal projection + vertical elevation
                Y = s * np.cos(theta)
                Z = -s * np.sin(theta)

                # print(X,Y,Z)

                world_pts.append([X, Y, Z, 1])

            world_pts = np.asarray(world_pts)

            L_s = (J - 1) * 0.5
            L_y = L_s * np.cos(theta)
            L_z = L_s * np.sin(theta)
            L_x = ((M - 1) * 0.5)
            s = 1.0 / np.sqrt(2.0)
    
            if view == 1:
                #BOTTOM-CENTER
                C = np.array([L_x / 2.0, L_y, -L_z + 10])  # 50
                R = np.array([
                    [1., 0., 0.],
                    [0., 0., -1.],
                    [0., -1., 0.]
                ])
                t = (-R @ C).reshape(3, 1)
                
            elif view == 2:
                #TOP-CENTER
                C = np.array([L_x / 2.0, 0.0, 10.0])
                R = np.array([
                    [1., 0., 0.],
                    [0., 0., -1.],
                    [0., 1., 0.]
                ])

                # print(R,t)
                angle_deg = 25
                angle_rad = np.deg2rad(angle_deg)
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                    [0, np.sin(angle_rad), np.cos(angle_rad)]
                ])
                R = R @ Rx
                t = (-R @ C).reshape(3, 1)
                
            elif view == 3:
                # LEFT
                C = np.array([0-50, L_y / 2, 0.0]) #0-50, L_y / 2, 10.0
                R = np.array([
                    [0., 1., 0.],
                    [0., 0., -1.],
                    [1., 0., 0.]
                ])
                """angle_deg = 25
                angle_rad = np.deg2rad(angle_deg)
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                    [0, np.sin(angle_rad), np.cos(angle_rad)]
                ])
                R = R @ Rx"""
                t = (-R @ C).reshape(3, 1)
                
            elif view == 4:
                # right
                C = np.array([L_x+50, L_y / 2, 0.0])
                R = np.array([
                    [0., -1., 0.],
                    [0., 0., -1.],
                    [-1., 0., 0.]
                ])
                """angle_deg = 35
                angle_rad = np.deg2rad(angle_deg)
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                    [0, np.sin(angle_rad), np.cos(angle_rad)]

                ])
                R = R @ Rx"""
                t = (-R @ C).reshape(3, 1)
                
            elif view == 5:
                # TOP-LEFT corner 
                C = np.array([0.0 , 0.0 , 10.0]) #0.0 + 20, 0.0 + 20, 20.0

                R = np.array([
                    [-s, s, 0.],  # right
                    [0., 0., -1.],  # up    = -Z_world
                    [s, s, 0.]  # fwd   (inward)
                ])

                angle_deg = 30
                angle_rad = np.deg2rad(angle_deg)
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                    [0, np.sin(angle_rad), np.cos(angle_rad)]
                ])
                R = R @ Rx
                t = (-R @ C).reshape(3, 1)
                
            elif view == 6:
                # TOP-RIGHT corner 
                C = np.array([L_x, 0.0, 10.0])

                R = np.array([
                    [-s, -s, 0.],  # right
                    [0., 0., -1.],  # up    = -Z_world
                    [-s, s, 0.]  # fwd   (inward)
                ])

                angle_deg = 30
                angle_rad = np.deg2rad(angle_deg)
                Rx = np.array([
                    [1, 0, 0],
                    [0, np.cos(angle_rad), -np.sin(angle_rad)],
                    [0, np.sin(angle_rad), np.cos(angle_rad)]
                ])
                R = R @ Rx
                t = (-R @ C).reshape(3, 1)
                
            elif view == 7:
                # BOTTOM-LEFT corner 
                C = np.array([0.0, L_y , -L_z + 10.0])  # 0.0-20, L_y+20 , -L_z + 50.0

                R = np.array([
                    [s, s, 0.],  # right
                    [0., 0., -1.],  # up    = -Z_world
                    [s, -s, 0.]  # fwd   (inward)
                ])
                t = (-R @ C).reshape(3, 1)

                # fx = 1500.0
                # fy = 1500.0

            elif view == 8:
                # BOTTOM-RIGHT corner 
                C = np.array([L_x, L_y , -L_z + 10.0])

                R = np.array([
                    [s, -s, 0.],  # right
                    [0., 0., -1.],  # up    = -Z_world
                    [-s, -s, 0.]  # fwd   (inward)
                ])
                t = (-R @ C).reshape(3, 1)
                # fx = 1500.0
                # fy = 1500.0

            """angle_deg = 45
            angle_rad = np.deg2rad(angle_deg)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(angle_rad), -np.sin(angle_rad)],
                [0, np.sin(angle_rad), np.cos(angle_rad)]
            ])
            R = R @ Rx"""

            # print(R.shape,t.shape)

            comb = np.concatenate([R, t], axis=1)

            # print(R.shape, t.shape, world_pts.T.shape, comb.shape)
            # Transform world points to camera frame
            cam_pts = comb @ world_pts.T
            # cam_pts = (R.T @ (world_pts.T - t[:, None]))

            # cam_pts = (R @ world_pts.T).T + t
            cam_pts = cam_pts.T
            Z = cam_pts[:, 2]
            # print(R.shape, t.shape, world_pts.T.shape, comb.shape, cam_pts.shape)

            valid = Z > 0.1
            cam_pts = cam_pts[valid]
            Z = Z[valid]

            # -------------------------------------------------------
            # Pinhole projection
            # print(fx, Z, cx)
            K =  np.array([[fx, 0,  cx],
                           [0,  fy, cy],
                           [0,  0,  1]])
            pts = K @ cam_pts.T
            #print(pts.shape)
            pts = pts.T
            w = pts[:, 2]
            u = pts[:, 0]/w
            v = pts[:, 1]/w
            
            proj_pts = np.stack([u, v, w], axis=1)
            

            # -------------------------------------------------------
            # Interpolate to video frames
            ca_times = start_step + np.arange(len(proj_pts)) * ca_dt


            interp_u = interp1d(ca_times, proj_pts[:, 0], kind="linear", bounds_error=False,
                                fill_value=np.nan)
            interp_v = interp1d(ca_times, proj_pts[:, 1], kind="linear", bounds_error=False,
                                fill_value=np.nan)
            interp_z = interp1d(ca_times, proj_pts[:, 2], kind="linear", bounds_error=False,
                                fill_value=np.nan) #Z

            u_f = interp_u(frame_times_global)
            v_f = interp_v(frame_times_global)
            z_f = interp_z(frame_times_global)

            inside_mask = (
                    ((u_f >= 0) & (u_f <= img_w) &
                     (v_f >= 0) & (v_f <= img_h))
                    | (~np.isfinite(u_f)) | (~np.isfinite(v_f))
            )

            

            # Valid where we have a real projection + in front of camera + inside image

            u_f = u_f[inside_mask]
            v_f = v_f[inside_mask]
            z_f = z_f[inside_mask]

            valid = (
                    np.isfinite(u_f) & np.isfinite(v_f) & np.isfinite(z_f) &
                    (u_f >= 0) & (u_f <= img_w) &
                    (v_f >= 0) & (v_f <= img_h)
            )

            proj_traj = np.stack([u_f, v_f], axis=1)

            # -------------------------------------------------------
            # Depth-consistent bounding boxes
            # Keep NaN where invalid (object not present / not visible)
            h_px = np.full_like(z_f, np.nan, dtype=float)
            w_px = np.full_like(z_f, np.nan, dtype=float)

            h_px[valid] = fy * person_height_m / z_f[valid]
            w_px[valid] = fx * person_width_m / z_f[valid]
            
            bboxes = np.stack([u_f, v_f, w_px, h_px], axis=1)

            # -------------------------------------------------------
            # Speed (pixels/sec)
            diffs = np.diff(proj_traj, axis=0)
            speed = np.linalg.norm(diffs, axis=1) * video_fps
            speed = np.concatenate([[0.0], speed])

            z_vals.append(z_f)
            all_bboxes.append(bboxes)
            proj_trajs.append(proj_traj)
            all_times.append(frame_times_global)
            all_speeds.append(speed)

    return all_bboxes, proj_trajs, z_vals  # optionally return all_times, all_speeds

def sample_ability(alpha, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    probs = ability_probs_from_alpha(alpha)
    return rng.choice(abilities, p=probs)

abilities = ["Beginner", "Intermediate", "Advanced"]

def ability_probs_from_alpha(alpha):
    """
    Returns probability distribution over abilities
    based on slope angle alpha.
    """

    if alpha <= 14:
        # gentle slope
        probs = [0.65, 0.30, 0.05]

    elif alpha <= 22:
        # moderate slope
        probs = [0.35, 0.45, 0.20]

    else:
        # steep slope
        probs = [0.05, 0.40, 0.55]

    return probs


def smooth_traj(mean_i, mean_j, sigma=3):
    """
    Gaussian smoothing of trajectory coordinates over time.
    sigma: smoothing strength (in time steps)
    """
    smooth_i = gaussian_filter1d(mean_i, sigma=sigma, mode="nearest")
    smooth_j = gaussian_filter1d(mean_j, sigma=sigma, mode="nearest")
    return smooth_i, smooth_j


VIEWPOINT_GROUPS_GT = {
    1: ["slope_track000001", "slope_track000002", "slope_track000016", "slope_track000017"],
    2: ["slope_track000003", "slope_track000004"],
    3: ["slope_track000005", "slope_track000020"],
    4: ["slope_track000006", "slope_track000018"],
    5: ["slope_track000007", "slope_track000012"],
    6: ["slope_track000008", "slope_track000013"],
    7: ["slope_track000009"],
    8: ["slope_track000010", "slope_track000011", "slope_track000015"],
    9: ["slope_track000014"],
    10: ["slope_track000019"],
}

def load_gt_tracks_all_splits_by_viewpoints(
    dataset_root,
    viewpoints,
    viewpoint_groups=VIEWPOINT_GROUPS_GT,
    splits=("train", "val", "test")
):
    """
    Load MOT GT tracks for all dataset splits, for multiple viewpoints.

    Parameters
    ----------
    dataset_root : str
        Root folder of dataset.
    viewpoints : list of int
        List of viewpoint numbers to load.
    viewpoint_groups : dict
        Mapping viewpoint -> list of sequence names.
    splits : tuple of str
        Dataset splits to load.

    Returns
    -------
    segments : dict
        {
            split: {
                viewpoint: {
                    sequence_name: {
                        track_id: ndarray (N,6) with [frame, id, x, y, w, h]
                    }
                }
            }
        }
    """

    segments = {}

    for split in splits:
        split_path = os.path.join(dataset_root, split)
        split_segments = {}

        for vp in viewpoints:
            if vp not in viewpoint_groups:
                # skip invalid viewpoints
                continue

            vp_segments = {}

            for seq_name in viewpoint_groups[vp]:
                gt_path = os.path.join(split_path, seq_name, "gt", "gt.txt")

                if not os.path.exists(gt_path):
                    # sequence may not exist in this split
                    continue

                tracks = defaultdict(list)

                with open(gt_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue

                        fields = line.strip().split(",")
                        frame = int(fields[0])
                        track_id = int(fields[1])
                        x = float(fields[2])
                        y = float(fields[3])
                        w = float(fields[4])
                        h = float(fields[5])

                        tracks[track_id].append([frame, track_id, x, y, w, h])

                # Sort and convert to numpy
                seq_tracks = {}
                for tid, rows in tracks.items():
                    arr = np.array(rows, dtype=np.float32)
                    arr = arr[np.argsort(arr[:, 0])]
                    seq_tracks[tid] = arr

                vp_segments[seq_name] = seq_tracks

            if vp_segments:
                split_segments[vp] = vp_segments

        if split_segments:
            segments[split] = split_segments

    return segments

def load_gen_tracks_all_splits_assume_viewpoint1(
    dataset_root,
    splits=("train", "val", "test"),
    assumed_viewpoint=1,
):
    """
    Load MOT GT tracks for all dataset splits, but keep the viewpoint structure.
    All sequences are assigned to `assumed_viewpoint`.

    Returns
    -------
    segments : dict
        {
            split: {
                assumed_viewpoint: {
                    sequence_name: {
                        track_id: ndarray (N,6) with [frame, id, x, y, w, h]
                    }
                }
            }
        }
    """
    segments = {}

    for split in splits:
        split_path = os.path.join(dataset_root, split)

        # Split folder may not exist or may be empty -> skip
        if not os.path.isdir(split_path):
            continue

        try:
            seq_names = [
                d for d in os.listdir(split_path)
                if os.path.isdir(os.path.join(split_path, d))
            ]
        except OSError:
            continue

        if not seq_names:
            continue

        vp_segments = {}

        for seq_name in seq_names:
            gt_path = os.path.join(split_path, seq_name, "gt", "gt.txt")
            if not os.path.isfile(gt_path):
                continue

            tracks = defaultdict(list)

            with open(gt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    fields = line.split(",")
                    frame = int(fields[0])
                    track_id = int(fields[1])
                    x = float(fields[2])
                    y = float(fields[3])
                    w = float(fields[4])
                    h = float(fields[5])

                    tracks[track_id].append([frame, track_id, x, y, w, h])

            # Sort and convert to numpy
            seq_tracks = {}
            for tid, rows in tracks.items():
                arr = np.asarray(rows, dtype=np.float32)
                arr = arr[np.argsort(arr[:, 0])]
                seq_tracks[tid] = arr

            if seq_tracks:
                vp_segments[seq_name] = seq_tracks

        if vp_segments:
            segments[split] = {assumed_viewpoint: vp_segments}

    return segments

def _infer_viewpoint_from_seq_name(seq_name: str, base_id: int = 33, num_views: int = 8):
    """
    Map sequence name to viewpoint index in {1..num_views} using the pattern:
      slope_track000033 -> view 1
      slope_track000034 -> view 2
      ...
      slope_track000040 -> view 8
      slope_track000041 -> view 1
      ...

    Parameters
    ----------
    seq_name : str
        e.g. "slope_track000033"
    base_id : int
        first sequence number corresponding to view 1
    num_views : int
        number of viewpoints in the cycle

    Returns
    -------
    int or None
        viewpoint in [1..num_views], or None if seq_name doesn't match.
    """
    m = re.search(r"slope_track(\d+)", seq_name)
    if not m:
        return None
    seq_id = int(m.group(1))
    vp = ((seq_id - base_id) % num_views) + 1
    return vp


def load_gen_tracks_all_splits_select_viewpoints(
    dataset_root,
    splits=("train", "val", "test"),
    selected_viewpoints=(1, 2, 3, 4, 5, 6, 7, 8),
    base_seq_id_for_view1=33,
    num_viewpoints=8,
):
    """
    Load MOT GT tracks for all dataset splits, grouped by viewpoint inferred from
    sequence naming order (cyclic over 8 views). The user can choose which viewpoints
    to include via `selected_viewpoints`.

    View mapping pattern:
      slope_track000033 -> view 1
      slope_track000034 -> view 2
      ...
      slope_track000040 -> view 8
      slope_track000041 -> view 1
      etc.

    Parameters
    ----------
    dataset_root : str
        Root folder containing split subfolders: train/val/test
    splits : tuple[str]
        Which splits to load
    selected_viewpoints : iterable[int]
        Viewpoints to include, subset of {1..8}
    base_seq_id_for_view1 : int
        Numeric ID for the first sequence that corresponds to view 1 (default 33)
    num_viewpoints : int
        Number of viewpoints in the cycle (default 8)

    Returns
    -------
    segments : dict
        {
            split: {
                viewpoint: {
                    sequence_name: {
                        track_id: ndarray (N,6) with [frame, id, x, y, w, h]
                    }
                }
            }
        }
    """
    """if len(selected_viewpoints)>1:
        selected_viewpoints = set(int(v) for v in selected_viewpoints)
    else:
        selected_viewpoints = set(int(selected_viewpoints))"""

    selected_viewpoints = set(int(v) for v in selected_viewpoints)
    segments = {}

    for split in splits:
        split_path = os.path.join(dataset_root, split)
        #print(split_path)

        # Split folder may not exist or may be empty -> skip
        if not os.path.isdir(split_path):
            continue

        try:
            seq_names = [
                d for d in os.listdir(split_path)
                if os.path.isdir(os.path.join(split_path, d))
            ]
        except OSError:
            continue

        if not seq_names:
            continue

        # Optional: sort for determinism (not required for ID-based mapping, but nice)
        seq_names = sorted(seq_names)

        vp_segments = defaultdict(dict)

        for seq_name in seq_names:
            vp = _infer_viewpoint_from_seq_name(
                seq_name,
                base_id=base_seq_id_for_view1,
                num_views=num_viewpoints,
            )
            if vp is None:
                continue
            if vp not in selected_viewpoints:
                continue

            gt_path = os.path.join(split_path, seq_name, "gt", "gt.txt")
            if not os.path.isfile(gt_path):
                continue

            tracks = defaultdict(list)

            with open(gt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    fields = line.split(",")
                    frame = int(fields[0])
                    track_id = int(fields[1])
                    x = float(fields[2])
                    y = float(fields[3])
                    w = float(fields[4])
                    h = float(fields[5])

                    tracks[track_id].append([frame, track_id, x, y, w, h])

            # Sort and convert to numpy
            seq_tracks = {}
            for tid, rows in tracks.items():
                arr = np.asarray(rows, dtype=np.float32)
                arr = arr[np.argsort(arr[:, 0])]
                seq_tracks[tid] = arr

            if seq_tracks:
                vp_segments[vp][seq_name] = seq_tracks

        if vp_segments:
            # Convert defaultdict to normal dict for clean serialization/printing
            segments[split] = {vp: dict(vp_segments[vp]) for vp in sorted(vp_segments.keys())}

    return segments


# Example usage:
# segments = load_gen_tracks_all_splits_select_viewpoints(
#     dataset_root="/path/to/slope-track",
#     splits=("train", "val", "test"),
#     selected_viewpoints=(1, 4, 7),  # user chooses
#     base_seq_id_for_view1=33,
#     num_viewpoints=8,
# )

def wasserstein_per_component(A_norm, B_norm):
    """
    Computes 1D Wasserstein-1 distance for each normalized component:
    cx_tilde, cy_tilde, w_tilde, h_tilde
    """
    A_cx, A_cy, A_w, A_h = flatten_normed_components(A_norm)
    B_cx, B_cy, B_w, B_h = flatten_normed_components(B_norm)

    if len(A_cx) == 0 or len(B_cx) == 0:
        raise ValueError("No normalized samples found in A_norm or B_norm.")

    print(A_cx, B_cx)

    out = {
        "W1_cx_tilde": wasserstein_distance(A_cx, B_cx),
        "W1_cy_tilde": wasserstein_distance(A_cy, B_cy),
        "W1_w_tilde":  wasserstein_distance(A_w,  B_w),
        "W1_h_tilde":  wasserstein_distance(A_h,  B_h),
    }
    return out

def flatten_normed_components(segments_norm):
    """
    segments_norm tracks are (N,6): [frame, id, cx_tilde, cy_tilde, w_tilde, h_tilde]
    Returns four 1D arrays: cx, cy, w, h pooled across everything.
    """
    cx_all, cy_all, w_all, h_all = [], [], [], []

    for split_data in segments_norm.values():
        for view_data in split_data.values():
            for seq_data in view_data.values():
                for arr in seq_data.values():
                    if arr is None:
                        continue
                    a = np.asarray(arr, dtype=float)
                    if a.size == 0:
                        continue
                    cx_all.append(a[:, 2])
                    cy_all.append(a[:, 3])
                    w_all.append(a[:, 4])
                    h_all.append(a[:, 5])

    def cat(lst):
        return np.concatenate(lst) if lst else np.array([], dtype=float)

    #print(cx_all)

    return cat(cx_all), cat(cy_all), cat(w_all), cat(h_all)

def normalize_segments_running_mean(segments, sort_by_frame=True):
    """
    segments format:
    {
        split: {
            viewpoint: {
                sequence_name: {
                    track_id: ndarray (N,6) with [frame, id, x, y, w, h]
                }
            }
        }
    }

    Returns same structure but each track is ndarray (N,6):
    [frame, id, cx_tilde, cy_tilde, w_tilde, h_tilde]
    using running means up to each step t.
    """
    out = deepcopy(segments)
    eps = 1e-9

    for split, split_data in out.items():
        for viewpoint, view_data in split_data.items():
            for sequence_name, seq_data in view_data.items():
                for track_key, arr in seq_data.items():

                    if arr is None or len(arr) == 0:
                        continue

                    arr = np.asarray(arr, dtype=float)
                    if arr.ndim != 2 or arr.shape[1] != 6:
                        raise ValueError(f"Track {track_key} in {split}/{viewpoint}/{sequence_name} is not (N,6).")

                    # sort by frame if needed
                    if sort_by_frame and arr.shape[0] > 1:
                        idx = np.argsort(arr[:, 0])
                        arr = arr[idx]

                    frames = arr[:, 0]
                    ids    = arr[:, 1]
                    x      = arr[:, 2]
                    y      = arr[:, 3]
                    w      = arr[:, 4]
                    h      = arr[:, 5]

                    # centers
                    cx = x + 0.5 * w
                    cy = y + 0.5 * h

                    # first-step values (t=1)
                    cx1, cy1, w1, h1 = cx[0], cy[0], w[0], h[0]

                    # running means up to t (inclusive)
                    t = np.arange(1, len(w) + 1, dtype=float)  # 1..T
                    mean_w_t = np.cumsum(w) / t
                    mean_h_t = np.cumsum(h) / t

                    # numeric safety
                    #mean_w_t = np.maximum(mean_w_t, eps)
                    #mean_h_t = np.maximum(mean_h_t, eps)

                    # your equations with running mean denominators
                    cx_tilde = np.abs((cx - cx1)) / mean_w_t
                    cy_tilde = np.abs((cy - cy1)) / mean_h_t
                    w_tilde  = np.abs((w  - w1))  / mean_w_t
                    h_tilde  = np.abs((h  - h1))  / mean_h_t

                    seq_data[track_key] = np.column_stack([
                        frames,
                        ids,
                        cx_tilde,
                        cy_tilde,
                        w_tilde,
                        h_tilde
                    ])

    return out

import numpy as np
from copy import deepcopy

def normalize_segments_running_mean_direction(segments, sort_by_frame=True):
    """
    Same structure returned but coordinates are first rotated so that
    the sequence mean motion direction aligns with the x-axis.

    Output per track:
    [frame, id, cx_tilde, cy_tilde, w_tilde, h_tilde]
    """

    out = deepcopy(segments)
    eps = 1e-9
    alpha = 1.0
    min_disp = 0.5

    for split, split_data in out.items():
        for viewpoint, view_data in split_data.items():
            for sequence_name, seq_data in view_data.items():
                """
                    Mean motion direction using both number of tracks and track length.

                    Weight per track = L_i ** alpha, where:
                      alpha = 0.0 -> equal track weighting
                      alpha = 0.5 -> compromise weighting
                      alpha = 1.0 -> full length weighting

                    Each track contributes its normalized start-to-end direction.
                    """

                # -------------------------------------------------
                # 1. Estimate mean motion direction for sequence
                # -------------------------------------------------
                weighted_dirs = []

                for track_key, arr in seq_data.items():

                    if arr is None or len(arr) == 0:
                        continue

                    arr = np.asarray(arr, dtype=float)

                    if sort_by_frame and arr.shape[0] > 1:
                        arr = arr[np.argsort(arr[:,0])]

                    x = arr[:,2]
                    y = arr[:,3]
                    w = arr[:,4]
                    h = arr[:,5]

                    cx = x + 0.5 * w
                    cy = y + 0.5 * h

                    if len(cx) < 2:
                        continue

                    dx = cx[-1] - cx[0]
                    dy = cy[-1] - cy[0]
                    d = np.array([dx, dy], dtype=float)

                    disp = np.linalg.norm(d)
                    if disp < min_disp:
                        continue

                    L = len(cx) - 1
                    weight = L ** alpha

                    d_hat = d / (disp) #+ eps)
                    weighted_dirs.append(weight * d_hat)

                if len(weighted_dirs) == 0:
                    return np.array([1.0, 0.0], dtype=float)

                mean_dir = np.sum(weighted_dirs, axis=0)
                norm = np.linalg.norm(mean_dir)

                if norm < eps:
                    return np.array([1.0, 0.0], dtype=float)
                else:
                    mean_dir /= norm

                # -------------------------------------------------
                # 2. Build rotation matrix to align mean_dir -> x axis
                # -------------------------------------------------
                theta = np.arctan2(mean_dir[1], mean_dir[0]) #mean_dir[1], mean_dir[0]

                c = np.cos(-theta)
                s = np.sin(-theta)

                R = np.array([
                    [c, -s],
                    [s,  c]
                ])

                # -------------------------------------------------
                # 3. Process each track
                # -------------------------------------------------
                for track_key, arr in seq_data.items():

                    if arr is None or len(arr) == 0:
                        continue

                    arr = np.asarray(arr, dtype=float)

                    if sort_by_frame and arr.shape[0] > 1:
                        arr = arr[np.argsort(arr[:,0])]

                    frames = arr[:,0]
                    ids    = arr[:,1]
                    x      = arr[:,2]
                    y      = arr[:,3]
                    w      = arr[:,4]
                    h      = arr[:,5]

                    # centers
                    cx = x + 0.5*w
                    cy = y + 0.5*h

                    cx1, cy1, w1, h1 = cx[0], cy[0], w[0], h[0]

                    # displacement from first frame
                    dxy = np.column_stack([cx - cx1, cy - cy1])

                    # rotate displacements into mean-direction frame
                    dxy_rot = dxy @ R.T

                    cx_rot = dxy_rot[:,0]
                    cy_rot = dxy_rot[:,1]

                    # running mean scales
                    t = np.arange(1, len(w)+1, dtype=float)
                    mean_w_t = np.cumsum(w) / t
                    mean_h_t = np.cumsum(h) / t

                    mean_w_t = np.maximum(mean_w_t, eps)
                    mean_h_t = np.maximum(mean_h_t, eps)

                    """# normalized features
                    cx_tilde = cx_rot / mean_w_t
                    cy_tilde = cy_rot / mean_h_t"""

                    """den = max(abs(cx_rot[-1]), eps)  # total along-direction displacement
                    cx_tilde = cx_rot / den
                    cy_tilde = cy_rot / den"""

                    #if cx_rot[-1] < 0:
                    #cx_rot = np.abs(cx_rot)
                    #if cy_rot
                    #cy_rot = np.abs(cy_rot)


                    den_x = max(abs(cx_rot[-1]), eps)
                    den_y = max(np.max(np.abs(cy_rot)), eps)

                    cx_tilde = cx_rot / den_x
                    cy_tilde = cy_rot / den_y

                    w_tilde  = (w - w1) / mean_w_t
                    h_tilde  = (h - h1) / mean_h_t

                    seq_data[track_key] = np.column_stack([
                        frames,
                        ids,
                        cx_tilde,
                        cy_tilde,
                        w_tilde,
                        h_tilde
                    ])

    return out

def normalize_segments_by_trajectory_direction(segments, sort_by_frame=True):
    """
    Normalize each track in its own motion-aligned frame.

    For each track:
      1. Compute trajectory direction from start center to end center.
      2. Rotate displacements so the trajectory aligns with the x-axis.
      3. Normalize:
           - cx_tilde by total along-trajectory displacement
           - cy_tilde by max absolute lateral deviation
           - w_tilde by running mean width
           - h_tilde by running mean height

    Output per track:
      [frame, id, cx_tilde, cy_tilde, w_tilde, h_tilde]
    """

    out = deepcopy(segments)
    eps = 1e-9
    min_disp = 0.5

    for split, split_data in out.items():
        for viewpoint, view_data in split_data.items():
            for sequence_name, seq_data in view_data.items():

                for track_key, arr in seq_data.items():

                    if arr is None or len(arr) == 0:
                        continue

                    arr = np.asarray(arr, dtype=float)

                    if sort_by_frame and arr.shape[0] > 1:
                        arr = arr[np.argsort(arr[:, 0])]

                    frames = arr[:, 0]
                    ids    = arr[:, 1]
                    x      = arr[:, 2]
                    y      = arr[:, 3]
                    w      = arr[:, 4]
                    h      = arr[:, 5]

                    # box centers
                    cx = x + 0.5 * w
                    cy = y + 0.5 * h

                    # fallback for degenerate tracks
                    if len(cx) < 2:
                        seq_data[track_key] = np.column_stack([
                            frames,
                            ids,
                            np.zeros_like(cx),
                            np.zeros_like(cy),
                            np.zeros_like(w),
                            np.zeros_like(h)
                        ])
                        continue

                    # -------------------------------------------------
                    # 1. Compute per-track direction
                    # -------------------------------------------------
                    dx = cx[-1] - cx[0]
                    dy = cy[-1] - cy[0]
                    disp = np.hypot(dx, dy)

                    if disp < min_disp:
                        # No meaningful direction: keep zero motion features
                        cx_tilde = np.zeros_like(cx)
                        cy_tilde = np.zeros_like(cy)
                    else:
                        theta = np.arctan2(dy, dx)

                        # Rotate by -theta so trajectory aligns with +x
                        c = np.cos(-theta)
                        s = np.sin(-theta)
                        R = np.array([
                            [c, -s],
                            [s,  c]
                        ])

                        # -------------------------------------------------
                        # 2. Translate to first center and rotate
                        # -------------------------------------------------
                        dxy = np.column_stack([cx - cx[0], cy - cy[0]])
                        dxy_rot = dxy @ R.T

                        cx_rot = dxy_rot[:, 0]   # along-trajectory
                        cy_rot = dxy_rot[:, 1]   # lateral

                        # -------------------------------------------------
                        # 3. Normalize motion coordinates
                        # -------------------------------------------------
                        den_x = max(abs(cx_rot[-1]), eps)
                        den_y = max(np.max(np.abs(cy_rot)), eps)

                        cx_tilde = cx_rot / den_x
                        cy_tilde = cy_rot / den_y

                    # -------------------------------------------------
                    # 4. Normalize size changes
                    # -------------------------------------------------
                    w1, h1 = w[0], h[0]

                    t = np.arange(1, len(w) + 1, dtype=float)
                    mean_w_t = np.maximum(np.cumsum(w) / t, eps)
                    mean_h_t = np.maximum(np.cumsum(h) / t, eps)

                    w_tilde = (w - w1) / mean_w_t
                    h_tilde = (h - h1) / mean_h_t

                    seq_data[track_key] = np.column_stack([
                        frames,
                        ids,
                        cx_tilde,
                        cy_tilde,
                        w_tilde,
                        h_tilde
                    ])

    return out

def debug_extreme_tracks(segments_norm, col_idx=2, threshold=8.0, name="cx_tilde"):
    for split, split_data in segments_norm.items():
        for viewpoint, view_data in split_data.items():
            for sequence_name, seq_data in view_data.items():
                for track_id, arr in seq_data.items():
                    if arr is None or len(arr) == 0:
                        continue
                    arr = np.asarray(arr, dtype=float)
                    if arr.ndim != 2 or arr.shape[1] <= col_idx:
                        continue

                    vals = arr[:, col_idx]
                    if np.any(~np.isfinite(vals)):
                        print(
                            f"[BAD {name}] non-finite: "
                            f"split={split}, viewpoint={viewpoint}, seq={sequence_name}, track={track_id}"
                        )
                        continue

                    vmin = vals.min()
                    vmax = vals.max()
                    if vmin < -threshold or vmax > threshold:
                        print(
                            f"[EXTREME {name}] "
                            f"split={split}, viewpoint={viewpoint}, seq={sequence_name}, track={track_id}, "
                            f"min={vmin:.3f}, max={vmax:.3f}, len={len(vals)}"
                        )

def filter_segments_by_motion(segments, min_displacement=5.0):
    """
    Keep only tracks whose center displacement is at least min_displacement.
    Works on raw segments[split][viewpoint][sequence][track_id] -> (N,6)
    """
    filtered = {}

    for split, split_data in segments.items():
        filtered[split] = {}
        for viewpoint, view_data in split_data.items():
            filtered[split][viewpoint] = {}
            for sequence_name, seq_data in view_data.items():
                kept = {}

                for track_id, arr in seq_data.items():
                    if arr is None or len(arr) < 2:
                        continue

                    arr = np.asarray(arr, dtype=float)
                    if arr.ndim != 2 or arr.shape[1] != 6:
                        continue

                    arr = sort_track(arr)
                    cx, cy, _, _ = get_centers(arr)

                    disp = np.hypot(cx[-1] - cx[0], cy[-1] - cy[0])
                    if disp >= min_displacement:
                        kept[track_id] = arr

                filtered[split][viewpoint][sequence_name] = kept

    return filtered

def sort_track(arr):
    """
    Sort track by frame index (column 0).
    arr shape: (N,6)
    [frame, id?, x, y, w, h]
    """
    arr = np.asarray(arr, dtype=float)

    if arr.ndim != 2 or arr.shape[1] < 6:
        return arr

    if len(arr) > 1:
        arr = arr[np.argsort(arr[:, 0])]

    return arr

def get_centers(arr):
    """
    Convert bbox to center coordinates.

    Input arr shape (N,6):
    [frame, id?, x, y, w, h]

    Returns:
        cx, cy, w, h
    """
    arr = np.asarray(arr, dtype=float)

    x = arr[:, 2]
    y = arr[:, 3]
    w = arr[:, 4]
    h = arr[:, 5]

    cx = x + 0.5 * w
    cy = y + 0.5 * h

    return cx, cy, w, h
