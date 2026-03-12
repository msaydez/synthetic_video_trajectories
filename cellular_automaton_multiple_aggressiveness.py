import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import imageio
import pickle
import os
from helper import (check_cell_collisions, generate_unique_positions, save_all_boxes_mot, sample_generation_config, sample_ability,
                    write_seqinfo_ini, choose_split, project_trajectories_pinhole)

from plotting import (plot_envelope_and_trajectories, plot_skiers_trajectories, plot_skiers_clashing, plot_simulation_curvature,
                      plot_speed_time, plot_space_time, plot_lateral_time, plot_space_time_2d, plot_bboxes_with_trajectories,
                      save_frames_and_video)


# --------------------------
# Directions
# --------------------------
DIRECTIONS = {
    -2: (-1, 0),
    -1: (-1, -1),
     0: (0, -1),
     1: (1, -1),
     2: (1, 0)
}

COS_THETA = {
    -2: 0.0,
    -1: np.cos(np.deg2rad(45)),
     0: 1.0,
     1: np.cos(np.deg2rad(45)),
     2: 0.0
}



# --------------------------
# Skier CA
# --------------------------
class SkierCA:
    def __init__(self, M=100, J=200, dt=0.04, alpha_deg=25, terrain_sigma=5, w6=0.5, min_dist=1, skier_id=None,
                 start_step=2, start_i=None, ability_level=None):
        self.M, self.J = M, J
        self.dt = dt
        self.turn_steps = 0
        self.min_dist = min_dist  # Minimum allowed distance to other skiers
        self.overtake_dist = 6  #cells ahead to consider overtaking

        self.delta_history = []   # stores past deltas
        self.d_safe = 30
        self.k_rhythm = 4       # number of past steps
        self.w8 = 0.2           # λ_r (rhythm strength)

        # --------------------------
        # Personal factors
        # --------------------------
        self.ability_level = ability_level  # "Beginner", "Intermediate", "Advanced"
        # Assign intrinsic aggressiveness A_s based on ability
        ability_to_aggressiveness = {
            "Beginner": 1.0,
            "Intermediate": 1.5,
            "Advanced": 2.0
        }
        self.aggressiveness = ability_to_aggressiveness.get(self.ability_level, 1.0)


        # Start at top random (can get the same random number)
        #rng = np.random.default_rng(42 + skier_id * 200)
        #self.i = rng.integers(1, M)
        #random.seed(42)

        #Start at top random (unique random number)
        self.i = start_i

        #Set start location at the top
        #self.i = random.randint(1, M) #M // 2
        self.j = J - 1

        self.start_step = start_step  # the simulation step when the skier actually begins
        self.active = False

        self.dir_prev = None
        self.delta_prev = 0
        #Random speed
        #rng = np.random.default_rng(42 + skier_id)
        #self.v = rng.integers(6, 20)

        #Set speed
        self.v = 2

        # Resistance parameters
        self.mu_straight = 0.002
        self.mu_turn = 0.006
        self.c_straight = 0.005
        self.c_turn = 0.007

        # Physics
        self.g = 9.81
        self.alpha = np.deg2rad(alpha_deg)

        # Transfer factors
        self.w1 = 0.1  # slope bias
        self.w2 = 0.1 # friction #0.01
        self.w3 = 5.0  # boundary
        self.w4 = 0.01  # curvature
        self.w5 = 0.1 # air resistance #0.01
        self.w6 = w6  # anticipation factor weight
        self.w7 = 5.0 #overtake
        self.gamma = 0.1 #random.uniform(0.1, 0.2) #0.1 # inertia #0.9 random.uniform(0.1, 0.2)

        # Terrain
        np.random.seed(42)
        h = gaussian_filter(np.random.rand(M, J), sigma=terrain_sigma)
        self.kappa = np.zeros_like(h)
        for i in range(1, M-1):
            for j in range(1, J-1):
                self.kappa[i,j] = (h[i+1,j] + h[i-1,j] + h[i,j+1] + h[i,j-1] - 4*h[i,j]) #*500 #1000

        """ self.d = np.array([0.0, -1.0])  # unit vector in (i, j) coordinates

        rng_terrain = np.random.default_rng(42)

        # Local perturbation (height map) \tilde{h}
        h_tilde = gaussian_filter(rng_terrain.random((M, J)), sigma=terrain_sigma)

        # Coordinates x = (i, j) on the lattice
        ii, jj = np.meshgrid(np.arange(M, dtype=float),
                             np.arange(J, dtype=float),
                             indexing="ij")

        # Base plane term: tan(alpha) * d^T x
        # d^T x = 0*ii + (-1)*jj = -jj
        h_plane = np.tan(self.alpha) * (self.d[0] * ii + self.d[1] * jj)

        # Total height field h(x) = tan(alpha) d^T x + \tilde{h}(x)
        self.h = h_plane + h_tilde

        # Discrete curvature (5-point Laplacian) from total height field
        self.kappa = np.zeros_like(self.h)
        for i in range(1, M - 1):
            for j in range(1, J - 1):
                self.kappa[i, j] = (
                        self.h[i + 1, j] + self.h[i - 1, j] +
                        self.h[i, j + 1] + self.h[i, j - 1] -
                        4.0 * self.h[i, j]
                )"""

        #segments = load_tracks_from_pickle("all_traj_occluded/view_01_clean_plus_occluded.pkl")

        #tracks_img = extract_image_space_tracks(segments)
        #print(segments)
        #tracks_ca  = image_to_ca(segments, M, J, IMG_W, IMG_H)

        #self.kappa = build_curvature_map(tracks_ca, M, J)


        # Friction field
        np.random.seed(60)
        self.mu = np.random.uniform(0.002, 0.012, size=(M, J))

        # Trajectory
        self.traj = [(self.i, self.j, self.v, self.delta_prev)]

    # --------------------------
    # Transfer factors
    # --------------------------
    """def fslope(self, delta):
        return 1 - np.exp(-self.w1 * np.sin(self.alpha))"""

    def fslope(self, delta):
        s = 1 - np.exp(-self.w1 * np.sin(self.alpha))  # Eq. (3)
        return s #* COS_THETA[delta]

    def fboundary(self, i_p):
        d = min(i_p, (self.M-1) - i_p)
        #print(d)
        return np.exp(-self.w3 / (d+ 1e-6))

    def fcurve(self, i_p, j_p):
        kappa = self.kappa[i_p, j_p]
        return np.exp(self.w4 * kappa)

    def fair(self):
        return np.exp(-self.w5 * self.v**2)

    def inertia(self, delta):
        return np.exp(-self.gamma * self.v * abs(delta - self.delta_prev))

    """def inertia(self, delta):
        return np.exp(self.v * abs(delta - self.delta_prev)/self.aggressiveness)"""

    def ffriction(self, i_p, j_p):
        return np.exp(-self.w2 * self.mu[i_p, j_p])

    def fovertake(self, i_p, j_p, others):
        factor=1.0
        for o in others:
            # same downhill path
            if o.i == i_p and o.j > j_p:
                d_ahead = o.j - j_p

                # ignore immediate collision zone
                if d_ahead <= self.min_dist:
                    continue

                # ignore too-far skiers
                if d_ahead > self.overtake_dist:
                   continue

                rel_speed = self.v - o.v
                if rel_speed > 0:
                    factor *= np.exp(-self.w7 * d_ahead / (rel_speed + 1e-6))
        return factor

    def fanticipation(self, i_p, j_p, others):
        dmin = np.inf

        for o in others:
            d = max(abs(i_p - o.i), abs(j_p - o.j))
            dmin = min(dmin, d)

            if dmin < self.min_dist:
                return 0.0  # hard constraint: too close

        # More aggressive skiers are less deterred by nearby skiers
        return np.exp(-1.0 / (self.aggressiveness * dmin))

    # --------------------------
    # Transition probabilities
    # --------------------------
    def transition_probabilities(self, others=None):
        admissible = list(DIRECTIONS.keys())
        P_tilde = {}
        for d in admissible:
            di, dj = DIRECTIONS[d]
            i_p = self.i + di
            j_p = self.j + dj
            if i_p < 0 or i_p >= self.M or j_p < 0:
                continue

            # Basic probability factors
            prob = self.fslope(d) * self.fboundary(i_p) * self.fcurve(i_p, j_p) * \
                   self.fair() * self.inertia(d) * self.ffriction(i_p, j_p) #* self.frhythm(d, i_p, j_p)

            # Anticipation
            if others:
                #print(l)
                prob = prob  * self.fanticipation(i_p, j_p, others) #* self.fovertake(i_p, j_p, others)

            if prob > 0:
                P_tilde[d] = prob

        if not P_tilde:
            return None
        Z = sum(P_tilde.values())
        return {d: P_tilde[d]/Z for d in P_tilde}

    # --------------------------
    # Speed update
    # --------------------------
    def update_speed(self, delta_new):
        d = np.array(DIRECTIONS[delta_new], dtype=float)
        d_curr = self.v * d
        theta_t = self.turning_angle(self.dir_prev, d_curr)
        turning = theta_t > 5.0
        if turning:
            self.turn_steps = 10
        if self.turn_steps > 0:
            mu = self.mu_turn
            c = self.c_turn
            self.turn_steps -= 1
        else:
            mu = self.mu_straight
            c = self.c_straight

        cos_theta = COS_THETA[delta_new]
        a = self.g * np.sin(self.alpha) * cos_theta - mu * self.g * np.cos(self.alpha) - c * self.v**2


        self.v += a * self.dt
        self.dir_prev = d_curr

    def turning_angle(self, v_prev, v_curr):
        if v_prev is None:
            return 0.0
        num = np.dot(v_prev, v_curr)
        den = np.linalg.norm(v_prev) * np.linalg.norm(v_curr)
        if den == 0:
            return 0.0
        cos_theta = np.clip(num / den, -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    # --------------------------
    # Single step
    # --------------------------
    # --------------------------
    # Single step with anticipation / hard constraint
    # --------------------------
    def step(self, others=None, reserved=None):
        probs = self.transition_probabilities(others=others)
        if probs is None:
            # Only record stationary step if not at bottom
            if self.j > 0:
                self.traj.append((self.i, self.j, self.v, self.delta_prev))
                return self.j > 0
            else:
                return False

        """if probs is None:
            return False"""


        # Choose move
        delta_new = random.choices(list(probs.keys()), weights=probs.values())[0]
        self.delta_history.append(delta_new)

        # Optional: cap history size
        if len(self.delta_history) > self.k_rhythm:
            self.delta_history.pop(0)
        #delta_new = max(probs, key=probs.get)

        # Update speed and position
        self.update_speed(delta_new)
        di, dj = DIRECTIONS[delta_new]
        self.i += di
        self.j += dj
        """if reserved is not None:
            reserved.add((i_p, j_p))"""
        self.delta_prev = delta_new

        # Record trajectory
        self.traj.append((self.i, self.j, self.v, delta_new))

        return self.j > 0  # True if still on slope


# --------------------------
# Multi-skier simulation
# --------------------------

def run_multi_skier(num_skiers=20, M=100, J=200, steps=1000, stagger=10, start_i=None, alpha=25):
    #abilities = ["Beginner", "Intermediate", "Advanced"]
    skiers=[]

    group_size = 10
    step = 50
    stagger_max = stagger  # your variable
    num_groups = (num_skiers + group_size - 1) // group_size
    num_steps = stagger_max // step  # available discrete start slots (excluding 0 if you want)

    # -----------------------------
    # 1) DISTINCT start time per group
    # -----------------------------
    if num_groups - 1 > num_steps:
        raise ValueError("Not enough distinct time slots to give each group a unique start. "
                         "Increase stagger_max or reduce num_groups/group_size.")

    rng_times = np.random.default_rng()
    available_slots = np.arange(1, num_steps + 1)  # 1..num_steps
    picked_slots = rng_times.choice(available_slots, size=num_groups - 1, replace=False)
    group_starts = np.empty(num_groups, dtype=int)
    group_starts[0] = 0
    group_starts[1:] = np.sort(picked_slots) * step  # distinct start times for groups

    # -----------------------------
    # 2) DISTINCT start position per skier *within* each group
    #    (but groups may reuse positions)
    # -----------------------------
    rng_pos = np.random.default_rng()
    group_positions = []
    for g in range(num_groups):
        # size of this group (last group may be smaller)
        g_start = g * group_size
        g_end = min((g + 1) * group_size, num_skiers)
        g_n = g_end - g_start

        # draw unique positions within the group
        # if you need positions in [1, M), there are (M-1) options
        if g_n > (M - 1):
            raise ValueError(f"Group {g} has {g_n} skiers but only {M - 1} unique positions available.")

        pos = rng_pos.choice(np.arange(5, (M-5)), size=g_n, replace=False)
        #print(pos)
        group_positions.append(pos)

    #print(group_positions)

    """# one start per group
    group_starts = np.random.randint(1, num_steps + 1, size=num_groups) * step
    group_starts[0] = 0  # force first group to start at 0"""
    for i in range(num_skiers):
        #start = i * stagger
        group_id = i // group_size
        within = i % group_size

        start = int(group_starts[group_id])
        level = sample_ability #random.choice(abilities)

        # position unique within the group

        pos = int(group_positions[group_id][within])
        #print(start, pos, level)
        #level = abilities[i % len(abilities)]  # simple assignment
        #level = random.choice(abilities)
        skiers.append(
            SkierCA(M=M, J=J, skier_id=i, start_step=start,
                    alpha_deg=alpha, ability_level=level,start_i=pos)
        )
    finished = [False]*num_skiers

    for step_num in range(steps):
        all_finished = True
        reserved = set()
        for idx, skier in enumerate(skiers):
            # Only step if skier is past their start_step
            if step_num >= skier.start_step and not finished[idx]:
                others = [s for j,s in enumerate(skiers) if j != idx and step_num >= s.start_step]
                cont = skier.step(others, reserved)
                finished[idx] = not cont
            # Skier hasn’t started yet
            elif step_num < skier.start_step:
                skier.active = False

            all_finished &= finished[idx]

        if all_finished:
            break

    return skiers


if __name__ == "__main__":


    dataset_root = "C:/Users/Saydez/OneDrive/Documents/Phd/Codes/tools/slope_track" #slope_track



    n_runs = 50 #30 #20
    views = [1,2,3,4,5,6,7,8] #, #5, 6, 7, 8 1, 2, 3, 4
    num_views = len(views)
    """num_skiers = 30 #50 #300

    M, J = 200, 600 #600
    img_w, img_h = 1920, 1080
    sigma = 9
    alpha = 18
    fps = 30

    steps_total = 3000
    stagger = 2500"""



    # Resolution options
    img_sizes = [
        (1920, 1080),
        (1920, 1088),
        (1280, 720),
    ]

    # Alpha ranges
    alpha_range = (12, 25)  # inclusive range you wanted

    # FPS options
    fps_list = [30, 12, 15]

    rng = np.random.default_rng()


    start_seq_id=32

    for run in range(n_runs):
        print("\n" + "=" * 60)
        print(f"RUN {run + 1}/{n_runs}")

        M = int(rng.integers(200, 301))  # width cells
        J = int(rng.integers(500, 601))  # length cells

        #num_skiers = int(rng.integers(30, 120))
        num_skiers = 10
        steps_total = int(rng.integers(1000, 1001))
        stagger = steps_total - 500

        img_w, img_h, alpha, fps = sample_generation_config(img_sizes, alpha_range, fps_list)
        alpha = int(alpha)
        #img_w, img_h = 1920,1080
        #fps = 30
        #J=300
        #M=200

        fov = rng.integers(1000, 1500)
        fx, fy = fov, fov


        print(f"Steps: {steps_total}, M: {M}, J: {J}, Angle: {alpha}, Img Sizes: {img_w, img_h}, FPS:{fps}, Num skiers: {num_skiers}, FOV: {fy, fx}")

        unique_positions = generate_unique_positions(M=M, num_skiers=num_skiers)
        skiers = run_multi_skier(
            num_skiers=num_skiers,
            start_i=unique_positions,
            M=M, J=J,
            steps=steps_total,
            stagger=stagger,
            alpha=alpha

        )
        # plot_trapezoidal_automaton(skiers, M=M, J=J, alpha_deg=alpha, sigma=sigma)
        # plot_skiers_clashing(skiers, M=M, J=J)
        #save_frames_and_video(skiers)
        check_cell_collisions(skiers)
        #plot_skiers_trajectories(skiers, M=M, J=J, sigma=9)
        # plot_trapezoidal_automaton(skiers, M=M, J=J, alpha_deg=alpha, sigma=sigma)
        # plot_traj_projected(skiers, M, J, img_w, img_h)
        #plot_simulation_curvature(skiers)
        # #plot_speed_time(skiers)
        # #plot_space_time(skiers)
        # #plot_lateral_time(skiers)
        # plot_space_time_2d(skiers)

        cell_size = 0.5
        alpha_rad = np.deg2rad(alpha)
        scale_top = 1 + np.tan(alpha_rad)
        slope_width_bottom_m = M * cell_size
        slope_width_top_m = M * scale_top * cell_size
        slope_length_m = J * cell_size

        #split = choose_split(rng.random(), train=0.7, val=0.2, test=0.1)




        for v_idx, view in enumerate(views):
            # Project for this view
            all_bboxes, proj_trajs, _ = project_trajectories_pinhole(
                skiers, img_w, img_h,
                slope_length_m=slope_length_m,
                slope_width_top_m=slope_width_top_m,
                slope_width_bottom_m=slope_width_bottom_m,
                slope_angle_deg=alpha,
                view=view,
                video_fps=fps, fx=fx, fy=fy

            )

            start_seq_id = start_seq_id + 1
            seq_name = f"slope_track{start_seq_id:06d}"

            seq_dir = os.path.join("gen", "train", seq_name)
            os.makedirs(seq_dir, exist_ok=True)

            # Write seqinfo.ini
            write_seqinfo_ini(
                seq_dir=seq_dir,
                seq_name=seq_name,
                fps=fps,
                seqlength=steps_total,
                img_w=img_w,
                img_h=img_h,
                imdir="img1",
                imext=".jpg"
            )

            os.makedirs(os.path.join(seq_dir, "img1"), exist_ok=True)

            # Save annotations
            gt_dir = os.path.join(seq_dir, "gt")
            os.makedirs(gt_dir, exist_ok=True)
            out_path = os.path.join(gt_dir, "gt.txt")

            save_all_boxes_mot(all_bboxes, out_path, frame_base=1, id_base=1, img_w=img_w,img_h=img_h )
            print(f"[Saved MOT gt for view {view} -> {out_path}")

            #plot_bboxes_with_trajectories(all_bboxes, proj_trajs, img_w=img_w, img_h=img_h)

