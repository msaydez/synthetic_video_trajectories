import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# --------------------------
# Camera & body parameters
# --------------------------
CELL_SIZE = 1.0          # meters per CA cell
SKIER_HEIGHT = 1.7       # meters
SKIER_WIDTH = 0.5        # meters
FOCAL_LENGTH = 800       # pixels
IMG_W, IMG_H = 1280, 720
CAMERA_HEIGHT = 2.0      # meters above snow at bottom

# Maximum uphill depth (for normalized image projection)
Z_START = None

# --------------------------
# Directions & angles
# --------------------------
DIRECTIONS = {
    -2: (-1, 0),   # left
    -1: (-1, -1),  # left-down
     0: (0, -1),   # straight-down
     1: (1, -1),   # right-down
     2: (1, 0)     # right
}

COS_THETA = {
    -2: 0.0,
    -1: np.cos(np.deg2rad(45)),
     0: 1.0,
     1: np.cos(np.deg2rad(45)),
     2: 0.0
}

# --------------------------
# Skier CA with improved lateral motion
# --------------------------
class SkierCA:
    def __init__(self, M=100, J=200, dt=0.04, alpha_deg=18, w1=0.12, w4=0.2, terrain_sigma=2.0):
        self.M = M
        self.J = J
        self.dt = dt

        # Start at center top
        self.i = M // 2
        self.j = J - 1

        self.delta_prev = 0
        self.v = 5

        # Resistance parameters
        self.mu_straight = 0.04
        self.mu_turn = 0.08
        self.c_straight = 0.015
        self.c_turn = 0.035

        # Physics
        self.g = 9.81
        self.alpha = np.deg2rad(alpha_deg)

        # CA transfer factors
        self.w1 = w1       # slope bias
        self.w3 = 0.1      # boundary sensitivity
        self.w4 = w4       # curvature sensitivity
        self.w5 = 0.3      # air resistance
        self.gamma = 2     # inertia

        # --------------------------
        # Generate random terrain and compute curvature
        # --------------------------
        height_map = np.random.rand(M, J)
        height_map = gaussian_filter(height_map, sigma=terrain_sigma)
        self.kappa_map = np.zeros_like(height_map)
        for i in range(1, M-1):
            for j in range(1, J-1):
                self.kappa_map[i,j] = (height_map[i+1,j] + height_map[i-1,j] +
                                       height_map[i,j+1] + height_map[i,j-1] -
                                       4*height_map[i,j])

        # Initialize trajectory: store i, j, v, delta, cx, cy, w, h
        cx, cy, w, h = self.compute_bbox()
        self.traj = [(self.i, self.j, self.v, self.delta_prev, cx, cy, w, h)]

    # --------------------------
    # Transfer factors
    # --------------------------
    def fslope(self, delta):
        if delta == 0:
            return 1 - np.exp(-self.w1 * np.sin(self.alpha))
        else:
            return (1 - np.exp(-self.w1 * np.sin(self.alpha))) * 0.6

    def fboundary(self, i_p):
        d = min(i_p, self.M - i_p) + 1e-6
        return np.exp(-self.w3 / d)

    def fcurve(self, i_p, j_p):
        kappa = self.kappa_map[i_p, j_p]
        return np.exp(-self.w4 * kappa)

    def fair(self):
        return np.exp(-self.w5 * self.v**2)

    def inertia(self, delta):
        return np.exp(-self.gamma * abs(delta - self.delta_prev))

    # --------------------------
    # Transition probabilities
    # --------------------------
    def transition_probabilities(self):
        admissible = list(DIRECTIONS.keys())
        P_tilde = {}
        for d in admissible:
            di, dj = DIRECTIONS[d]
            i_p = self.i + di
            j_p = self.j + dj
            if i_p < 0 or i_p >= self.M or j_p < 0:
                continue
            P_tilde[d] = (self.fslope(d) *
                          self.fboundary(i_p) *
                          self.fcurve(i_p, j_p) *
                          self.fair() *
                          self.inertia(d) *
                          (COS_THETA[d]+0.1))
            P_tilde[d] *= random.uniform(0.85,1.15)
        if not P_tilde:
            return None
        Z = sum(P_tilde.values())
        return {d: P_tilde[d]/Z for d in P_tilde}

    # --------------------------
    # Update speed
    # --------------------------
    def update_speed(self, delta_new):
        turning = delta_new != self.delta_prev
        mu = self.mu_turn if turning else self.mu_straight
        c = self.c_turn if turning else self.c_straight
        cos_theta = COS_THETA[delta_new]
        a = self.g * np.sin(self.alpha) * cos_theta - mu * self.g * np.cos(self.alpha) - c * self.v**2
        self.v = max(0.1, self.v + a * self.dt)

    # --------------------------
    # Relative depth from slope
    # --------------------------
    def compute_depth(self, j):
        # Distance from camera at bottom
        return j * CELL_SIZE

    # --------------------------
    # Compute bounding box
    # --------------------------
    def compute_bbox(self):
        global Z_START

        X = (self.i - self.M / 2) * CELL_SIZE
        Z = self.compute_depth(self.j)

        if Z_START is None:
            Z_START = Z

        # --------------------------
        # IMAGE PROJECTION
        # --------------------------
        cx = IMG_W / 2 + FOCAL_LENGTH * X / (Z + 1e-6)

        depth_norm = 1.0 - (Z / Z_START)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        cy_bottom = depth_norm * IMG_H

        # --------------------------
        # Bounding box size
        # --------------------------
        posture = min(1.0, abs(self.delta_prev) / 2)
        H_eff = SKIER_HEIGHT * (1 - 0.3 * posture)
        W_eff = SKIER_WIDTH * (1 + 0.5 * posture)

        scale = FOCAL_LENGTH / (Z + 1e-6)
        h = scale * H_eff
        w = scale * W_eff

        cy = cy_bottom - h / 2
        return cx, cy, w, h

    # --------------------------
    # Step
    # --------------------------
    def step(self):
        probs = self.transition_probabilities()
        if probs is None:
            return False
        delta_new = random.choices(list(probs.keys()), weights=probs.values())[0]
        self.update_speed(delta_new)
        di, dj = DIRECTIONS[delta_new]
        self.i += di
        self.j += dj
        self.delta_prev = delta_new

        cx, cy, w, h = self.compute_bbox()
        self.traj.append((self.i, self.j, self.v, delta_new, cx, cy, w, h))
        return self.j > 0

    # --------------------------
    # Run simulation
    # --------------------------
    def run(self, max_steps=600):
        for _ in range(max_steps):
            if not self.step():
                break
        return self.traj

# --------------------------
# Smooth trajectory
# --------------------------
def smooth_trajectory(mean_i, mean_j, sigma=3):
    smooth_i = gaussian_filter1d(mean_i, sigma=sigma, mode="nearest")
    smooth_j = gaussian_filter1d(mean_j, sigma=sigma, mode="nearest")
    return smooth_i, smooth_j

# --------------------------
# Plot mean trajectory
# --------------------------
def plot_mean_trajectory(mean_i, mean_j, sigma=3):
    smooth_i, smooth_j = smooth_trajectory(mean_i, mean_j, sigma)
    plt.figure(figsize=(10,6))
    plt.plot(mean_i, mean_j, color="gray", lw=1, alpha=0.5, label="Raw mean trajectory")
    plt.plot(smooth_i, smooth_j, color="red", lw=3, label="Smoothed trajectory")
    plt.xlim(0, 100)
    plt.ylim(0, 200)
    plt.xlabel("Lateral position (cells)")
    plt.ylabel("Downhill position (cells)")
    plt.title("Mean Skier Trajectory (Gaussian Smoothed)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------
# Run multiple simulations
# --------------------------
def run_mean_trajectory(runs=1, M=100, J=200, w1=0.8, w4=0.6, terrain_sigma=2.0):
    all_i, all_j, all_cx, all_cy, all_w, all_h = [], [], [], [], [], []

    for r in range(runs):
        random.seed(r)
        skier = SkierCA(M=M, J=J, w1=w1, w4=w4, terrain_sigma=terrain_sigma)
        traj = skier.run()

        all_i.append([p[0] for p in traj])
        all_j.append([p[1] for p in traj])
        all_cx.append([p[4] for p in traj])
        all_cy.append([p[5] for p in traj])
        all_w.append([p[6] for p in traj])
        all_h.append([p[7] for p in traj])

    max_len = max(len(t) for t in all_i)
    def pad_array(arr):
        arr_padded = np.zeros((runs, max_len))
        for idx in range(runs):
            pad_len = max_len - len(arr[idx])
            arr_padded[idx] = arr[idx] + [arr[idx][-1]]*pad_len
        return arr_padded

    mean_i = pad_array(all_i).mean(axis=0)
    mean_j = pad_array(all_j).mean(axis=0)
    mean_cx = pad_array(all_cx).mean(axis=0)
    mean_cy = pad_array(all_cy).mean(axis=0)
    mean_w = pad_array(all_w).mean(axis=0)
    mean_h = pad_array(all_h).mean(axis=0)

    return mean_i, mean_j, mean_cx, mean_cy, mean_w, mean_h, max_len

def plot_bboxes(cx_list, cy_list, w_list, h_list, num_boxes=10):
    plt.figure(figsize=(12, 7))
    plt.xlim(0, IMG_W)
    plt.ylim(IMG_H, 0)  # y goes top->bottom
    plt.gca().set_aspect('equal')
    plt.title(f"First {num_boxes} simulated skier bounding boxes")
    plt.xlabel("Image X (pixels)")
    plt.ylabel("Image Y (pixels)")

    # Draw background lines (slope grid)
    for y in np.linspace(0, IMG_H, 20):
        plt.axhline(y=y, color='lightgray', linewidth=0.5)
    for x in np.linspace(0, IMG_W, 20):
        plt.axvline(x=x, color='lightgray', linewidth=0.5)

    for k in range(num_boxes):
        cx, cy, w, h = cx_list[k], cy_list[k], w_list[k], h_list[k]
        rect = plt.Rectangle((cx - w/2, cy - h/2), w, h, edgecolor='red', facecolor='none', linewidth=2)
        plt.gca().add_patch(rect)
        plt.plot(cx, cy, 'bo')  # center point

    plt.tight_layout()
    plt.show()

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    mean_i, mean_j, mean_cx, mean_cy, mean_w, mean_h, max_len = run_mean_trajectory(runs=1)
    plot_mean_trajectory(mean_i, mean_j)

    # Print first 10 bounding boxes
    print("First 10 bounding boxes (cx, cy, w, h):")
    for k in range(200):
        print(f"{k}: ({mean_cx[k]:.1f}, {mean_cy[k]:.1f}, {mean_w[k]:.1f}, {mean_h[k]:.1f})")

    plot_bboxes(mean_cx, mean_cy, mean_w, mean_h, num_boxes=200)
    