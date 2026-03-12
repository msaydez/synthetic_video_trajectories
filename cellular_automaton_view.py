import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

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
    def __init__(self,  M=100, J=200, dt=0.04, alpha_deg=18, terrain_sigma=2.5):
        self.M, self.J = M, J
        self.dt = dt

        self.turn_steps = 0

        # Start at center top
        self.i = M // 2
        self.j = J - 1
        
        self.dir_prev = None
        self.delta_prev = 0
        self.v = 6
        
        #self.mu = 0.002

        # Resistance parameters
        self.mu_straight = 0.002
        self.mu_turn = 0.006
        self.c_straight = 0.005
        self.c_turn = 0.007

        # Physics
        self.g = 9.81
        self.alpha = np.deg2rad(alpha_deg)

        # CA transfer factors
         #  (increase straighter downhill motion, decrease more exploration) slope bias (how strongly gravity biases motion downhill)
        self.w1 = 0.2    # slope bias
        # (increase strong avoidance of high-μ cells, decrease friction nearly irrelevant )
        self.w2 = 0.03      # local surface friction
        #(increase Strong “invisible wall” near edges, decrease Trajectories drift toward boundaries)
        self.w3 = 0.1    # boundary sensitivity
        # (increase strong avoidance of bumps, decrease terrain roughness ignored)
        self.w4 = 0.1     # curvature sensitivity
        #(increase High speed strongly discouraged, decrease skier accelerates freely)
        self.w5 = 0.01      # air resistance
        # (increase skier stays in that direction (behavioural) )
        self.gamma = 0.9 # inertia

        # Projection grid
        X, Y = np.meshgrid(np.arange(M), np.arange(J))
        scale = 1 + np.tan(self.alpha) * (Y / J)
        self.Xg = (X - M/2) * scale + M/2
        self.Yg = Y

        # Terrain curvature
        h = gaussian_filter(np.random.rand(M, J), sigma=terrain_sigma)
        self.kappa = np.zeros_like(h)
        for i in range(1, M-1):
            for j in range(1, J-1):
                self.kappa[i,j] = (
                    h[i+1,j] + h[i-1,j] + h[i,j+1] + h[i,j-1] - 4*h[i,j]
                )
        """h = np.zeros((M, J))           # flat height
        self.kappa = np.zeros_like(h)""" 

        #Surface Friction
        mu_min=0.002
        mu_max=0.012
        self.mu = np.random.uniform(mu_min, mu_max, size=(M, J))

        self.traj = [(self.i, self.j, self.v, self.delta_prev)]

    def row_bounds(self, j):
        scale = 1 + np.tan(self.alpha) * (j / self.J)
        width = self.M * scale

        i_left  = int(np.floor((self.M - width) / 2))
        i_right = int(np.ceil((self.M + width) / 2))

        return i_left, i_right

    # --------------------------
    # Transfer factors
    # --------------------------
    def fslope(self, delta):
        # Boost lateral moves by reducing slope bias for delta != 0
        if delta == 0:
            return 1 - np.exp(-self.w1 * np.sin(self.alpha))
        else:
            return (1 - np.exp(-self.w1 * np.sin(self.alpha)))

    def fboundary(self, i_p):
        d = min(i_p, self.M - i_p) + 1e-6
        return np.exp(-self.w3 / d)

    def fcurve(self, i_p, j_p):
        kappa = self.kappa[i_p, j_p]
        return np.exp(-self.w4 * kappa)

    def fair(self):
        return np.exp(-self.w5 * self.v**2)

    def inertia(self, delta):
        return np.exp(-self.gamma * abs(delta - self.delta_prev))

    def ffriction(self, i_p, j_p):
        mu_local = self.mu[i_p, j_p]
        return (np.exp(-self.w2 * mu_local))

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
            # Transition probability now includes cos θ to favor diagonal moves realistically
            P_tilde[d] = (self.fslope(d) *
                          self.fboundary(i_p) *
                          self.fcurve(i_p, j_p) *
                          self.fair() *
                          self.inertia(d) *
                          self.ffriction(i_p, j_p))# small bias to allow diagonal moves
            # Stochastic spread
            #P_tilde[d] *= random.uniform(0.85,1.15)
        if not P_tilde:
            return None
        Z = sum(P_tilde.values())
        return {d: P_tilde[d]/Z for d in P_tilde}

    
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
    # Update speed
    # --------------------------
    def update_speed(self, delta_new):
        d = np.array(DIRECTIONS[delta_new], dtype=float)
        #print(d)
        #d /= np.linalg.norm(d)
        d_curr = self.v * d
        theta_t = self.turning_angle(self.dir_prev, d_curr)
        #print(d, self.v, theta_t)
        turning = theta_t > 5.0
        if turning:
           self.turn_steps = 10  # apply turn penalty for next 3 steps
        if self.turn_steps > 0:
           mu = self.mu_turn
           c = self.c_turn
           self.turn_steps -= 1
        else:
           mu = self.mu_straight
           c = self.c_straight
        #mu = self.mu_turn if turning else self.mu_straight
        #c = self.c_turn if turning else self.c_straight
        cos_theta = COS_THETA[delta_new]
        a = self.g * np.sin(self.alpha) * cos_theta - mu * self.g * np.cos(self.alpha) - c * self.v**2
        self.v =  self.v + a * self.dt
        self.dir_prev = d_curr

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

        self.traj.append((self.i, self.j, self.v, delta_new))
        return self.j > 0

    def run(self, steps=600):
        for _ in range(steps):
            if not self.step():
                break
        return np.array(self.traj)

def project_trapezoid(i, j, M, J, alpha_deg):
    alpha = np.deg2rad(alpha_deg)
    scale = 1 + np.tan(alpha) * (j / J)
    x = (i - M / 2) * scale + M / 2
    y = j
    return x, y

# --------------------------
# Run multiple simulations and mean trajectory
# --------------------------
def run_mean_trajectory(runs=1, M=100, J=200, terrain_sigma=2.5):
    all_i = []
    all_j = []
    for r in range(runs):
        #random.seed(r)
        skier = SkierCA(M=M, J=J, terrain_sigma=terrain_sigma)
        traj = skier.run()
        plot_simulation_curvature(skier)
        all_i.append([p[0] for p in traj])
        all_j.append([p[1] for p in traj])
    
    proj_x = []
    proj_y = []

    for i, j, v, d in traj:
        x, y = project_trapezoid(i, j, M, J, alpha_deg=18)
        proj_x.append(x)
        proj_y.append(y)
    proj_traj = np.stack([np.array(proj_x), np.array(proj_y)], axis=1)

    """# Pad trajectories
    max_len = max(len(t) for t in all_i)
    i_array = np.zeros((runs, max_len))
    j_array = np.zeros((runs, max_len))
    for idx in range(runs):
        pad_len = max_len - len(all_i[idx])
        i_array[idx] = all_i[idx] + [all_i[idx][-1]]*pad_len
        j_array[idx] = all_j[idx] + [all_j[idx][-1]]*pad_len
    mean_i = i_array.mean(axis=0)
    mean_j = j_array.mean(axis=0)"""

    return traj[:,0:2], proj_traj

def smooth_trajectory(mean_i, mean_j, sigma=3):
    """
    Gaussian smoothing of trajectory coordinates over time.
    sigma: smoothing strength (in time steps)
    """
    smooth_i = gaussian_filter1d(mean_i, sigma=sigma, mode="nearest")
    smooth_j = gaussian_filter1d(mean_j, sigma=sigma, mode="nearest")
    return smooth_i, smooth_j

# --------------------------
# Plot automaton + trajectory
# --------------------------
def plot_mean_trajectory(traj, M=100, J=200, sigma=9):
    mean_i, mean_j = traj[:,0], traj[:,1]
    smooth_i, smooth_j = smooth_trajectory(mean_i, mean_j, sigma)

    corners = np.array([
    [0, 0],    # bottom-left
    [M, 0],    # bottom-right
    [M, J],    # top-right
    [0, J],    # top-left
    [0, 0]     # close loop
    ])

    plt.figure(figsize=(10, 6))
    plt.plot(corners[:,0], corners[:,1], "b--", lw=2, label="Automaton boundary")
    plt.plot(mean_i, mean_j, color="gray", lw=1, alpha=0.5, label="Raw mean trajectory")
    plt.plot(smooth_i, smooth_j, color="red", lw=3, label="Smoothed trajectory")

    plt.xlim(0, M)
    plt.ylim(0, J)
    plt.axis("equal")
    plt.xlabel("Lateral position (cells)")
    plt.ylabel("Downhill position (cells)")
    plt.title("Mean Skier Trajectory (Gaussian Smoothed)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --------------------------
# Plot trapezoidal automaton + trajectory
# --------------------------
def plot_trapezoidal_automaton(traj, M=100, J=200, alpha_deg=18):
    alpha = np.deg2rad(alpha_deg)

    # Automaton corners
    scale_top = 1 + np.tan(alpha)
    corners = np.array([
        [0, 0],
        [M, 0],
        [(M - M/2)*scale_top + M/2, J],
        [(0 - M/2)*scale_top + M/2, J],
        [0, 0]
    ])

    # Smooth trajectory
    x_s = gaussian_filter1d(traj[:,0], 9)
    y_s = gaussian_filter1d(traj[:,1], 9)

    plt.figure(figsize=(10,6))
    plt.plot(corners[:,0], corners[:,1], "b--", lw=2, label="Automaton boundary")
    plt.plot(traj[:,0], traj[:,1], color="gray", alpha=0.5)
    plt.plot(x_s, y_s, "r", lw=3, label="Trajectory")

    plt.xlabel("X (projected)")
    plt.ylabel("Y")
    plt.title("Trajectory on Trapezoidal Automaton Grid")
    plt.xlim(0, M)
    plt.ylim(0, J)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_example_grids():
    
    # --------------------------
    # Parameters
    # --------------------------
    M, J = 100, 200          # columns x rows
    alpha_deg = 18
    alpha_rad = np.deg2rad(alpha_deg)

    # --------------------------
    # Original rectangular grid
    # --------------------------
    X, Y = np.meshgrid(np.arange(M), np.arange(J))  # X: 0→M-1, Y: 0→J-1

    # --------------------------
    # Projected trapezoid grid
    # --------------------------
    scale = 1 + np.tan(alpha_rad) * (Y / J)   # widening with Y
    X_trap = (X - M/2) * scale + M/2          # expand about center
    Y_trap = Y

    # --------------------------
    # Zoom on rows 194-199, columns 0-3
    # --------------------------
    row_slice = slice(194, 200)  # rows 194,195,196,197,198,199
    col_slice = slice(0, 4)      # columns 0,1,2,3

    X_zoom = X[row_slice, col_slice]
    Y_zoom = Y[row_slice, col_slice]

    X_trap_zoom = X_trap[row_slice, col_slice]
    Y_trap_zoom = Y_trap[row_slice, col_slice]

    # --------------------------
    # Plot
    # --------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Original grid zoom
    axes[0].scatter(X_zoom.flatten(), Y_zoom.flatten(), s=80, c='blue', alpha=0.6)
    axes[0].set_title("Original Rectangular Grid (Zoom)")
    axes[0].set_xlabel("X (columns)")
    axes[0].set_ylabel("Y (rows)")
    axes[0].set_xlim(0, M)
    axes[0].set_ylim(0, J)
    axes[0].grid(True)
    axes[0].axis('equal')

    # Projected trapezoid zoom
    axes[1].scatter(X_trap_zoom.flatten(), Y_trap_zoom.flatten(), s=80, c='red', alpha=0.6)
    axes[1].set_title("Projected Trapezoidal Grid (Zoom)")
    axes[1].set_xlabel("X (projected)")
    axes[1].set_ylabel("Y (rows)")
    axes[1].set_xlim(0, M)
    axes[1].set_ylim(0, J)
    axes[1].grid(True)
    axes[1].axis('equal')

    plt.tight_layout()
    plt.show()


def plot_simulation_curvature(skier):
    """
    Plots the terrain curvature (kappa) generated by the simulation.

    Parameters:
    -----------
    skier : SkierCA
        An instance of the SkierCA class with kappa already computed.
    """
    kappa = skier.kappa

    plt.figure(figsize=(10, 6))
    im = plt.imshow(kappa.T, origin='lower', cmap='RdBu', extent=[0, skier.M, 0, skier.J])
    plt.colorbar(im, label='Curvature')
    plt.title("Terrain Curvature Map (Simulation)")
    plt.xlabel("X (columns)")
    plt.ylabel("Y (rows)")
    plt.grid(False)
    plt.show()



# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    traj, proj_traj= run_mean_trajectory(runs=1)
    plot_mean_trajectory(traj)
    plot_trapezoidal_automaton(proj_traj)
    #plot_example_grids()
    

    all_equal = np.array_equal(traj, proj_traj)
    print(all_equal)
    #print(proj_traj[:,0] - traj[:, 0])










