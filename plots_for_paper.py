import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ------------------------------------------------------------
# Terrain / slope
# ------------------------------------------------------------
def slope_height(x, y, Lx, Ly, z_top=35.0, z_bottom=0.0):
    """
    Simple planar ski slope descending along +y.
    """
    return z_top + (z_bottom - z_top) * (y / Ly)


def make_slope_mesh(Lx=80, Ly=180, nx=80, ny=120):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)
    Z = slope_height(X, Y, Lx, Ly)
    return X, Y, Z


# ------------------------------------------------------------
# Example skier trajectories on the slope
# ------------------------------------------------------------
def make_skier_trajectory(x0, amp, phase, Lx=80, Ly=180, n=120):
    """
    A simple carving-like path descending the slope.
    """
    y = np.linspace(10, Ly - 10, n)
    x = x0 + amp * np.sin(2 * np.pi * y / 55.0 + phase)

    # keep within slope bounds
    x = np.clip(x, 4, Lx - 4)

    z = slope_height(x, y, Lx, Ly)
    return x, y, z


# ------------------------------------------------------------
# Camera positions for your 8 views
# ------------------------------------------------------------
def get_camera_positions(Lx, Ly):
    return {
        1: np.array([Lx / 2.0, Ly + 20.0, -10.0]),    # bottom/front
        2: np.array([Lx / 2.0, -20.0, 45.0]),         # top

        # swapped
        3: np.array([Lx + 20.0, Ly / 2.0, 30.0]),     # was view 4
        4: np.array([-20.0, Ly / 2.0, 30.0]),         # was view 3

        # swapped
        5: np.array([Lx + 20.0, -10.0, 40.0]),        # was view 6
        6: np.array([-20.0, -10.0, 40.0]),            # was view 5

        # swapped
        7: np.array([Lx + 20.0, Ly + 15.0, 5.0]),     # was view 8
        8: np.array([-20.0, Ly + 15.0, 5.0]),         # was view 7
    }


def get_slope_target(Lx, Ly):
    """
    Mid-slope target point on surface.
    """
    tx = Lx / 2.0
    ty = Ly / 2.0
    tz = slope_height(tx, ty, Lx, Ly)
    return np.array([tx, ty, tz])


# ------------------------------------------------------------
# 3D plot
# ------------------------------------------------------------
def draw_scene(ax, Lx=80, Ly=180):
    X, Y, Z = make_slope_mesh(Lx=Lx, Ly=Ly)
    ax.plot_surface(X, Y, Z, alpha=0.7, linewidth=0, antialiased=True)

    # Example trajectories
    traj_specs = [
        (20, 7, 0.0),
        (40, 9, 0.9),
        (60, 6, 1.7),
    ]
    for x0, amp, phase in traj_specs:
        x, y, z = make_skier_trajectory(x0, amp, phase, Lx=Lx, Ly=Ly)
        ax.plot(x, y, z + 0.15, linewidth=2)

    # Cameras
    cams = get_camera_positions(Lx, Ly)
    target = get_slope_target(Lx, Ly)

    for view_id, C in cams.items():
        ax.scatter(C[0], C[1], C[2], s=45)
        ax.text(C[0], C[1], C[2] + 2.0, f"V{view_id}", fontsize=9)
        ax.plot(
            [C[0], target[0]],
            [C[1], target[1]],
            [C[2], target[2]],
            linestyle="--",
            linewidth=1,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D slope with camera viewpoints")
    ax.set_xlim(-25, Lx + 25)
    ax.set_ylim(-25, Ly + 25)
    ax.set_zlim(-15, 50)
    ax.view_init(elev=25, azim=-60)


# ------------------------------------------------------------
# Load example view images
# ------------------------------------------------------------
def load_image_or_placeholder(path, title="Missing image"):
    if path is not None and Path(path).exists():
        return plt.imread(path), False
    # placeholder
    img = np.ones((240, 320, 3), dtype=float)
    img[:] = 0.95
    return img, True


# ------------------------------------------------------------
# Main figure
# ------------------------------------------------------------
def plot_slope_with_views(
    image_paths=None,
    output_path="slope_views_overview.png",
    Lx=80,
    Ly=180,
):
    """
    image_paths: dict like {1: 'view1.png', 3: 'view3.png', 8: 'view8.png'}
    """
    if image_paths is None:
        image_paths = {}

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.2, 1.0])

    # Big 3D scene
    ax3d = fig.add_subplot(gs[0, :], projection="3d")
    draw_scene(ax3d, Lx=Lx, Ly=Ly)

    # Example view panels
    example_views = [1, 3, 6, 8]
    for i, view_id in enumerate(example_views):
        ax = fig.add_subplot(gs[1, i])
        img, missing = load_image_or_placeholder(
            image_paths.get(view_id), title=f"View {view_id}"
        )
        ax.imshow(img)
        ax.set_title(f"View {view_id}")
        ax.axis("off")
        if missing:
            ax.text(
                0.5, 0.5, f"Add image\nfor View {view_id}",
                ha="center", va="center", transform=ax.transAxes, fontsize=11
            )

    fig.suptitle("3D slope geometry and example camera views", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Replace these with your actual saved screenshots if available
    image_paths = {
        1: "images/view1.png",
        3: "images/view3.png",
        6: "images/view6.png",
        8: "images/view8.png",
    }

    plot_slope_with_views(
        image_paths=image_paths,
        output_path="slope_views_overview.png",
        Lx=80,
        Ly=180,
    )