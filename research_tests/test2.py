import numpy as np
from typing import Tuple, List


class FluidSimulator:
    def __init__(self, scene_num: int = 1):
        # Constants
        self.U_FIELD = 0
        self.V_FIELD = 1
        self.S_FIELD = 2

        # Simulation parameters
        self.gravity = -9.81
        self.dt = 1.0 / 120.0
        self.num_iters = 100
        self.frame_nr = 0
        self.over_relaxation = 1.9
        self.scene_num = scene_num

        # Initialize fluid grid
        self.setup_scene(scene_num)

        # Results storage [x, y, t]
        self.results = []

    def setup_scene(self, scene_num: int):
        """Initialize the simulation scene based on scenario number"""
        self.scene_num = scene_num
        self.obstacle_radius = 0.15
        self.over_relaxation = 1.9
        self.dt = 1.0 / 60.0
        self.num_iters = 40

        res = 100
        if scene_num == 0:
            res = 50
        elif scene_num == 3:
            res = 200

        domain_height = 1.0
        domain_width = domain_height  # Assuming square domain for simplicity
        h = domain_height / res

        num_x = int(domain_width / h)
        num_y = int(domain_height / h)
        density = 1000.0

        self.initialize_fluid(density, num_x, num_y, h)

        n = self.fluid.num_y

        if scene_num == 0:  # tank
            for i in range(self.fluid.num_x):
                for j in range(self.fluid.num_y):
                    s = 1.0  # fluid
                    if i == 0 or i == self.fluid.num_x - 1 or j == 0:
                        s = 0.0  # solid
                    self.fluid.s[i * n + j] = s
            self.gravity = -9.81

        elif scene_num == 1 or scene_num == 3:  # vortex shedding
            in_vel = 2.0
            for i in range(self.fluid.num_x):
                for j in range(self.fluid.num_y):
                    s = 1.0  # fluid
                    if i == 0 or j == 0 or j == self.fluid.num_y - 1:
                        s = 0.0  # solid
                    self.fluid.s[i * n + j] = s

                    if i == 1:
                        self.fluid.u[i * n + j] = in_vel

            pipe_h = 0.1 * self.fluid.num_y
            min_j = int(0.5 * self.fluid.num_y - 0.5 * pipe_h)
            max_j = int(0.5 * self.fluid.num_y + 0.5 * pipe_h)

            for j in range(min_j, max_j):
                self.fluid.m[j] = 0.0

            self.set_obstacle(0.4, 0.5, True)
            self.gravity = 0.0

            if scene_num == 3:
                self.dt = 1.0 / 120.0
                self.num_iters = 100

        elif scene_num == 2:  # paint
            self.gravity = 0.0
            self.over_relaxation = 1.0
            self.obstacle_radius = 0.1

    def initialize_fluid(self, density: float, num_x: int, num_y: int, h: float):
        """Initialize the fluid grid"""
        self.fluid = Fluid(density, num_x, num_y, h)

    def set_obstacle(self, x: float, y: float, reset: bool):
        """Set obstacle in the fluid"""
        vx, vy = 0.0, 0.0
        if not reset:
            vx = (x - self.obstacle_x) / self.dt
            vy = (y - self.obstacle_y) / self.dt

        self.obstacle_x = x
        self.obstacle_y = y
        r = self.obstacle_radius
        f = self.fluid
        n = f.num_y
        cd = np.sqrt(2) * f.h

        for i in range(1, f.num_x - 2):
            for j in range(1, f.num_y - 2):
                f.s[i * n + j] = 1.0

                dx = (i + 0.5) * f.h - x
                dy = (j + 0.5) * f.h - y

                if dx * dx + dy * dy < r * r:
                    f.s[i * n + j] = 0.0
                    if self.scene_num == 2:
                        f.m[i * n + j] = 0.5 + 0.5 * np.sin(0.1 * self.frame_nr)
                    else:
                        f.m[i * n + j] = 1.0
                    f.u[i * n + j] = vx
                    f.u[(i + 1) * n + j] = vx
                    f.v[i * n + j] = vy
                    f.v[i * n + j + 1] = vy

    def run_simulation(self, steps: int = 1000) -> np.ndarray:
        """Run the simulation for specified number of steps and return results"""
        for _ in range(steps):
            self.fluid.simulate(self.dt, self.gravity, self.num_iters)
            self.frame_nr += 1

            # Store the current state (u, v, m)
            current_state = np.zeros((3, self.fluid.num_x, self.fluid.num_y))
            n = self.fluid.num_y
            for i in range(self.fluid.num_x):
                for j in range(self.fluid.num_y):
                    current_state[0, i, j] = self.fluid.u[i * n + j]
                    current_state[1, i, j] = self.fluid.v[i * n + j]
                    current_state[2, i, j] = self.fluid.m[i * n + j]

            self.results.append(current_state)

        # Convert to numpy array with shape (steps, 3, num_x, num_y)
        return np.array(self.results)

    def save_results(self, filename: str):
        """Save the simulation results to a file"""
        results_array = self.run_simulation()
        np.save(filename, results_array)


class Fluid:
    def __init__(self, density: float, num_x: int, num_y: int, h: float):
        # Constants
        self.U_FIELD = 0
        self.V_FIELD = 1
        self.S_FIELD = 2

        self.density = density
        self.num_x = num_x + 2
        self.num_y = num_y + 2
        self.num_cells = self.num_x * self.num_y
        self.h = h

        # Initialize arrays
        self.u = np.zeros(self.num_cells, dtype=np.float32)
        self.v = np.zeros(self.num_cells, dtype=np.float32)
        self.new_u = np.zeros(self.num_cells, dtype=np.float32)
        self.new_v = np.zeros(self.num_cells, dtype=np.float32)
        self.p = np.zeros(self.num_cells, dtype=np.float32)
        self.s = np.zeros(self.num_cells, dtype=np.float32)
        self.m = np.zeros(self.num_cells, dtype=np.float32)
        self.new_m = np.zeros(self.num_cells, dtype=np.float32)
        self.m.fill(1.0)

    def integrate(self, dt: float, gravity: float):
        """Apply gravity to the velocity field"""
        n = self.num_y
        for i in range(1, self.num_x):
            for j in range(1, self.num_y - 1):
                if self.s[i * n + j] != 0.0 and self.s[i * n + j - 1] != 0.0:
                    self.v[i * n + j] += gravity * dt

    def solve_incompressibility(self, num_iters: int, dt: float):
        """Solve the incompressibility condition using pressure projection"""
        n = self.num_y
        cp = self.density * self.h / dt

        for _ in range(num_iters):
            for i in range(1, self.num_x - 1):
                for j in range(1, self.num_y - 1):
                    if self.s[i * n + j] == 0.0:
                        continue

                    s = self.s[i * n + j]
                    sx0 = self.s[(i - 1) * n + j]
                    sx1 = self.s[(i + 1) * n + j]
                    sy0 = self.s[i * n + j - 1]
                    sy1 = self.s[i * n + j + 1]
                    s = sx0 + sx1 + sy0 + sy1
                    if s == 0.0:
                        continue

                    div = (
                        self.u[(i + 1) * n + j]
                        - self.u[i * n + j]
                        + self.v[i * n + j + 1]
                        - self.v[i * n + j]
                    )

                    p = -div / s
                    p *= 1.9  # Over-relaxation
                    self.p[i * n + j] += cp * p

                    self.u[i * n + j] -= sx0 * p
                    self.u[(i + 1) * n + j] += sx1 * p
                    self.v[i * n + j] -= sy0 * p
                    self.v[i * n + j + 1] += sy1 * p

    def extrapolate(self):
        """Extrapolate velocity field to boundaries"""
        n = self.num_y
        for i in range(self.num_x):
            self.u[i * n + 0] = self.u[i * n + 1]
            self.u[i * n + self.num_y - 1] = self.u[i * n + self.num_y - 2]

        for j in range(self.num_y):
            self.v[0 * n + j] = self.v[1 * n + j]
            self.v[(self.num_x - 1) * n + j] = self.v[(self.num_x - 2) * n + j]

    def sample_field(self, x: float, y: float, field: int) -> float:
        """Sample a field (u, v, or m) at a given position"""
        n = self.num_y
        h = self.h
        h1 = 1.0 / h
        h2 = 0.5 * h

        x = max(min(x, self.num_x * h), h)
        y = max(min(y, self.num_y * h), h)

        dx, dy = 0.0, 0.0

        if field == self.U_FIELD:
            f = self.u
            dy = h2
        elif field == self.V_FIELD:
            f = self.v
            dx = h2
        elif field == self.S_FIELD:
            f = self.m
            dx = h2
            dy = h2

        x0 = min(int((x - dx) * h1), self.num_x - 1)
        tx = ((x - dx) - x0 * h) * h1
        x1 = min(x0 + 1, self.num_x - 1)

        y0 = min(int((y - dy) * h1), self.num_y - 1)
        ty = ((y - dy) - y0 * h) * h1
        y1 = min(y0 + 1, self.num_y - 1)

        sx = 1.0 - tx
        sy = 1.0 - ty

        val = (
            sx * sy * f[x0 * n + y0]
            + tx * sy * f[x1 * n + y0]
            + tx * ty * f[x1 * n + y1]
            + sx * ty * f[x0 * n + y1]
        )

        return val

    def avg_u(self, i: int, j: int) -> float:
        """Average u velocity at cell center"""
        n = self.num_y
        u = (
            self.u[i * n + j - 1]
            + self.u[i * n + j]
            + self.u[(i + 1) * n + j - 1]
            + self.u[(i + 1) * n + j]
        ) * 0.25
        return u

    def avg_v(self, i: int, j: int) -> float:
        """Average v velocity at cell center"""
        n = self.num_y
        v = (
            self.v[(i - 1) * n + j]
            + self.v[i * n + j]
            + self.v[(i - 1) * n + j + 1]
            + self.v[i * n + j + 1]
        ) * 0.25
        return v

    def advect_vel(self, dt: float):
        """Advect velocity field"""
        self.new_u[:] = self.u
        self.new_v[:] = self.v

        n = self.num_y
        h = self.h
        h2 = 0.5 * h

        for i in range(1, self.num_x):
            for j in range(1, self.num_y):
                # u component
                if (
                    self.s[i * n + j] != 0.0
                    and self.s[(i - 1) * n + j] != 0.0
                    and j < self.num_y - 1
                ):
                    x = i * h
                    y = j * h + h2
                    u = self.u[i * n + j]
                    v = self.avg_v(i, j)
                    x = x - dt * u
                    y = y - dt * v
                    u = self.sample_field(x, y, self.U_FIELD)
                    self.new_u[i * n + j] = u

                # v component
                if (
                    self.s[i * n + j] != 0.0
                    and self.s[i * n + j - 1] != 0.0
                    and i < self.num_x - 1
                ):
                    x = i * h + h2
                    y = j * h
                    u = self.avg_u(i, j)
                    v = self.v[i * n + j]
                    x = x - dt * u
                    y = y - dt * v
                    v = self.sample_field(x, y, self.V_FIELD)
                    self.new_v[i * n + j] = v

        self.u[:] = self.new_u
        self.v[:] = self.new_v

    def advect_smoke(self, dt: float):
        """Advect smoke density field"""
        self.new_m[:] = self.m

        n = self.num_y
        h = self.h
        h2 = 0.5 * h

        for i in range(1, self.num_x - 1):
            for j in range(1, self.num_y - 1):
                if self.s[i * n + j] != 0.0:
                    u = (self.u[i * n + j] + self.u[(i + 1) * n + j]) * 0.5
                    v = (self.v[i * n + j] + self.v[i * n + j + 1]) * 0.5
                    x = i * h + h2 - dt * u
                    y = j * h + h2 - dt * v
                    self.new_m[i * n + j] = self.sample_field(x, y, self.S_FIELD)

        self.m[:] = self.new_m

    def simulate(self, dt: float, gravity: float, num_iters: int):
        """Run one simulation step"""
        self.integrate(dt, gravity)
        self.p.fill(0.0)
        self.solve_incompressibility(num_iters, dt)
        self.extrapolate()
        self.advect_vel(dt)
        self.advect_smoke(dt)


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize


def load_and_animate(filename: str, fps: int = 30, save_animation: bool = False):
    """
    Load the simulation results and create an animation.

    Args:
        filename: Path to the .npy file with simulation results
        fps: Frames per second for the animation
        save_animation: If True, save as MP4 file
    """
    # Load the simulation data
    data = np.load(filename)
    num_frames, _, num_x, num_y = data.shape

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Fluid Simulation Animation", fontsize=16)

    # Prepare velocity magnitude plot
    velocity_magnitude = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
    vmin, vmax = velocity_magnitude.min(), velocity_magnitude.max()
    norm_vel = Normalize(vmin=vmin, vmax=vmax)
    vel_plot = ax1.imshow(
        velocity_magnitude[0], cmap="viridis", norm=norm_vel, origin="lower"
    )
    fig.colorbar(vel_plot, ax=ax1, label="Velocity Magnitude")
    ax1.set_title("Velocity Field")

    # Prepare velocity vector plot
    # Subsample the grid for clearer vector display
    step = max(1, num_x // 15)
    x, y = np.meshgrid(np.arange(0, num_x, step), np.arange(0, num_y, step))
    quiver = ax2.quiver(
        x,
        y,
        data[0, 0, ::step, ::step],
        data[0, 1, ::step, ::step],
        scale=20,
        scale_units="inches",
    )
    ax2.set_title("Velocity Vectors")

    # Prepare smoke density plot
    smoke = data[:, 2]
    vmin, vmax = smoke.min(), smoke.max()
    norm_smoke = Normalize(vmin=vmin, vmax=vmax)
    smoke_plot = ax3.imshow(smoke[0], cmap="plasma", norm=norm_smoke, origin="lower")
    fig.colorbar(smoke_plot, ax=ax3, label="Smoke Density")
    ax3.set_title("Smoke Density")

    # Adjust layout
    plt.tight_layout()

    def update(frame):
        """Update function for animation"""
        # Update velocity magnitude
        vel_mag = velocity_magnitude[frame]
        vel_plot.set_array(vel_mag)

        # Update velocity vectors
        U = data[frame, 0, ::step, ::step]
        V = data[frame, 1, ::step, ::step]
        quiver.set_UVC(U, V)

        # Update smoke density
        smoke_plot.set_array(smoke[frame])

        # Update title with frame number
        fig.suptitle(
            f"Fluid Simulation Animation - Frame {frame}/{num_frames}", fontsize=16
        )

        return vel_plot, quiver, smoke_plot

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        frames=min(500, num_frames),  # Limit to 500 frames for performance
        interval=1000 / fps,
        blit=False,
    )

    if save_animation:
        # Save as MP4 (requires ffmpeg)
        anim.save("fluid_simulation_animation.mp4", writer="ffmpeg", fps=fps, dpi=200)
        print("Animation saved as fluid_simulation_animation.mp4")

    plt.show()
    return anim


# Example usage
if __name__ == "__main__":

    simulator = FluidSimulator(scene_num=1)
    simulator.save_results("fluid_simulation_results.npy")

    animation = load_and_animate(
        "fluid_simulation_results.npy", fps=30, save_animation=True
    )
