import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

cylinder_size = 0.3
min_value = 1e-10  # Минимальное значение для стабилизации


def flow_around_square_cylinder(geometry_size, Re, grid_size, time_steps=500, dt=0.005):
    """
    Решает задачу обтекания квадратного цилиндра с турбулентностью (модель k-epsilon)
    Возвращает координаты сетки и истории скоростей, функции тока
    """
    # Инициализация сетки
    x = np.linspace(-geometry_size, geometry_size + 5, grid_size)
    y = np.linspace(-geometry_size, geometry_size, grid_size)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Параметры моделирования
    nu = 1.0 / Re
    Cmu = 0.09
    C1 = 1.44
    C2 = 1.92

    # Инициализация полей
    omega = np.zeros((grid_size, grid_size))
    psi = np.zeros_like(omega)
    k = np.full((grid_size, grid_size), 0.01)  # Начальная турбулентная энергия
    epsilon = np.full_like(k, 0.01)  # Начальная диссипация

    # Маски для граничных условий
    cylinder_mask = (np.abs(X) <= cylinder_size) & (np.abs(Y) <= cylinder_size)
    psi[:, 0] = 1  # Линейный профиль функции тока для притока

    # Массивы истории
    u_history = np.zeros((grid_size, grid_size, time_steps))
    v_history = np.zeros_like(u_history)
    psi_history = np.zeros_like(u_history)

    for t in range(time_steps):
        # Шаг 1: Решение уравнения Пуассона для функции тока
        for _ in range(50):
            psi_new = psi.copy()
            psi_new[1:-1, 1:-1] = 0.25 * (
                psi[2:, 1:-1]
                + psi[:-2, 1:-1]
                + psi[1:-1, 2:]
                + psi[1:-1, :-2]
                + dx**2 * omega[1:-1, 1:-1]
            )

            # Граничные условия
            psi_new[cylinder_mask] = 0  # Поверхность цилиндра
            psi_new[0, :] = psi_new[1, :]  # Верхняя стенка
            psi_new[-1, :] = psi_new[-2, :]  # Нижняя стенка
            psi_new[:, -1] = psi_new[:, -2]  # Отток справа
            psi = psi_new

        # Шаг 2: Расчет скоростей
        u = np.zeros_like(psi)
        v = np.zeros_like(psi)
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * dy)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * dx)

        # Шаг 3: Расчет турбулентных параметров
        # Турбулентная вязкость со стабилизацией
        k_clip = np.clip(k, min_value, 1e3)
        epsilon_clip = np.clip(epsilon, min_value, 1e3)
        nu_t = Cmu * k_clip**2 / (epsilon_clip + min_value)

        # Производные скоростей
        du_dx, du_dy = np.gradient(u, dx, dy)
        dv_dx, dv_dy = np.gradient(v, dx, dy)

        # Производство турбулентной энергии
        P_k = nu_t * (2 * (du_dx**2 + dv_dy**2) + (du_dy + dv_dx) ** 2)
        P_k = np.clip(P_k, 0, 1e3)  # Ограничение производства

        # Обновление k
        k[1:-1, 1:-1] += dt * (
            P_k[1:-1, 1:-1]
            - epsilon[1:-1, 1:-1]
            + nu
            * (
                (k[2:, 1:-1] - 2 * k[1:-1, 1:-1] + k[:-2, 1:-1]) / dy**2
                + (k[1:-1, 2:] - 2 * k[1:-1, 1:-1] + k[1:-1, :-2]) / dx**2
            )
        )
        k[:] = np.clip(k, min_value, 1e3)  # Стабилизация значений

        # Обновление epsilon
        epsilon[1:-1, 1:-1] += dt * (
            C1 * P_k[1:-1, 1:-1] * epsilon[1:-1, 1:-1] / (k[1:-1, 1:-1] + min_value)
            - C2 * epsilon[1:-1, 1:-1] ** 2 / (k[1:-1, 1:-1] + min_value)
            + nu
            * (
                (epsilon[2:, 1:-1] - 2 * epsilon[1:-1, 1:-1] + epsilon[:-2, 1:-1])
                / dy**2
                + (epsilon[1:-1, 2:] - 2 * epsilon[1:-1, 1:-1] + epsilon[1:-1, :-2])
                / dx**2
            )
        )
        epsilon[:] = np.clip(epsilon, min_value, 1e3)

        # Шаг 4: Обновление завихренности с турбулентной диффузией
        laplacian = (
            omega[1:-1, 2:] - 2 * omega[1:-1, 1:-1] + omega[1:-1, :-2]
        ) / dx**2 + (
            omega[2:, 1:-1] - 2 * omega[1:-1, 1:-1] + omega[:-2, 1:-1]
        ) / dy**2

        domega_dx = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2 * dx)
        domega_dy = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2 * dy)

        omega[1:-1, 1:-1] += dt * (
            -u[1:-1, 1:-1] * domega_dx
            - v[1:-1, 1:-1] * domega_dy
            + (nu + nu_t[1:-1, 1:-1]) * laplacian
        )

        # Граничные условия для завихренности
        omega[:, 0] = 0  # Входной поток
        omega[0, :] = 2 * (psi[0, :] - psi[1, :]) / dy**2  # Верхняя стенка
        omega[-1, :] = 2 * (psi[-1, :] - psi[-2, :]) / dy**2  # Нижняя стенка
        omega[:, -1] = omega[:, -2]  # Условие оттока

        # Сохранение результатов
        u_history[..., t] = u
        v_history[..., t] = v
        psi_history[..., t] = psi

    return X, Y, u_history, v_history, psi_history


def animate_stream_function(X, Y, psi_history):
    """
    Анимирует эволюцию функции тока (psi) во времени
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    cylinder = plt.Rectangle(
        (-cylinder_size, -cylinder_size),
        2 * cylinder_size,
        2 * cylinder_size,
        facecolor="white",
        edgecolor="black",
        zorder=5,
    )
    ax.add_patch(cylinder)

    ax.set_title("Эволюция функции тока (psi)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())

    # Начальный контурный график
    contour = ax.contourf(X, Y, psi_history[..., 0], levels=50, cmap="viridis")
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Функция тока (psi)")

    def update(frame):
        ax.clear()
        ax.set_title(f"Функция тока (psi), шаг {frame+1}/{psi_history.shape[-1]}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())

        cylinder = plt.Rectangle(
            (-cylinder_size, -cylinder_size),
            2 * cylinder_size,
            2 * cylinder_size,
            edgecolor="red",
            zorder=5,
        )
        ax.add_patch(cylinder)

        contour = ax.contourf(X, Y, psi_history[..., frame], levels=50, cmap="viridis")
        return (contour,)

    anim = FuncAnimation(
        fig, update, frames=psi_history.shape[-1], interval=100, blit=False
    )

    plt.tight_layout()
    return fig, anim


def visualize_flow(X, Y, u_history, v_history, vectorize=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Flow Around Square Cylinder")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cylinder = plt.Rectangle(
        (-cylinder_size, -cylinder_size),
        2 * cylinder_size,
        2 * cylinder_size,
        edgecolor="red",
        zorder=5,
    )
    ax.add_patch(cylinder)
    if vectorize:
        quiver = ax.quiver(X, Y, u_history[..., 0], v_history[..., 0])

        def update(frame):
            quiver.set_UVC(u_history[..., frame], v_history[..., frame])
            return (quiver,)

    else:
        velocity_history = np.sqrt(u_history**2 + v_history**2)
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())

        contour = ax.contourf(X, Y, velocity_history[..., 0], levels=50, cmap="viridis")
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label("Скорость")

        def update(frame):
            ax.clear()
            ax.set_title(f"Скорость, шаг {frame+1}/{velocity_history.shape[-1]}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_xlim(X.min(), X.max())
            ax.set_ylim(Y.min(), Y.max())

            cylinder = plt.Rectangle(
                (-cylinder_size, -cylinder_size),
                2 * cylinder_size,
                2 * cylinder_size,
                edgecolor="red",
                zorder=5,
            )
            ax.add_patch(cylinder)

            contour = ax.contourf(
                X, Y, velocity_history[..., frame], levels=50, cmap="viridis"
            )
            return contour.collections

    anim = FuncAnimation(
        fig, update, frames=u_history.shape[-1], interval=50, blit=True
    )
    plt.show()


geometry_size = 3
Re = 3000
grid_size = 50
time_steps = 200

X, Y, u_history, v_history, psi_history = flow_around_square_cylinder(
    geometry_size=geometry_size, Re=Re, grid_size=grid_size, time_steps=time_steps
)

fig, anim = animate_stream_function(X, Y, psi_history)
visualize_flow(X, Y, u_history, v_history)
plt.show()
