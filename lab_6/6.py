import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- физика и условия ---
U_x0 = lambda x: (x**2) * (10 - x)   # начальное условие u(x,0)
U_0t = lambda t: 0                   # граничные условия u(0,t)=u(10,t)=0
D = lambda x: 25 - 2 * x
f = lambda x: x
source_term = lambda x, t: 5.0       # правая часть (здесь константа 5)

# --- область ---
x_min, x_max = 0.0, 10.0
t_min, t_max = 0.0, 1.0

# --- сетки ---
h_values = [0.85, 0.65, 0.35, 0.15, 0.08]
D_max = 25
safety = 0.4
tau = [safety * h*h / (2*D_max) for h in h_values]

print("tau для соответствующих h:", tau)

# куда будем сохранять результаты
solutions = []   # решение на последнем временном слое для каждой сетки
meshes = []      # соответствующие x-сетки
timings = []     # время расчёта для каждой сетки


def explicit_fast(h, tau, plot_profiles=False):
    """
    Явный метод.
    Возвращает x, t, u (матрица len(x) x len(t)), и время расчёта в секундах.
    Параметры:
    h - шаг по пространству
    tau - шаг по времени
    """
    x = np.arange(x_min, x_max + 1e-12 + h, h)   # включаем x_max
    t = np.arange(t_min, t_max + 1e-12 + tau, tau)
    Nx = len(x)  # Количество точек по пространству
    Nt = len(t)  # Количество точек по времени

    # инициализация
    u = np.zeros((Nx, Nt))

    # начальное условие
    u[:, 0] = U_x0(x)
    # граничные условия (все времена)
    u[0, :] = U_0t(t)
    u[-1, :] = U_0t(t)

    # предвычисляем D и f по x и коэффициенты, чтобы не вычислять в цикле
    D_x = D(x)
    f_x = f(x)

    # коэффициенты на внутренних узлах (вектор)
    coef_side = D_x * tau / (h ** 2)
    coef_center = 1.0 - 2.0 * coef_side - f_x * tau


    inner = slice(1, -1)


    rhs_const = source_term(None, None) * tau

    t0 = time.perf_counter()

    # цикл только по времени (векторное обновление по x)
    for j in range(Nt - 1):
        # u[1:-1, j+1] вычисляется векторно
        u[inner, j + 1] = (
            coef_center[inner] * u[inner, j]
            + coef_side[inner] * (u[2:, j] + u[:-2, j])
            + rhs_const
        )
        # граничные значения уже установлены и остаются

    t1 = time.perf_counter()
    elapsed = t1 - t0


    print(f"h={h:.6g}, tau={tau:.6g}, Nx={Nx}, Nt={Nt}, time={elapsed:.4f}s")

    # визуализация: 2D-профили в нескольких временах
    if plot_profiles:
        t_vals = np.linspace(t_min, t_max, 6)
        plt.figure(figsize=(8, 5))
        for tt in t_vals:
            j = int(np.round((tt - t_min) / tau))
            j = min(max(j, 0), Nt - 1)
            plt.plot(x, u[:, j], label=f"t={t[j]:.3f}")
        plt.title(f"Профили u(x,t) для h={h}, tau={tau}")
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend()
        plt.grid(True)
        plt.show()

    return x, t, u, elapsed


def compute_and_plot_all():
    # вычисляем решения для всех сеток
    for idx, (h, ta) in enumerate(zip(h_values, tau)):

        plot_profiles = True if idx < 4 else False
        x, t, u, elapsed = explicit_fast(h, ta, plot_profiles=plot_profiles)
        solutions.append(u[:, -1].copy())
        meshes.append(x.copy())
        timings.append(elapsed)


    x_fine, t_fine, u_fine_full, _ = explicit_fast(h_values[-1], tau[-1], plot_profiles=False)

    X, T = np.meshgrid(x_fine, t_fine)
    U = u_fine_full.T

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T, U, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title(f"3D-поверхность u(x,t) для h={h_values[-1]}, tau={tau[-1]}")
    fig.colorbar(surf, shrink=0.5, aspect=8, label='u(x,t)')
    plt.show()


    solutions[-1] = u_fine_full[:, -1].copy()
    meshes[-1] = x_fine.copy()

    # --- оценка ошибок и порядок аппроксимации ---
    u_fine = solutions[-1]
    x_fine = meshes[-1]

    errors = []
    for k in range(len(h_values) - 1):
        x_coarse = meshes[k]
        u_coarse = solutions[k]

        interp = interp1d(x_fine, u_fine, kind='linear', fill_value="extrapolate", bounds_error=False)
        u_fine_interp = interp(x_coarse)

        err = np.sqrt(np.mean((u_coarse - u_fine_interp) ** 2))
        errors.append(err)
        print(" ")
        print(f"h={h_values[k]:.6g}, Среднеквадратичная ошибка против 'точной' = {err:.6e}")

    # график сходимости
    h_plot = np.array(h_values[:-1])
    errors = np.array(errors)

    plt.figure(figsize=(7, 5))
    plt.loglog(h_plot, errors, 'o-')
    plt.xlabel('h')
    plt.ylabel('RMS error')
    plt.grid(True, which='both', ls='--')
    plt.title('Сходимость по h')
    plt.show()

    # аппроксимированный порядок p
    log_h = np.log(h_plot)
    log_E = np.log(errors)
    p_fit = np.polyfit(log_h, log_E, 1)[0]
    print(f"Приближённый порядок аппроксимации p ≈ {p_fit:.3f}")

    # напечатаем времена расчёта
    #for h, tcalc in zip(h_values, timings):
        #print(f"h={h:.6g}  расчет занял {tcalc:.4f}s")


if __name__ == "__main__":
    compute_and_plot_all()
