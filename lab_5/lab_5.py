import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Параметры задачи
x_min, x_max = 0.0, 10.0
t_min, t_max = 0.0, 100.0  # Обратите внимание: t_max довольно большое, экспонента в аналитике может расти быстро

V = lambda x: x + 5
f = lambda x: -27.2
u_init = lambda x: np.minimum(0, (x - 3) * (x - 8))
u_left = lambda t: t / 50


# --- Численные методы (без изменений) ---

def explicit_corner(N_x, N_t):
    h = (x_max - x_min) / N_x
    tau = (t_max - t_min) / N_t
    x = np.linspace(x_min, x_max, N_x + 1)
    t = np.linspace(t_min, t_max, N_t + 1)
    U = np.zeros((N_t + 1, N_x + 1))
    U[0, :] = u_init(x)
    for n in range(N_t):
        U[n + 1, 0] = u_left(t[n + 1])
        for i in range(1, N_x + 1):
            U[n + 1, i] = U[n, i] - (tau / h) * V(x[i]) * (U[n, i] - U[n, i - 1]) + tau * f(x[i])
    return x, t, U.T


def implicit_corner(N_x, N_t):
    h = (x_max - x_min) / N_x
    tau = (t_max - t_min) / N_t
    x = np.linspace(x_min, x_max, N_x + 1)
    t = np.linspace(t_min, t_max, N_t + 1)
    U = np.zeros((N_t + 1, N_x + 1))
    U[0, :] = u_init(x)
    for n in range(N_t):
        U[n + 1, 0] = u_left(t[n + 1])
        for i in range(1, N_x + 1):
            sigma = tau / h * V(x[i])
            U[n + 1, i] = (U[n, i] + tau * f(x[i]) + sigma * U[n + 1, i - 1]) / (1 + sigma)
    return x, t, U.T


def U_analytic_func(x, t):
    # Примечание: при больших t (t=100) np.exp(t) даст огромное число.
    # Это может вызвать переполнение, но для python int/float обычно обрабатывается.
    # Однако x0 будет очень большим отрицательным числом.
    x0 = 5.0 - (x + 5) * np.exp(t)
    if x0 >= 0.0:
        return u_init(x0) - 27.2 * t
    else:
        # Логарифм от отрицательного числа выдаст ошибку, если x < -5.
        # В вашем диапазоне x [0, 10], поэтому 5 + x/5 положительно.
        s = t + np.log(5.0 + x / 5.0)
        # s может быть отрицательным или больше t, нужно быть осторожным с u_left
        return u_left(s) - 27.2 * (t - s)


def analytic(N_x, N_t):
    x = np.linspace(x_min, x_max, N_x + 1)
    t = np.linspace(t_min, t_max, N_t + 1)
    U = np.zeros((N_x + 1, N_t + 1))
    for i, x_i in enumerate(x):
        for j, t_j in enumerate(t):
            U[i][j] = U_analytic_func(x_i, t_j)
    return x, t, U


# --- Функция расширенной визуализации ---

def visualize_results(x, t, u_num, u_an, title_num="Численное", title_an="Аналитическое"):
    """
    Строит 3 графика:
    1. 3D поверхность аналитического решения.
    2. Тепловая карта численного решения.
    3. Срез решения в последний момент времени (сравнение).
    """
    # Создаем сетку для 3D графиков
    T_grid, X_grid = np.meshgrid(t, x)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Визуализация решений ({len(x) - 1}x{len(t) - 1})", fontsize=16)

    # 1. 3D График (Аналитика)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax1.plot_surface(X_grid, T_grid, u_an, cmap=cm.viridis, alpha=0.8, edgecolor='none')
    ax1.set_title(f"3D Поверхность: {title_an}")
    ax1.set_xlabel("Координата X")
    ax1.set_ylabel("Время T")
    ax1.set_zlabel("U(x,t)")
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    # 2. Тепловая карта (Численное решение - Неявная схема)
    ax2 = fig.add_subplot(2, 2, 2)
    # Используем pcolormesh для тепловой карты. shading='auto' для корректной отрисовки
    c = ax2.pcolormesh(X_grid, T_grid, u_num, cmap=cm.plasma, shading='auto')
    ax2.set_title(f"Тепловая карта: {title_num}")
    ax2.set_xlabel("Координата X")
    ax2.set_ylabel("Время T")
    fig.colorbar(c, ax=ax2)

    # 3. График ошибки (разница) в 3D (или просто другое представление)
    # Вместо этого давайте построим сравнение срезов в середине времени и в конце
    ax3 = fig.add_subplot(2, 1, 2)

    # Срез в конечный момент
    ax3.plot(x, u_num[:, -1], 'r-', label=f"{title_num} (t={t_max})", lw=2)
    ax3.plot(x, u_an[:, -1], 'k--', label=f"{title_an} (t={t_max})", lw=2)

    # Срез в середине времени (примерно)
    mid_idx = len(t) // 2
    t_mid = t[mid_idx]
    ax3.plot(x, u_num[:, mid_idx], 'r:', label=f"{title_num} (t={t_mid:.1f})", lw=1.5)
    ax3.plot(x, u_an[:, mid_idx], 'k:', label=f"{title_an} (t={t_mid:.1f})", lw=1.5)

    ax3.set_title("Сравнение профилей решений (срезы по времени)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("U")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


# --- Основная логика ---

def solve(N_x, N_t, show_visualization=False):
    print(f"=== Расчет на сетке: {N_x} точек по X, {N_t} точек по времени ===")

    # Запуск расчетов
    x_ex, t_ex, u_ex = explicit_corner(N_x, N_t)
    x_impl, t_impl, u_impl = implicit_corner(N_x, N_t)
    x_al, t_al, u_al = analytic(N_x, N_t)

    # Расчет расхождений (ошибок)
    # np.var вычисляет дисперсию разности. Чем ближе к 0, тем решения более похожи.
    var_impl_ex = np.var(u_impl - u_ex)
    var_ex_al = np.var(u_ex - u_al)
    var_al_impl = np.var(u_al - u_impl)

    # Среднеквадратичное отклонение (стандартное отклонение)
    std_impl_ex = np.std(u_impl - u_ex)  # или np.sqrt(var_impl_ex)
    std_ex_al = np.std(u_ex - u_al)
    std_al_impl = np.std(u_al - u_impl)

    print(f"  Разница между схемами:")
    print(f"    Дисперсия: {var_impl_ex:}")
    print(f"    Стандартное отклонение: {std_impl_ex:}")

    print(f"  Ошибка Явной схемы (относительно точного):")
    print(f"    Дисперсия: {var_ex_al:}")
    print(f"    Стандартное отклонение: {std_ex_al:}")

    print(f"  Ошибка Неявной схемы (относительно точного):")
    print(f"    Дисперсия: {var_al_impl:}")
    print(f"    Стандартное отклонение: {std_al_impl:}")

    print("-" * 40)

    # Если флаг включен, вызываем большую визуализацию
    if show_visualization:
        # Сравниваем Неявную схему (более устойчивую) с Аналитикой
        visualize_results(x_impl, t_impl, u_impl, u_al,
                          title_num="Неявная схема",
                          title_an="Аналитическое")


# Запуск тестов
solve(5, 500, show_visualization=True)
solve(10, 1000, show_visualization=True)
solve(30, 3000, show_visualization=True)
solve(50, 5000, show_visualization=True)

print("\n--- ГЕНЕРАЦИЯ ФИНАЛЬНОЙ ВИЗУАЛИЗАЦИИ ДЛЯ ЛУЧШЕЙ СЕТКИ ---")
# Включаем визуализацию только для самой детальной сетки
solve(100, 10000, show_visualization=True)