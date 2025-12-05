import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

# ---------------------------------------------------------
# УСЛОВИЯ ВАРИАНТА 24
# ---------------------------------------------------------

L = 10.0      # длина отрезка по x
maxT = 1.0    # максимальное время моделирования


def D(x):
    return 25 - 2 * x   # коэффициент диффузии по условию


def f(x):
    return x            # функция f(x) по условию


def initial_condition(x):
    return x ** 2 * (10 - x)   # начальное условие U(0, x)


# ---------------------------------------------------------
# ЯВНАЯ СХЕМА
# ---------------------------------------------------------
def explicit_scheme(h, tau):
    Nx = int(L / h) + 1      # число узлов по x
    Nt = int(maxT / tau) + 1 # число шагов по времени

    x = np.linspace(0, L, Nx)    # сетка по x
    t = np.linspace(0, maxT, Nt) # сетка по t

    U = np.zeros((Nt, Nx))       # массив решения
    U[0, :] = initial_condition(x)  # начальное состояние

    # основной цикл по времени
    for n in range(0, Nt - 1):
        # цикл по внутренним узлам (без границ)
        for i in range(1, Nx - 1):
            # вторая производная по x (центральная разность)
            d2u = (U[n, i + 1] - 2 * U[n, i] + U[n, i - 1]) / h ** 2

            # явное обновление решения
            U[n + 1, i] = U[n, i] + tau * (D(x[i]) * d2u - f(x[i]) * U[n, i] + 5)

        # граничные условия (по условию U = 0 на границах)
        U[n + 1, 0] = 0
        U[n + 1, -1] = 0

    return x, t, U


# ---------------------------------------------------------
# НЕЯВНАЯ СХЕМА
# ---------------------------------------------------------
def implicit_scheme(h, tau):
    Nx = int(L / h) + 1
    Nt = int(maxT / tau) + 1

    x = np.linspace(0, L, Nx)
    t = np.linspace(0, maxT, Nt)

    U = np.zeros((Nt, Nx))
    U[0, :] = initial_condition(x)

    # коэффициенты для матрицы A (трёхдиагональная)
    alpha = np.zeros(Nx)  # нижняя диагональ
    beta = np.zeros(Nx)   # главная диагональ
    gamma = np.zeros(Nx)  # верхняя диагональ

    # формирование коэффициентов для внутренних узлов
    for i in range(1, Nx - 1):
        alpha[i] = -tau * D(x[i]) / h ** 2
        beta[i]  = 1 + 2 * tau * D(x[i]) / h ** 2 + tau * f(x[i])
        gamma[i] = -tau * D(x[i]) / h ** 2

    # граничные условия: U = 0, значит в матрице на границах стоит 1
    beta[0] = 1
    beta[-1] = 1

    # собираем трёхдиагональную матрицу A
    diagonals = [alpha[1:], beta, gamma[:-1]]
    A = diags(diagonals, [-1, 0, 1], format='csc')

    # цикл по времени
    for n in range(0, Nt - 1):
        b = U[n, :].copy()   # правая часть

        # добавляем +5*tau по внутренним узлам
        b[1:-1] += 5 * tau

        # граничные условия
        b[0] = 0
        b[-1] = 0

        # решение линейной системы A * U(n+1) = b
        U[n + 1, :] = spsolve(A, b)

    return x, t, U


# ---------------------------------------------------------
# Оценка ошибки по пространству
# ---------------------------------------------------------
def compute_error_h(u_real, u_calc, x_real, x_calc):
    # интерполируем "реальное" решение в узлы другой сетки
    interp_func = interp1d(
        x_real, u_real, kind='cubic',
        bounds_error=False, fill_value='extrapolate'
    )
    u_real_interp = interp_func(x_calc)
    return np.max(np.abs(u_real_interp - u_calc))


# ---------------------------------------------------------
# Оценка ошибки по времени
# ---------------------------------------------------------
def compute_error_t(u_real, u_calc, t_real, t_calc):
    # интерполяция по времени (массив двумерный!)
    interp_func = interp1d(
        t_real, u_real, axis=0, kind='cubic',
        bounds_error=False, fill_value='extrapolate'
    )
    u_real_interp = interp_func(t_calc)
    return np.max(np.abs(u_real_interp - u_calc))


# ---------------------------------------------------------
# Визуализация 2D
# ---------------------------------------------------------
def solve(h, tau):
    # запускаем обе схемы
    x_ex, t_ex, u_ex = explicit_scheme(h, tau)
    x_im, t_im, u_im = implicit_scheme(h, tau)

    plt.figure(figsize=(14, 6))

    # график явной схемы
    plt.subplot(1, 2, 1)
    plt.plot(x_ex, u_ex[0, :], label="U")
    plt.title("Явная схема")
    plt.xlabel("x")
    plt.ylabel("U")
    plt.grid()
    plt.legend()

    # график неявной схемы
    plt.subplot(1, 2, 2)
    plt.plot(x_im, u_im[0, :], label="U")
    plt.title("Неявная схема")
    plt.xlabel("x")
    plt.ylabel("U")
    plt.grid()
    plt.legend()

    plt.show()

    return x_ex, t_ex, u_ex, u_im


# ---------------------------------------------------------
# Запуск экспериментов с разными шагами h
# ---------------------------------------------------------
x_r, t_r, u_ex_r, u_im_r = solve(0.05, 0.00001)

x_1, _, u_ex1, u_im1 = solve(0.7, 0.00001)
x_2, _, u_ex2, u_im2 = solve(0.5, 0.00001)
x_3, _, u_ex3, u_im3 = solve(0.25, 0.00001)
x_4, _, u_ex4, u_im4 = solve(0.125, 0.00001)


