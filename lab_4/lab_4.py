from random import random, choice
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
#              ТРЁХЛИСТНОЕ СТОХАСТИЧЕСКОЕ ОТОБРАЖЕНИЕ (IFS)
# ============================================================
# Матрицы A и векторы b из задания (Вариант 14)
# Каждая пара (A, b) — это линейное отображение x → A·x + b
# Три таких отображения вместе образуют IFS (iterated function system)
A1 = np.array([[ 0.754,  0.568],
               [-0.500, -0.576]])
b1 = np.array([28.586, 49.567])

A2 = np.array([[-0.611,  0.476],
               [-0.564, -0.320]])
b2 = np.array([20.525, 29.918])

A3 = np.array([[ 0.243, -1.568],
               [-0.144,  0.324]])
b3 = np.array([-49.663, -69.656])

transforms = [(A1, b1), (A2, b2), (A3, b3)]


# ============================================================
#                  ПОСТРОЕНИЕ IFS-АТТРАКТОРА
# ============================================================
def ifs_attractor(n_points=200000):
    # Начальная точка (0, 0)
    x = np.zeros(2)

    pts = []

    for i in range(n_points):
        #На каждом шаге случайно выбирается одно из трёх отображений — и получается новая точка.
        A, b = transforms[np.random.randint(3)]

        # Вычисляем новую точку: x_new = A*x + b
        x = A @ x + b

        # Пропускаем первые 100, пока система не "войдёт" в аттрактор
        if i > 100:
            pts.append(x)
    # После большого числа итераций формируется устойчивое множество точек (=аттрактор)
    return np.array(pts)


# ============================================================
#              ФРАКТАЛЬНАЯ РАЗМЕРНОСТЬ (BOX COUNTING)
# ============================================================
def fractal_dimension(Z):
    assert len(Z.shape) == 2

    # # колличество непустых квадратов массива Z размером k на k
    def boxcount(Z, k):
        S = np.add.reduceat(
            # Делит массив Z на квадраты k на k.
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            # считает сколько квадратов не пустые
                np.arange(0, Z.shape[1], k), axis=1
            )
        return len(np.where((S > 0) & (S < k*k))[0])

    Z = (Z > 0)
    p = min(Z.shape)
    n = 2 ** np.floor(np.log2(p))
    sizes = 2 ** np.arange(int(np.log2(n)), 1, -1)

    # Для каждого размера квадрата считаем количество занятых блоков
    counts = [boxcount(Z, int(size)) for size in sizes]

    # Линейная аппроксимация
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)

    return -coeffs[0]


# ============================================================
#        СЛУЧАЙНОЕ БЛУЖДАНИЕ С ПРИЛИПАНИЕМ (DLA-модель)
# ============================================================
def random_walk_stick(grid_size=201, n_particles=500):
    """
    Модель образования фрактального кластера DLA.
    Частицы стартуют на окружности, бродят случайно и прилипают.
    """

    # Создаём пустую решётку
    grid = np.zeros((grid_size, grid_size), dtype=int)

    # Начальная точка — "семечко" в центре
    center = grid_size // 2
    grid[center, center] = 1 # начальная "заданная точка"

    max_r = 1  # максимальный радиус скопления

    for _ in range(n_particles):
        # Старт частицы — на окружности вокруг центра
        # Каждая частица начинает на окружности чуть дальше от уже образованного кластера:
        angle = 2 * np.pi * random()
        r0 = max_r + 5  # старт чуть дальше кластера
        x = int(center + r0 * np.cos(angle))
        y = int(center + r0 * np.sin(angle))

        # Пока частица не прилипла
        while True:
            # Вероятность горизонтального перехода в 2 раза больше вертикального
            p = random()
            if p < 0.5:
                dx, dy = choice([(1, 0), (-1, 0)])
            else:
                dx, dy = choice([(0, 1), (0, -1)])

            x += dx
            y += dy

            # Проверка выхода за границы
            if not (1 <= x < grid_size-1 and 1 <= y < grid_size-1):
                break # "улетела"

            # Проверка прилипания
            # Если частица оказывается рядом (в квадрате 3×3) с уже прилипшими точками — она “приклеивается”:
            if np.any(grid[x-1:x+2, y-1:y+2]):
                grid[x, y] = 1

                r = np.sqrt((x-center)**2 + (y-center)**2)
                max_r = max(max_r, r)
                break

    return grid


# ============================================================
#         I. СЛУЧАЙНОЕ БЛУЖДАНИЕ + РАЗМЕРНОСТЬ
# ============================================================

for N in [800, 1500, 5000]:
    grid = random_walk_stick(grid_size=301, n_particles=N)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='inferno', origin='lower')
    plt.title(f"DLA-кластер, частиц = {N}")
    plt.axis('off')
    plt.show()

    fd = fractal_dimension(grid)
    print(f"Метрическая размерность для {N} частиц: {fd:.4f}")


# ============================================================
#            II. ПОСТРОЕНИЕ IFS-АТТРАКТОРА
# ============================================================

for N in [50000, 150000, 300000]:
    pts = ifs_attractor(N)

    plt.figure(figsize=(7, 7))
    plt.scatter(pts[:, 0], pts[:, 1], s=0.25, color='black')
    plt.title(f"IFS-аттрактор, точек = {N}")
    plt.axis("equal")
    plt.axis("off")
    plt.show()


# Оценка размерности аттрактора
def points_to_grid(points, res=800):
    pts = points - points.min(axis=0)
    pts /= pts.max()
    grid = np.zeros((res, res))
    for x, y in pts:
        grid[int(x*(res-1)), int(y*(res-1))] = 1
    return grid

grid_ifs = points_to_grid(ifs_attractor(250000))
fd_ifs = fractal_dimension(grid_ifs)
print("\nФрактальная размерность аттрактора IFS:", fd_ifs)
