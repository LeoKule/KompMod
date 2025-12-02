import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import math


# ---- 1. Определяем правильный 13-угольник ----
def regular_polygon(n_sides=13, radius=1.0, center=(0, 0)):
    """
    Создает правильный многоугольник (равносторонний и равноугольный)
    n_sides: количество сторон (по умолчанию 13)
    radius: радиус описанной окружности
    center: координаты центра
    """
    cx, cy = center  # координаты центра
    # Создаем массив углов для вершин многоугольника (от 0 до 2π без последней точки)
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    # Вычисляем координаты всех вершин многоугольника
    points = np.array([
        (cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles
    ])
    return Polygon(points)  # Создаем объект Polygon из библиотеки shapely


# Создаем 13-угольник с радиусом 1 и центром в (0, 0)
polygon = regular_polygon(n_sides=13, radius=1.0)
print(f"Площадь 13-угольника (точная, shapely): {polygon.area:.5f}")


# ---- 2. Функция для интегрирования ----
def f(x, y):
    """Функция, которую мы интегрируем по области многоугольника: f(x,y) = x²y²"""
    return x ** 2 * y ** 2


# ---- 3. Метод Монте-Карло ----
def monte_carlo_area_and_integral(polygon, func, n_points):
    """
    Вычисляет площадь многоугольника и интеграл функции по его области
    с использованием метода Монте-Карло

    polygon: объект Polygon (многоугольник)
    func: функция для интегрирования
    n_points: количество случайных точек
    """
    # Получаем ограничивающий прямоугольник многоугольника (минимальные и максимальные координаты)
    minx, miny, maxx, maxy = polygon.bounds

    # Генерируем случайные точки внутри ограничивающего прямоугольника
    xs = np.random.uniform(minx, maxx, n_points)
    ys = np.random.uniform(miny, maxy, n_points)
    points = np.vstack((xs, ys)).T  # Объединяем x и y в массив точек

    # Проверяем, какие точки лежат внутри многоугольника
    inside = np.array([polygon.contains(Point(x, y)) for x, y in points])
    inside_points = points[inside]  # Точки, которые действительно внутри

    # Вычисляем площадь ограничивающего прямоугольника
    area_box = (maxx - minx) * (maxy - miny)

    # Оценка площади многоугольника методом Монте-Карло:
    # Площадь ≈ Площадь_прямоугольника × (Количество_точек_внутри / Общее_количество_точек)
    area_est = area_box * np.sum(inside) / n_points

    # Оценка интеграла методом Монте-Карло:
    # Интеграл ≈ Площадь_прямоугольника × Среднее_значение_функции_в_точках_внутри × (Внутри/Всего)
    integral_est = area_box * np.mean(func(xs[inside], ys[inside])) * np.sum(inside) / n_points

    return area_est, integral_est, points, inside


# ---- 4. Проверка для разных N ----
# Разные количества точек для проверки сходимости метода
N_values = [1000, 5000, 20000, 50000, 100000]
areas, integrals = [], []  # Списки для хранения результатов

print("\n--- Результаты метода Монте-Карло ---")
for N in N_values:
    # Выполняем расчет для каждого N
    area, integral, points, inside = monte_carlo_area_and_integral(polygon, f, N)
    areas.append(area)  # Сохраняем оценку площади
    integrals.append(integral)  # Сохраняем оценку интеграла
    # Выводим результаты
    print(f"N={N:7d} | Площадь ≈ {area:.5f} | Интеграл ≈ {integral:.5f}")

# ---- 5. Визуализация точек ----
plt.figure(figsize=(7, 6))
# Рисуем границу 13-угольника
plt.plot(*polygon.exterior.xy, 'k-', linewidth=2, label="13-угольник")
# Рисуем все точки: зеленые - внутри, красные - снаружи
plt.scatter(points[:, 0], points[:, 1], s=4,
            c=np.where(inside, 'green', 'red'), alpha=0.5)
plt.title("Метод Монте-Карло: точки внутри/вне 13-угольника")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis('equal')  # Одинаковый масштаб по осям
plt.show()

# ---- 6. Графики зависимости ----
plt.figure(figsize=(10, 4))

# График 1: Сходимость оценки площади
plt.subplot(1, 2, 1)
plt.plot(N_values, areas, marker='o', color='blue')
plt.title("Сходимость оценки площади")
plt.xlabel("N (количество точек)")
plt.ylabel("Площадь")

# График 2: Сходимость оценки интеграла
plt.subplot(1, 2, 2)
plt.plot(N_values, integrals, marker='o', color='orange')
plt.title("Сходимость оценки интеграла ∫ x²y² dA")
plt.xlabel("N (количество точек)")
plt.ylabel("Интеграл")

plt.tight_layout()  # Автоматическая настройка отступов
plt.show()