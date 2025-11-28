import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import math

# ---- 1. Определяем правильный 13-угольник ----
def regular_polygon(n_sides=13, radius=1.0, center=(0, 0)):
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    points = np.array([
        (cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles
    ])
    return Polygon(points)

polygon = regular_polygon(n_sides=13, radius=1.0)
print(f"Площадь 13-угольника (точная, shapely): {polygon.area:.5f}")

# ---- 2. Функция для интегрирования ----
def f(x, y):
    return x ** 2 * y ** 2


# ---- 3. Метод Монте-Карло ----
def monte_carlo_area_and_integral(polygon, func, n_points):
    minx, miny, maxx, maxy = polygon.bounds
    xs = np.random.uniform(minx, maxx, n_points)
    ys = np.random.uniform(miny, maxy, n_points)
    points = np.vstack((xs, ys)).T

    inside = np.array([polygon.contains(Point(x, y)) for x, y in points])
    inside_points = points[inside]

    area_box = (maxx - minx) * (maxy - miny)
    area_est = area_box * np.sum(inside) / n_points
    integral_est = area_box * np.mean(func(xs[inside], ys[inside])) * np.sum(inside) / n_points

    return area_est, integral_est, points, inside


# ---- 4. Проверка для разных N ----
N_values = [1000, 5000, 20000, 50000, 100000]
areas, integrals = [], []

print("\n--- Результаты метода Монте-Карло ---")
for N in N_values:
    area, integral, points, inside = monte_carlo_area_and_integral(polygon, f, N)
    areas.append(area)
    integrals.append(integral)
    print(f"N={N:7d} | Площадь ≈ {area:.5f} | Интеграл ≈ {integral:.5f}")

# ---- 5. Визуализация точек ----
plt.figure(figsize=(7, 6))
plt.plot(*polygon.exterior.xy, 'k-', linewidth=2, label="13-угольник")
plt.scatter(points[:, 0], points[:, 1], s=4,
            c=np.where(inside, 'green', 'red'), alpha=0.5)
plt.title("Метод Монте-Карло: точки внутри/вне 13-угольника")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis('equal')
plt.show()

# ---- 6. Графики зависимости ----
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(N_values, areas, marker='o', color='blue')
plt.title("Сходимость оценки площади")
plt.xlabel("N")
plt.ylabel("Площадь")

plt.subplot(1, 2, 2)
plt.plot(N_values, integrals, marker='o', color='orange')
plt.title("Сходимость оценки интеграла ∫ x²y² dA")
plt.xlabel("N")
plt.ylabel("Интеграл")

plt.tight_layout()
plt.show()
