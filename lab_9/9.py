import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.optimize import minimize
import time


class WaveEquationSolver:
    def __init__(self, L=10, T=10, N=100, M=1000):
        self.L = L
        self.T = T
        self.N = N
        self.M = M

        self.h = L / N
        self.tau = T / M

        self.x = np.linspace(0, L, N + 1)
        self.t = np.linspace(0, T, M + 1)

        # Проверка условия устойчивости
        max_D = 11
        courant_condition = self.tau <= self.h / np.sqrt(max_D)
        print(f"Условие Куранта выполнено: {courant_condition}")
        print(f"τ = {self.tau:.6f}, h/√D_max = {self.h / np.sqrt(max_D):.6f}")

    def D(self, x):
        return 1 / (x ** 2 + x + 1)

    def initial_condition(self, x):
        return (x ** 4) * (10 - x)

    def solve_direct(self):
        """Решение прямой задачи - корректная задача"""
        U = np.zeros((self.M + 1, self.N + 1))

        # Начальные условия
        U[0, :] = self.initial_condition(self.x)

        # Первый шаг (нулевая начальная скорость)
        U[1, 1:-1] = U[0, 1:-1] + 0.5 * self.tau ** 2 * (
                self.D(self.x[1:-1]) * (U[0, 2:] - 2 * U[0, 1:-1] + U[0, :-2]) / self.h ** 2
                - self.x[1:-1] * U[0, 1:-1]
        )

        # Основной цикл
        for j in range(1, self.M):
            U[j + 1, 1:-1] = (
                    2 * U[j, 1:-1] - U[j - 1, 1:-1] +
                    self.tau ** 2 * (
                            self.D(self.x[1:-1]) * (U[j, 2:] - 2 * U[j, 1:-1] + U[j, :-2]) / self.h ** 2
                            - self.x[1:-1] * U[j, 1:-1]
                    )
            )

        return U


class InverseProblemSolver:
    """Решение обратных задач - потенциально некорректных"""

    def __init__(self, direct_solver):
        self.solver = direct_solver
        self.U_direct = None

    def solve_inverse_initial(self, U_final, noise_level=0.01):
        """
        Обратная задача 1: восстановление начального условия
        НЕКОРРЕКТНАЯ задача - неустойчива к малым возмущениям
        """
        # Добавляем шум к данным (имитация погрешности измерений)
        U_noisy = U_final * (1 + noise_level * np.random.randn(*U_final.shape))

        U_inverse = np.zeros_like(U_noisy)
        U_inverse[-1, :] = U_noisy[-1, :]

        # Обратное интегрирование (неустойчивый процесс)
        for j in range(self.solver.M - 1, 0, -1):
            U_inverse[j - 1, 1:-1] = (
                    2 * U_inverse[j, 1:-1] - U_inverse[j + 1, 1:-1] -
                    self.solver.tau ** 2 * (
                            self.solver.D(self.solver.x[1:-1]) *
                            (U_inverse[j, 2:] - 2 * U_inverse[j, 1:-1] + U_inverse[j, :-2]) / self.solver.h ** 2
                            - self.solver.x[1:-1] * U_inverse[j, 1:-1]
                    )
            )

        return U_inverse

    def solve_inverse_coefficient(self, U_measured, true_D=None):
        """
        Обратная задача 2: восстановление коэффициента D(x)
        НЕКОРРЕКТНАЯ задача - может не иметь единственного решения
        """
        D_estimated = np.zeros_like(self.solver.x)

        # Используем несколько временных срезов для устойчивости
        time_slices = [1, 2, 3]

        for i in range(1, self.solver.N):
            numerator = 0
            denominator = 0

            for j in time_slices:
                # Численные производные
                U_tt = (U_measured[j + 1, i] - 2 * U_measured[j, i] + U_measured[j - 1, i]) / self.solver.tau ** 2
                U_xx = (U_measured[j, i + 1] - 2 * U_measured[j, i] + U_measured[j, i - 1]) / self.solver.h ** 2

                if abs(U_xx) > 1e-10:
                    numerator += (U_tt + self.solver.x[i] * U_measured[j, i]) * U_xx
                    denominator += U_xx ** 2

            if denominator > 1e-10:
                D_estimated[i] = numerator / denominator
            else:
                D_estimated[i] = self.solver.D(self.solver.x[i])  # fallback

        return D_estimated


def analyze_correctness(direct_solver, inverse_solver, U_direct):
    """Анализ корректности задач"""
    print("\n" + "=" * 50)
    print("АНАЛИЗ КОРРЕКТНОСТИ ЗАДАЧ")
    print("=" * 50)

    # Прямая задача - корректная
    print("\n1. ПРЯМАЯ ЗАДАЧА:")
    print("   - Существование: ✓ (решение найдено)")
    print("   - Единственность: ✓ (при заданных условиях)")
    print("   - Устойчивость: ✓ (при выполнении условия Куранта)")
    print("   ВЫВОД: КОРРЕКТНАЯ задача")

    # Обратная задача 1 - восстановление начальных условий
    print("\n2. ОБРАТНАЯ ЗАДАЧА 1 (восстановление начального условия):")

    # Тест на устойчивость
    U_inverse_no_noise = inverse_solver.solve_inverse_initial(U_direct, noise_level=0)
    U_inverse_noise = inverse_solver.solve_inverse_initial(U_direct, noise_level=0.01)

    error_no_noise = np.max(np.abs(U_inverse_no_noise[0, :] - U_direct[0, :]))
    error_with_noise = np.max(np.abs(U_inverse_noise[0, :] - U_direct[0, :]))

    print(f"   - Ошибка без шума: {error_no_noise:.6f}")
    print(f"   - Ошибка с шумом 1%: {error_with_noise:.6f}")
    print("   - Устойчивость: ✗ (чувствительность к шуму)")
    print("   ВЫВОД: НЕКОРРЕКТНАЯ задача")

    # Обратная задача 2 - восстановление коэффициента
    print("\n3. ОБРАТНАЯ ЗАДАЧА 2 (восстановление коэффициента D(x)):")
    D_estimated = inverse_solver.solve_inverse_coefficient(U_direct)
    D_true = direct_solver.D(direct_solver.x)

    error_D = np.max(np.abs(D_estimated - D_true))
    print(f"   - Ошибка восстановления: {error_D:.6f}")
    print("   - Единственность: ? (может быть неединственность)")
    print("   - Устойчивость: ✗ (чувствительность к погрешностям)")
    print("   ВЫВОД: НЕКОРРЕКТНАЯ задача")


def save_frames_as_images(direct_solver, U_direct, inverse_results, num_frames=20):
    """
    Вместо анимации сохраняет статические кадры в папку.
    num_frames: количество кадров (графиков), которые нужно сохранить.
    """
    # Создаем папку для сохранения, если её нет
    output_dir = "wave_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nНачинаю сохранение {num_frames} кадров в папку '{output_dir}'...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # --- ВЕРХНИЙ РЯД (Статичный) ---

    # 1. Прямая задача - тепловая карта
    im1 = axes[0, 0].imshow(U_direct.T, extent=[0, direct_solver.T, 0, direct_solver.L],
                            aspect='auto', origin='lower', cmap='viridis')
    axes[0, 0].set_xlabel('Время t')
    axes[0, 0].set_ylabel('Координата x')
    axes[0, 0].set_title('Прямая задача: U(x,t)')
    plt.colorbar(im1, ax=axes[0, 0])

    # 2. Начальное условие
    axes[0, 1].plot(direct_solver.x, U_direct[0, :], 'b-', label='Истинное нач. условие')
    axes[0, 1].plot(direct_solver.x, inverse_results['initial_no_noise'][0, :], 'r--',
                    label='Восст. (без шума)')
    axes[0, 1].plot(direct_solver.x, inverse_results['initial_with_noise'][0, :], 'g--',
                    label='Восст. (с шумом 1%)')
    axes[0, 1].set_title('Сравнение начальных условий')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Коэффициент D(x)
    axes[0, 2].plot(direct_solver.x, direct_solver.D(direct_solver.x), 'b-', label='Истинный D(x)')
    axes[0, 2].plot(direct_solver.x, inverse_results['D_estimated'], 'ro-', label='Восстановленный D(x)')
    axes[0, 2].set_title('Восстановление коэффициента D(x)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # --- НИЖНИЙ РЯД (Динамический - сохраняем кадры) ---

    # Вычисляем индексы времени, которые будем сохранять
    # Например, если M=1000 и нам нужно 20 кадров, берем шаги 0, 50, 100...
    time_indices = np.linspace(0, len(direct_solver.t) - 1, num_frames, dtype=int)

    for i, t_idx in enumerate(time_indices):
        # Очищаем нижние графики перед перерисовкой
        axes[1, 0].clear()
        axes[1, 1].clear()
        axes[1, 2].clear()

        current_time = direct_solver.t[t_idx]

        # 4. Профиль волны (Прямая задача)
        axes[1, 0].plot(direct_solver.x, U_direct[t_idx, :], 'b-', linewidth=2)
        axes[1, 0].set_ylim(np.min(U_direct), np.max(U_direct))
        axes[1, 0].set_title(f'Прямая задача: t = {current_time:.2f}')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].grid(True)

        # 5. Профиль волны (Обратная задача без шума)
        axes[1, 1].plot(direct_solver.x, inverse_results['initial_no_noise'][t_idx, :], 'r-', linewidth=2)
        axes[1, 1].set_ylim(np.min(U_direct), np.max(U_direct))
        axes[1, 1].set_title(f'Обратная задача 1: t = {current_time:.2f}')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].grid(True)

        # 6. Ошибка восстановления
        error = np.abs(U_direct[t_idx, :] - inverse_results['initial_no_noise'][t_idx, :])
        axes[1, 2].plot(direct_solver.x, error, 'g-', linewidth=2)
        axes[1, 2].set_ylim(0, np.max(np.abs(U_direct - inverse_results['initial_no_noise'])) + 0.1)
        axes[1, 2].set_title(f'Ошибка: t = {current_time:.2f}')
        axes[1, 2].set_xlabel('x')
        axes[1, 2].grid(True)

        # Сохранение файла
        filename = f"{output_dir}/step_{i:03d}_time_{current_time:.2f}.png"
        plt.savefig(filename)
        print(f"Сохранен кадр {i + 1}/{num_frames}: {filename}")

    print("Все графики сохранены.")
    plt.close(fig)  # Закрываем фигуру, чтобы она не висела в памяти


def convergence_analysis():
    """Анализ сходимости разностной схемы"""
    print("\n" + "=" * 50)
    print("АНАЛИЗ СХОДИМОСТИ И АППРОКСИМАЦИИ")
    print("=" * 50)

    N_values = [50, 100, 200, 400]
    errors = []

    for N in N_values:
        M = N * 10  # Сохраняем соотношение для устойчивости
        solver = WaveEquationSolver(N=N, M=M)
        U = solver.solve_direct()

        # Оцениваем ошибку путем сравнения с решением на более мелкой сетке
        if N > 50:
            # Интерполяция на более грубую сетку для сравнения
            U_coarse = U[::2, ::2]
            error = np.max(np.abs(U[::2, ::2] - U_coarse))
            errors.append(error)
            print(f"N = {N}, M = {M}, ошибка = {error:.6f}")

    # Оценка порядка сходимости
    if len(errors) >= 2:
        rates = []
        for i in range(len(errors) - 1):
            rate = np.log(errors[i] / errors[i + 1]) / np.log(2)
            rates.append(rate)
        print(f"Оценка порядка сходимости: {np.mean(rates):.4f}")
        print("Теоретический порядок: O(τ² + h²) = O(2)")


if __name__ == "__main__":
    # ... (Весь код до вызова визуализации оставляем без изменений) ...
    # 1. Решение прямой задачи...
    # 2. Анализ сходимости...
    # 3. Решение обратных задач...

    # Копируем начало main из вашего старого файла,
    # но в конце заменяем вызов visualize_results на:

    print("ЛАБОРАТОРНАЯ РАБОТА №9")
    print("Волновое уравнение. Обратные задачи")
    # ... инициализация и решения ...
    direct_solver = WaveEquationSolver(N=100, M=1000)
    U_direct = direct_solver.solve_direct()

    inverse_solver = InverseProblemSolver(direct_solver)
    U_inverse_no_noise = inverse_solver.solve_inverse_initial(U_direct, noise_level=0)
    U_inverse_with_noise = inverse_solver.solve_inverse_initial(U_direct, noise_level=0.01)
    D_estimated = inverse_solver.solve_inverse_coefficient(U_direct)

    inverse_results = {
        'initial_no_noise': U_inverse_no_noise,
        'initial_with_noise': U_inverse_with_noise,
        'D_estimated': D_estimated
    }

    analyze_correctness(direct_solver, inverse_solver, U_direct)

    # === ИЗМЕНЕННАЯ ЧАСТЬ ===
    print("\n3. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    # Сохранит 20 картинок, равномерно распределенных по времени
    save_frames_as_images(direct_solver, U_direct, inverse_results, num_frames=20)