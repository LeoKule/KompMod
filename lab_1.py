import numpy as np
import matplotlib.pyplot as plt

# Параметры системы
g = 9.81  # ускорение свободного падения, м/с^2
m = 0.1  # масса шарика, кг


# Функция для расчета общего времени процесса
def total_time(h0, alpha, g, m):
    h = h0
    # Время первого падения
    total_t = np.sqrt(2 * h0 / g)

    # Минимальная высота для остановки
    epsilon = 1e-6

    while h > epsilon:
        # Вычисляем потерю энергии (Дж)
        delta_E = alpha * (2 * g * h) ** (1 / 4)
        # Новая высота после потерь
        h_new = h - delta_E / (m * g)

        # Если высота стала отрицательной или слишком малой, выходим из цикла
        if h_new <= epsilon:
            break

        # Время полного цикла (подскок и падение) для этой высоты
        cycle_time = 2 * np.sqrt(2 * h_new / g)
        total_t += cycle_time

        # Обновляем высоту для следующей итерации
        h = h_new

    return total_t


# Создаем диапазон начальных высот
h0_range = np.linspace(0.5, 10, 100)  # от 0.5 до 10 метров

# Задаем несколько значений константы потерь alpha
alpha_values = [0.005, 0.01, 0.02, 0.05]

# Строим график
plt.figure(figsize=(10, 6))

for alpha in alpha_values:
    T_values = [total_time(h0, alpha, g, m) for h0 in h0_range]
    plt.plot(h0_range, T_values, label=f'α = {alpha} Дж·с¹ᐧ²/м¹ᐧ²')

plt.xlabel('Начальная высота h₀, м')
plt.ylabel('Общее время процесса T, с')
plt.title('Зависимость времени подпрыгивания шарика от начальной высоты\nи константы потерь α')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.ylim(bottom=0)  # Время не может быть отрицательным
plt.show()