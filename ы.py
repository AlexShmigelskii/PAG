import math
from math import cos, log

import numpy as np

import approximated


# Теперь мы располагаем всеми необходимыми зависимостями
# Реализация функций расчета времени в разных режимах

def razgon(H, V1, V2, mass, S, phi=0):
    """
    Режим разгон
    :param H: высота полета = const
    :param V1: начальная скорость
    :param V2: конечная скорость
    :param mass: масса ЛА
    :param S: приведенная площадь
    :param phi: угол наклона двигателя
    :return: время для режима "разгон"
    """
    V = (V1 + V2) / 2

    ro = approximated.ro(H)
    g = approximated.g(H)
    sound_of_speed = approximated.sound_speed(H)
    M = V / sound_of_speed
    P = approximated.P_h(M, H)
    # Сy(alpha) = 0.0834*alpha + 0.3579
    Cya = 0.0834
    Cy0 = 0.3579
    alpha = (mass * g - (P * phi / 57.3) - Cy0 * ro * V ** 2 * S / 2) / (P / 57.3 + Cya * ro * V ** 2 * S / 2)
    alpha_radians = math.radians(alpha)

    Cy = approximated.Cy(alpha)
    Cx = approximated.Cx(Cy, M)
    return (mass * (V2 - V1)) / (P * cos(alpha_radians + phi) - Cx * ro * V ** 2 * S / 2)


def pod(H1, H2, V, mass, S, phi=0):
    """
    Режим подъём
    :param H1: начальная высота полета
    :param H2: конечная высота полета
    :param V: скорость ЛА = const
    :param mass: масса ЛА
    :param S: приведенная площадь
    :param phi: угол наклона двигателя
    :return: время для режима "подъём"
    """
    H = (H1 + H2) / 2

    ro = approximated.ro(H)
    g = approximated.g(H)
    speed_of_sound = approximated.sound_speed(H)
    M = V / speed_of_sound
    P = approximated.P_h(M, H)
    # Сy(alpha) = 0.0834*alpha + 0.3579
    Cya = 0.0834
    Cy0 = 0.3579
    alpha = (mass * g - (P * phi / 57.3) - Cy0 * ro * V ** 2 * S/ 2) / (P / 57.3 + Cya * ro * V ** 2 * S / 2)

    Cy = approximated.Cy(alpha)
    Cx = approximated.Cx(Cy, M)
    X = Cx * ro * V ** 2 * S / 2
    teta = ((P - X) * 57.3) / (mass * g)
    return (57.3 * (H2 - H1)) / (V * teta)


def raz_pod(H1, H2, V1, V2, mass, S, phi=0):
    """
    Режим разгон-подъём
    :param H1: начальная высота полёта
    :param H2: конечаня высота полёта
    :param V1: начальная скорость ЛА
    :param V2: конечная скорость ЛА
    :param mass: масса ЛА
    :param S: приведенная площадь
    :param phi: угол наклона двигателя
    :return: время для режима "разгон-подъём"
    """
    V = (V1 + V2) / 2
    H = (H1 + H2) / 2

    ro = approximated.ro(H)
    g = approximated.g(H)
    speed_of_sound = approximated.sound_speed(H)
    M = V / speed_of_sound
    P = approximated.P_h(M, H)
    # Сy(alpha) = 0.0834*alpha + 0.3579
    Cya = 0.0834
    Cy0 = 0.3579
    alpha = (mass * g - (P * phi / 57.3) - Cy0 * ro * V ** 2 * S / 2) / (P / 57.3 + Cya * ro * V ** 2 * S / 2)
    alpha_radians = math.radians(alpha)

    Cy = approximated.Cy(alpha)
    Cx = approximated.Cx(Cy, M)

    k = (V2 - V1) / (H2 - H1)
    X = Cx * ro * V ** 2 * S / 2
    sin_teta = (P * cos(alpha_radians + phi) - X) / (mass * (k * V + g))
    return (1 / (k * sin_teta) * log(V2 / V1))

import pandas as pd

# Параметры ЛА
mass = 47000 # кг
S = 127 # м^2
phi = 0

n = 10
Vn = 350 * 1000 / 3600  # из км/ч в м/с
Vk = 880 * 1000 / 3600
Hn = 500
Hk = 8000
deltaV = (Vk - Vn) / n  # приращение скорости
deltaH = (Hk - Hn) / n  # приращение высоты

# Создаем временные матрицы Tr, Tp, Trp с избежанием отрицательных значений
Tr = np.empty((n, n + 1), dtype=object)  # Матрица Tr
Tp = np.empty((n + 1, n), dtype=object)  # Матрица Tp
Trp = np.empty((n, n), dtype=object)  # Матрица Trp

# Заполняем матрицу Tr
for i in range(n + 1):
    Hi = Hn + i * deltaH
    for j in range(n):
        Vi = Vn + j * deltaV
        Traz = razgon(Hi, Vi, Vi + deltaV, mass, S, phi)
        Tr[j, i] = Traz

# Заполняем матрицу Tp
for i in range(n + 1):
    Vi = Vn + i * deltaV
    for j in range(n):
        Hi = Hn + j * deltaH
        Tpod = pod(Hi, Hi + deltaH, Vi, mass, S, phi)
        Tp[i, j] = Tpod

# Заполняем матрицу Trp
for i in range(n):
    Vi = Vn + i * deltaV
    for j in range(n):
        Hi = Hn + j * deltaH
        Traz_pod = raz_pod(Hi, Hi + deltaH, Vi, Vi + deltaV, mass, S, phi)
        Trp[i, j] = Traz_pod


df_Tr = pd.DataFrame(Tr)
df_Tp = pd.DataFrame(Tp)
df_Trp = pd.DataFrame(Trp)

print("Матрица Tr:")
print(df_Tr.to_string(index=False))  # Использую to_string() для вывода в виде текста
print("\nМатрица Tp:")
print(df_Tp.to_string(index=False))
print("\nМатрица Trp:")
print(df_Trp.to_string(index=False))