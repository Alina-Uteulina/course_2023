# -*- coding: utf-8 -*-
"""Untitled11.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RltwrZRoIJULwWTxdXOLe1Z2U2AxhreC
"""

# Commented out IPython magic to ensure Python compatibility.
# Загрузка библиотек необходимых для отрисовки графиков
import matplotlib
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp
# %matplotlib inline
import json
from math import pi, log, radians, cos

def calc_ws(
        gamma_wat: float
) -> float:
    """
    Функция для расчета солесодержания в воде

    :param gamma_water: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.

    :return: солесодержание в воде, г/г
    """
    ws = (
            1 / (gamma_wat * 1000)
            * (1.36545 * gamma_wat * 1000 - (3838.77 * gamma_wat * 1000 - 2.009 * (gamma_wat * 1000) ** 2) ** 0.5)
    )
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if ws > 0:
        return ws
    else:
        return 0

def calc_rho_w(
        ws: float,
        t: float
) -> float:
    """
    Функция для расчета плотности воды в зависимости от температуры и солесодержания

    :param ws: солесодержание воды, г/г
    :param t: температура, К

    :return: плотность воды, кг/м3
    """
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)

    return rho_w / (1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8))

def calc_mu_w(
        ws: float,
        t: float,
        p: float
) -> float:
    """
    Функция для расчета динамической вязкости воды по корреляции Matthews & Russel

    :param ws: солесодержание воды, г/г
    :param t: температура, К
    :param p: давление, МПа

    :return: динамическая вязкость воды, сПз
    """
    a = (
            109.574
            - (0.840564 * 1000 * ws)
            + (3.13314 * 1000 * ws ** 2)
            + (8.72213 * 1000 * ws ** 3)
    )
    b = (
            1.12166
            - 2.63951 * ws
            + 6.79461 * ws ** 2
            + 54.7119 * ws ** 3
            - 155.586 * ws ** 4
    )

    mu_w = (
            a * (1.8 * t - 460) ** (-b)
            * (0.9994 + 0.0058 * (p * 0.101325) + 0.6534 * 1e-4 * (p * 0.101325) ** 2)
    )
    return mu_w

def calc_n_re(
        rho_w: float,
        q_liq: float,
        mu_w: float,
        d_tub: float
) -> float:
    """
    Функция для расчета числа Рейнольдса

    :param rho_w: плотность воды, кг/м3
    :param q_liq: дебит жидкости, м3/с
    :param mu_w: динамическая вязкость воды, сПз
    :param d_tub: диаметр НКТ, м

    :return: число Рейнольдса, безразмерн.
    """
    v = q_liq / (pi * d_tub ** 2 / 4)
    return rho_w * v * d_tub / mu_w * 1000

def calc_f_churchill(
        n_re: float,
        roughness: float,
        d_tub: float
) -> float:
    """
    Функция для расчета коэффициента трения по корреляции Churchill

    :param n_re: число Рейнольдса, безразмерн.
    :param roughness: шероховатость стен трубы, м
    :param d_tub: диаметр НКТ, м

    :return: коэффициент трения, безразмерн.
    """
    a = (-2.457 * log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16

    f = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1/12)
    return f

"""**Расчет распределения давления**"""

def dp(p, l, t,t_wh, temp_grad, md_vdp, gamma_water, roughness,  angle,  d_tub, q_liq):
    """
    Функция для расчета градиента давления для произвольного участка скважины

    :param t_wh: температура жидкости у буферной задвижки, градусы цельсия
    :param temp_grad: геотермический градиент, градусы цельсия/100 м
    :param md_vdp: измеренная глубина верхних дыр перфорации, м
    :param gamma_water: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.
    :param angle: угол наклона скважины к горизонтали, градусы
    :param q_liq: дебит закачиваемой жидкости
    :param d_tub: диаметр НКТ, м

    :return: градиент давления для произвольного участка скважины
    """
    t = (t_wh + (temp_grad * l) / 100) + 273 #учет температурного градиента
    ws = calc_ws(gamma_water) #рассчитанная минерализация
    rho_w = calc_rho_w(ws, t) #рассчитанная плотность от солесод
    g = 9.81
    mu_w = calc_mu_w(ws, t, p)
    n_re = calc_n_re (rho_w, q_liq, mu_w, d_tub)
    f = calc_f_churchill(n_re, roughness, d_tub)
    dp = (1 / 10 ** (6)) * (rho_w * g * mt.cos(radians((angle))) - 0.815 * f * rho_w / (d_tub ** 5) * (q_liq/86400) ** 2)
    return dp

def main(data):
    gamma_water = data['gamma_water']
    md_vdp = data['md_vdp']
    d_tub = data['d_tub']
    angle = data['angle']
    roughness = data['roughness']
    p_wh = data['p_wh']
    t_wh = data['t_wh'] + 273
    temp_grad = data['temp_grad'] * 0.01
    q_liq = np.linspace(1, 400, 41)
    q_liq = [int(q_liq[_]) for _ in range(len(q_liq))]
    q_liq_second = np.array(q_liq)
    p_wf = []
    for _ in q_liq_second:
        sol = solve_ivp(dp,
                        t_span = [0, md_vdp],
                        y0 = [p_wh * 0.101325],
                        args = (
                                t_wh, t_wh, temp_grad, md_vdp,
                                gamma_water, roughness, angle, d_tub, _),
                        t_eval = [md_vdp])
        p_wf.append(sol.y[0][-1] * 9.86923)
    q_liq = list(q_liq)
    q_liq = [int(q_liq[_]) for _ in range(len(q_liq))]
    result = {'q_liq': q_liq, 'p_wf': p_wf}
    plt.plot(q_liq, p_wf)
    plt.title('P_wf~Q')
    plt.xlabel("Закачка, м3/сут")
    plt.ylabel('Забойное давление, атм')
    plt.show()
    return result

if __name__ == "__main__":
    with open('17.json') as file:
        data = json.load(file)
        results = main(data)

    with open(r"output.json", "w", ) as  file:
        json.dump(results, file)