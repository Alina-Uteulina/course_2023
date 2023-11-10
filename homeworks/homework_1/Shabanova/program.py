# Загрузка библиотек необходимых для отрисовки графиков
import matplotlib
import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp
import json

Q = [i for i in range(0, 410, 10)]
q_m3_sec = [x / 86400 for x in Q]

with open('19.json', 'r') as f:
    data = json.load(f)

gamma_water = data['gamma_water']
md_vdp = data['md_vdp']
d_tub = data['d_tub']
angle = data['angle']
roughness = data['roughness']
p_wh = data['p_wh']
t_wh = data['t_wh']
temp_grad = data['temp_grad']


rho_w = 1000
ksi = 1e-6
L = md_vdp
H = L * np.cos(angle)


def calc_ws(
        gamma_wat: float
) -> float:
    """
    Функция для расчета солесодержания в воде

    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.

    :return: солесодержание в воде, г/г
    """
    ws = (
            1 / (gamma_wat * 1000)
            * (1.36545 * gamma_wat * 1000 - (3838.77 * gamma_wat * 1000 - 2.009 * (gamma_wat * 1000) ** 2) ** 0.5)
    )
    print(ws)
    # если значение отрицательное, значит скорее всего плотность ниже допустимой 992 кг/м3
    if ws > 0:
        return ws
    else:
        return 0


def calc_mu_w(
        ws: float,
        t: float,
        p: float
) -> float:
    """
    Функция для расчета динамической вязкости воды по корреляции Matthews & Russel

    :param ws: солесодержание воды, г/г
    :param t: температура, К
    :param p: давление, Па

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
            * (0.9994 + 0.0058 * (p * 1e-6) + 0.6534 * 1e-4 * (p * 1e-6) ** 2)
    )
    return mu_w


def calc_n_re(
        rho_w: float,
        q_ms: float,
        mu_w: float,
        d_tub: float
) -> float:
    """
    Функция для расчета числа Рейнольдса

    :param rho_w: плотность воды, кг/м3
    :param q_ms: дебит жидкости, м3/с
    :param mu_w: динамическая вязкость воды, сПз
    :param d_tub: диаметр НКТ, м

    :return: число Рейнольдса, безразмерн.
    """
    v = q_ms / (np.pi * d_tub ** 2 / 4)
    return rho_w * v * d_tub / mu_w * 1000


def calc_ff_churchill(
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
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16

    ff = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)
    return ff


def calc_p_wf(
        p_wh: float,
        ksi: float,
        rho_w: float,
        L: float,
        angle: float,
        ff: float,
        d_tub: float,
        q_ms: float
) -> float:
    p_wf = p_wh + ksi * (rho_w * 9.81 * L * np.cos(angle) - 8 / np.pi ** 2 * ff * rho_w * q_ms ** 2 * L / d_tub ** 5)
    return p_wf

mu_w_res = calc_mu_w(calc_ws(gamma_water), temp_grad*H/np.cos(angle)/100 + 273, 1*101325)
print(mu_w_res)
calc_n_re_res = [calc_n_re(rho_w, q_ms = q_m3_sec[i], mu_w = mu_w_res, d_tub = d_tub) for i in range(len(q_m3_sec))]
# print(temp_grad*H/np.cos(angle))
ff_churchill_res = [calc_ff_churchill(calc_n_re_res[i], roughness, d_tub) for i in range(len(calc_n_re_res))]

print([calc_p_wf(p_wh, ksi, rho_w, L, angle, ff_churchill_res[i], d_tub, q_m3_sec[i]) for i in range(len(q_m3_sec))])

calc_p_wf_res = [calc_p_wf(p_wh, ksi, rho_w, L, angle, ff_churchill_res[i], d_tub, q_m3_sec[i]) for i in range(len(q_m3_sec))]

plt.plot(Q, calc_p_wf_res)
plt.show()

output_dict = {'q_liq': Q, 'p_wf': calc_p_wf_res}
with open('output.json', 'w') as f:
    json.dump(output_dict, f)