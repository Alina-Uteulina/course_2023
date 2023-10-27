import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from scipy.integrate import odeint, solve_ivp
import math
import json
# основные корреляции



def calc_ws(gamma_wat: float) -> float:
    """
    Функция для расчета солесодержания в воде

    :param gamma_wat: относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм.

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


def calc_rho_w(ws: float, t: float) -> float:
    """
    Функция для расчета плотности воды в зависимости от температуры и солесодержания

    :param ws: солесодержание воды, г/г
    :param t: температура, К

    :return: плотность воды, кг/м3
    """
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)

    return rho_w / (1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8))


def calc_mu_w(ws: float, t: float, p: float) -> float:
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


def calc_n_re(rho_w: float, q_ms: float, mu_w: float, d_tub: float) -> float:
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

    ff = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1/12)
    return ff

def calc_ff_jain(
        n_re: float,
        roughness: float,
        d_tub: float
) -> float:
    """
    Функция для расчета коэффициента трения по корреляции Jain

    :param n_re: число Рейнольдса, безразмерн.
    :param roughness: шероховатость стен трубы, м
    :param d_tub: диаметр НКТ, м

    :return: коэффициент трения, безразмерн.
    """
    if n_re < 3000:
        ff = 64 / n_re
    else:
        ff = 1 / (1.14 - 2 * np.log10(roughness / d_tub + 21.25 / (n_re**0.9))) ** 2
    return ff

def calc_sin_angle(md1: float, md2: float, incl:dict) -> float:
    """
    Расчет синуса угла с горизонталью по интерполяционной функции скважины

    Parameters
    ----------
    :param md1: measured depth 1, м
    :param md2: measured depth 2, м

    :return: синус угла к горизонтали
    """
    md = incl["md"]
    tvd = incl["tvd"]
    tube_func = interp1d(md, tvd, fill_value="extrapolate")
    return min((tube_func(md2) - tube_func(md1)) / (md2 - md1), 1)

def calc_angle(md1, incl):
    """
    Функция для расчета угла наклона трубы в точке

    :param md1: measured depth 1, м
    """

    md2 = md1 + 0.0001
    return np.degrees(np.arcsin(calc_sin_angle(md1, md2, incl)))

# расчет давления

# входные параметры


def calc_pressure_grad(p, h, gamma_water, q_liq, d_tub, angle, roughness, temp_grad, t_wh):
    # step - глубина, на которую рассчитываем
    eps = 1 / (10 ** 5)
    ws = calc_ws(gamma_water)
    t = t_wh + temp_grad * h
    rho_w = calc_rho_w(ws, t)
    mu_w = calc_mu_w(ws, t, p)
    n_re = calc_n_re(rho_w=rho_w, q_ms=q_liq, mu_w=mu_w, d_tub=d_tub)
    f = calc_ff_churchill(n_re=n_re, roughness=roughness, d_tub=d_tub)
    grad_p = eps * (rho_w * 9.8 * np.cos(math.radians(90 - angle)) - 0.815 * f * rho_w / d_tub ** 5 * q_liq ** 2)

    return grad_p



def main_function(data):

    gamma_water = data['gamma_water'] # относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм
    md_vdp = data['md_vdp'] # измеренная глубина забоя скважины
    d_tub = data['d_tub'] # диаметр НКТ, м
    angle = data['angle']# угол наклона скважины к горизонтали, градусы
    roughness = data['roughness'] # шероховатость трубы, м
    p_wh = data['p_wh'] # давление на устье, Па
    t_wh = data['t_wh'] + 273 # температура на устье скважины, K
    temp_grad = data['temp_grad'] * 0.01 # геотермический градиент, К/м

    # проверочные данные из семинара (сходится)
    # Q = 800
    # q_liq = Q / 86400
    # gamma_water = 1.015 # относительная плотность по пресной воде с плотностью 1000 кг/м3, безразм
    # md_vdp = 2500 # измеренная глубина забоя скважины
    # d_tub = 0.068 # диаметр НКТ, м
    # angle = 0 # угол наклона скважины к горизонтали, градусы
    # roughness = 0.0001 # шероховатость трубы, м
    # p_wh = 100  # давление на устье, атм
    # t_wh = 34 + 273 # температура на устье скважины, С
    # temp_grad = 3 * 0.01 # геотермический градиент, К/м * (1e-2)
    p_wf = []
    q_ = []
    for q in np.arange(1, 400):
        q_liq = q / 86400
        res = solve_ivp(
            calc_pressure_grad,
            t_span=[0, md_vdp],
            y0=[p_wh],
            method='RK23',
            args=(gamma_water, q_liq, d_tub, angle, roughness, temp_grad, t_wh),
            t_eval=[md_vdp]
        )
        p_wf.append(res.y[0][0])
        q_.append(int(q))
    plt.plot(p_wf, np.arange(1, 400))
    # ax = plt.gca()
    # ax.invert_yaxis()
    plt.ylabel('Дебит, м3/сут')
    plt.xlabel('Забойное давление, атм')
    plt.show()
    print(q_)
    result = {'p_wf': p_wf, 'q_liq': q_}
    return result


if __name__ == "__main__":

    with open('3.json') as file:
        data = json.load(file)
        res = main_function(data)

    with open(r"output.json", "w", ) as file:
        json.dump(res, file)
