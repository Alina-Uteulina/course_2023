import json
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d

INPUT_DATA_PATH = Path(r"input_data.json")
OUTPUT_DATA_PATH = Path(r"output_data.json")


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


def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:
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


def calc_ff_jain(n_re: float, roughness: float, d_tub: float) -> float:
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
        ff = 1 / (1.14 - 2 * np.log10(roughness / d_tub + 21.25 / (n_re ** 0.9))) ** 2
    return ff


def calc_sin_angle(md1: float, md2: float, incl: dict) -> float:
    """
    Расчет синуса угла с горизонталью по интерполяционной функции скважины

    Parameters
    ----------
    :param md1: measured depth 1, м
    :param md2: measured depth 2, м
    :param incl: данные инклинометрии

    :return: синус угла к горизонтали
    """
    md = incl["md"]
    tvd = incl["tvd"]
    tube_func = interp1d(md, tvd, fill_value="extrapolate")
    return min((tube_func(md2) - tube_func(md1)) / (md2 - md1), 1)


def calc_angle(md1: float, incl: dict) -> np.ndarray:
    """
    Функция для расчета угла наклона трубы в точке

    :param md1: measured depth 1, м
    :param incl: данные инклинометрии
    
    :return: угол наклона трубы в точке
    """

    md2 = md1 + 0.0001
    return np.degrees(np.arcsin(calc_sin_angle(md1, md2, incl)))


def calc_pressure_grad(rho_w, angle, f, d_tub, q_liq):
    eps = 1 / (10 ** 5)
    grad_p = eps * (rho_w * 9.8 * np.cos(math.radians(90 - angle)) - 0.815 * f * rho_w / d_tub ** 5 * q_liq ** 2)

    return grad_p


def integr_func(hh, pt, gamma_water, q_liq, d_tub, angle, roughness, temp_grad):
    p, t = pt

    ws = calc_ws(gamma_water)
    rho_w = calc_rho_w(ws, t)
    mu_w = calc_mu_w(ws, t, p)
    n_re = calc_n_re(rho_w=rho_w, q_ms=q_liq, mu_w=mu_w, d_tub=d_tub)
    f = calc_ff_churchill(n_re=n_re, roughness=roughness, d_tub=d_tub)

    # Расчет градиента давления, используя необходимую гидравлическую корреляцию
    dp_dl = calc_pressure_grad(rho_w, angle, f, d_tub, q_liq)

    # Геотермический градиент
    dt_dl = temp_grad

    return dp_dl, dt_dl


def main():
    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as input_data:
        input_data_dict = json.load(fp=input_data)

    gamma_water = input_data_dict['gamma_water']  # Относительная плотность воды по пресной воде с плотностью 1000 кг/м3
    md_vdp = input_data_dict['md_vdp']  # Измеренная глубина верхних дыр перфорации, м
    d_tub = input_data_dict['d_tub']  # Диаметр НКТ, м
    angle = input_data_dict['angle']  # Угол наклона скважины к горизонтали, градусы цельсия
    roughness = input_data_dict['roughness']  # Шероховатость, м
    p_wh = input_data_dict['p_wh']  # Буферное давление, атм
    t_wh = input_data_dict['t_wh'] + 273  # Температура жидкости у буферной задвижки, К
    temp_grad = input_data_dict['temp_grad'] * 0.01  # Геотермический градиент, градусы цельсия/100 м

    p_wf_list = []
    q_list = list(np.linspace(1, 400, num=50, endpoint=True))
    for q in q_list:
        q_liq = q / 86400
        res = solve_ivp(
            integr_func,
            t_span=(0, md_vdp),
            y0=[p_wh, t_wh],
            method='RK23',
            args=(gamma_water, q_liq, d_tub, angle, roughness, temp_grad),
            t_eval=[md_vdp]
        )
        p_wf_list.append(res.y[0][0])
    plt.plot(p_wf_list, q_list)
    plt.xlabel('Забойное давление, атм')
    plt.ylabel('Дебит, м3/сут')
    plt.show()
    result = {'p_wf': p_wf_list, 'q_liq': q_list}
    
    with open(OUTPUT_DATA_PATH, "w", encoding="utf-8") as file:
        json.dump(result, file)


if __name__ == "__main__":
    main()
