import math
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
# %matplotlib inline


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


def calc_mu_w(
        ws: float,
        t_data: list[float],
        p: float
) -> list[float]:
    """
    Функция для расчета динамической вязкости воды по корреляции Matthews & Russel

    :param t_data: Список распределения температуры по стволу НКТ, К
    :param ws: солесодержание воды, г/г
    :param p: давление, Па

    :return: Список динамической вязкости воды, сПз
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
    mu_w_data = []
    for temp in t_data:
        mu_w = (
                a * (1.8 * temp - 460) ** (-b)
                * (0.9994 + 0.0058 * (p * 1e-6) + 0.6534 * 1e-4 * (p * 1e-6) ** 2)
        )
        mu_w_data.append(mu_w)
    return mu_w_data


def calc_n_re(
        rho_w_data: list[float],
        q_ms: float,
        mu_w_data: list[float],
        d_tub: float
) -> float:
    """
    Функция для расчета числа Рейнольдса

    :param rho_w_data: Список плотностей воды, кг/м3
    :param q_ms: дебит жидкости, м3/с
    :param mu_w_data: Список динамической вязкости воды, сПз
    :param d_tub: диаметр НКТ, м

    :return: Список чисел Рейнольдса, безразмерн.
    """
    v = q_ms / (np.pi * d_tub ** 2 / 4)
    # rho_w_and_mu_w_data = zip(rho_w_data, mu_w_data)
    n_re_data = []
    for rho_w, mu_w in zip(rho_w_data, mu_w_data):
        n_re_step = rho_w * v * d_tub / mu_w * 1000
        n_re_data.append(n_re_step)
    return n_re_data


def calc_ff_churchill(
        n_re_data: list[float],
        roughness: float,
        d_tub: float
) -> list[float]:
    """
    Функция для расчета коэффициента трения по корреляции Churchill

    :param n_re_data: Список чисел Рейнольдса, безразмерн.
    :param roughness: шероховатость стен трубы, м
    :param d_tub: диаметр НКТ, м

    :return: Список коэффициентов трения, безразмерн.
    """
    ff_data = []
    for n_re in n_re_data:
        a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
        b = (37530 / n_re) ** 16
        ff_step = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1 / 12)
        ff_data.append(ff_step)
    return ff_data


def calc_temperature(
        **kwargs: dict[str: float]
) -> list[float]:
    """
    Фунцкия для расчета температуры по стволу скважины

    :param kwargs: Словарь различных характеристик скважины и закачиваемой жидкости

    :return: Список распределения температуры по стволу скважины
    """
    t_begin = kwargs['t_wh']
    t_data = []
    for i in range(int(kwargs['md_vdp'] / 100) + 1):
        t_step = (t_begin + i * kwargs['temp_grad']) + 273
        t_data.append(t_step)
    return t_data


def calc_rho_w(
        ws: float,
        t_data: list[float]
) -> list[float]:
    """
    Функция для расчета плотности воды в зависимости от температуры и солесодержания

    :param t_data: Список температур
    :param ws: солесодержание воды, г/г

    :return: Список плотностей воды, кг/м3
    """
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)
    rho_w_data = []
    for temperature in t_data:
        rho_step = rho_w / (1 + (temperature - 273) * 1e-4 * (0.269 * (temperature - 273) ** 0.637 - 0.8))
        rho_w_data.append(rho_step)
    return rho_w_data


def calc_pwf(
        **kwargs: dict[str: float]
) -> list[float]:
    """
    Функция для расчета забойного давления

    :param: kwargs: Словарь различных характеристик скважины и закачиваемой жидкости

    :return: Список забойных давлений от диапазона дебитов жидкости для генерации VLP [1 - 400]м3/сут
    """
    q = np.linspace(1, 400, 21)  # Массив дебитов
    p_wf_data = []  # Список для забойных давлений
    ws = calc_ws(kwargs['gamma_water'])  # Cолесодержание
    t_data = calc_temperature(**text)
    mu_data = calc_mu_w(ws, t_data, kwargs['p_wh'] * 1.01325)  # Вязкость
    rho_w_data = calc_rho_w(ws, t_data)

    def dpdl(l: float, p: float):
        """
        :param l: Длина НКТ, м
        :param p: Давление, Бар

        :return: правая часть дифференциального уравнения градиента давления для произвольного участка скважины
        """
        return 1 / 10 ** 5 * (rho_w * 9.8 * math.cos((math.radians(kwargs['angle']))) -
                              ((0.815 * ff * rho_w * (q_step / 86400) ** 2) / kwargs['d_tub'] ** 5))

    for q_step in q:
        n_re_data = calc_n_re(rho_w_data, q_step / 86400, mu_data, kwargs['d_tub'])
        ff_data = calc_ff_churchill(n_re_data, kwargs['roughness'], kwargs['d_tub']) # Коэффициент трения
        for rho_w, ff in zip(rho_w_data, ff_data):
            sol = solve_ivp(dpdl, t_span=(0, kwargs['md_vdp']),
                            y0=[kwargs['p_wh'], kwargs['t_wh']], method='RK23')
        p_wf = sol.y[0][-1]
        p_wf_data.append(p_wf)
    return p_wf_data


if __name__ == "__main__":
    with open('3.json', 'r', encoding='utf-8') as file:
        text = dict(json.load(file))

    q = list(map(lambda x: int(x), np.linspace(1, 400, 21)))
    plt.plot(q, calc_pwf(**text))
    plt.title('Зависимость забойного давления от дебита закачиваемой жидкости')
    plt.xlabel("Дебит, м3/сут")
    plt.ylabel('Забойное давление, Бар')
    plt.show()
    to_json = {'q_liq': q, 'p_wf': calc_pwf(**text)}

    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(to_json, f, sort_keys=False, indent=2)