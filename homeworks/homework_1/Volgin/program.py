import math
import json
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline



if __name__ == "__main__":
    with open('3.json', 'r', encoding='utf-8') as file:
        text = dict(json.load(file))

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

    def calc_pwf(*args: json) -> list[float]:
        q = np.linspace(1, 400, 39) # Массив дебитов
        p_wf_data = [] # Список для забойных давлений
        a = dict(*args)
        ws = calc_ws(a['gamma_water']) # Cолесодержание
        t = (a['t_wh'] + (a['temp_grad'] * a['md_vdp']) / 100) + 273 # Температура на забое
        mu = calc_mu_w(ws, t, a['p_wh'] * 1.01325) # Вязкость
        for i in q:
            n_re = calc_n_re(a['gamma_water'] * 1000, i / 86400, mu, a['d_tub'])
            ff = calc_ff_churchill(n_re, a['roughness'], a['d_tub'])
            p_wf = a['p_wh'] * 1.01325 + 1 / 10 ** 5 * (a['gamma_water'] * 1000 * 9.8 *
                                                        math.cos((math.radians(a['angle'])))
                                              - (8 * ff * a['gamma_water'] * 1000 * (i / 86400) ** 2 * a['md_vdp'])
                                              / (math.pi ** 2 * a['d_tub'] ** 5))
            p_wf_data.append(p_wf)
        return p_wf_data

    q = list(np.linspace(1, 400, 39))
    plt.plot(q, calc_pwf(text))
    plt.title('Зависимость забойного давления от дебита закачиваемой жидкости')
    plt.xlabel("Дебит, м3/сут")
    plt.ylabel('Забойное давление, Бар')
    plt.show()
    to_json = {'q_liq': q, 'p_wf': calc_pwf(text)}

    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(to_json, f, sort_keys=False, indent=2)