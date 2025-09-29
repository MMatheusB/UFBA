from manim import *
import numpy as np
import casadi as ca
from scipy.optimize import fsolve

# --- Parâmetros do sistema ---
A1 = 2.6e-3
Lc = 2
kv = 0.38
P1 = 4.5
P_out = 5
C = 479

# --- Lookup fictício ---
def lut(args):
    return 1.65  # valor de teste

# --- Função do sistema ---
def fun(variables, alpha, N, lut):
    x, y = variables  # x = vazão, y = pressão
    phi_value = float(lut([N, x]))
    eqn_1 = (A1 / Lc) * ((phi_value * P1) - y) * 1e3
    eqn_2 = ((C**2) / 2) * (x - alpha * kv * np.sqrt(y*1000 - P_out*1000))
    return [eqn_1, eqn_2]

# --- Simulação ---
def simulation(t_max=120):
    y = []
    u = []
    dt = 0.5
    pontos = int(t_max/dt)

    # Condições iniciais
    alphas = [0.5]
    N_RotS = [38500]
    result = fsolve(fun, (10, 10), args=(alphas[0], N_RotS[0], lut))
    init_m, init_p = result

    # Variáveis CasADi
    x_sym = ca.MX.sym('x', 2)
    p_sym = ca.MX.sym('p', 2)
    alpha_sym, N_sym = p_sym[0], p_sym[1]

    rhs = ca.vertcat(
        (A1 / Lc) * ((lut(ca.vertcat(N_sym, x_sym[0])) * P1) - x_sym[1]) * 1e3,
        ((C**2)/2)*(x_sym[0] - alpha_sym*kv*ca.sqrt(x_sym[1]*1000 - P_out*1000))
    )

    ode = {'x': x_sym, 'ode': rhs, 'p': p_sym}
    F = ca.integrator('F', 'cvodes', ode, 0, dt)

    for j in range(pontos):
        t_atual = j*dt
        if t_atual < 20:
            params = [0.5, 38500]
        elif t_atual < 40:
            params = [0.6, 40000]
        elif t_atual < 60:
            params = [0.4, 37000]
        elif t_atual < 80:
            params = [0.55, 36000]
        elif t_atual < 100:
            params = [0.35, 40000]
        else:
            params = [0.65, 42500]

        sol = F(x0=[init_m, init_p], p=params)
        xf_values = np.array(sol["xf"]).flatten()
        init_m, init_p = xf_values
        y.append([init_m, init_p])
        u.append(params[0])

    y = np.array(y)
    u = np.array(u)
    t_array = np.arange(0, t_max, dt)
    return t_array, y[:,0], y[:,1], u

# --- Obter dados ---
t_exp, m_exp, p_exp, alpha_exp = simulation()

# --- Animação ---
class SistemaPressaoVazaoAnim(Scene):
    def construct(self):
        # --- Eixos Vazão ---
        axes_vazao = Axes(
            x_range=[0, 120, 20],
            y_range=[min(m_exp)-0.5, max(m_exp)+0.5, 1],
            x_length=6,
            y_length=3.5,
            axis_config={"include_tip": False},
        ).to_corner(UL)
        axes_vazao_labels = axes_vazao.get_axis_labels("t (s)", "Vazao (kg/s)")
        self.add(axes_vazao, axes_vazao_labels)

        # --- Eixos Pressão ---
        axes_pressao = Axes(
            x_range=[0, 120, 20],
            y_range=[min(p_exp)-0.5, max(p_exp)+0.5, 0.5],
            x_length=6,
            y_length=3.5,
            axis_config={"include_tip": False},
        ).to_corner(UR)
        axes_pressao_labels = axes_pressao.get_axis_labels("t (s)", "Pressao (MPa)")
        self.add(axes_pressao, axes_pressao_labels)

        # --- Pontos ---
        point_vazao = Dot(color=YELLOW).move_to(axes_vazao.c2p(0, m_exp[0]))
        point_pressao = Dot(color=BLUE).move_to(axes_pressao.c2p(0, p_exp[0]))
        self.add(point_vazao, point_pressao)

        # --- Rastros usando TracedPath ---
        trail_vazao = TracedPath(point_vazao.get_center, stroke_color=YELLOW, stroke_width=4)
        trail_pressao = TracedPath(point_pressao.get_center, stroke_color=BLUE, stroke_width=4)
        self.add(trail_vazao, trail_pressao)

        # --- Texto tempo e abertura ---
        # --- Texto tempo e abertura com rótulos ---
        tempo_label = MathTex("t =")
        tempo_value = DecimalNumber(0, num_decimal_places=1)
        tempo_unit = MathTex("s")
        tempo_group = VGroup(tempo_label, tempo_value, tempo_unit).arrange(RIGHT, buff=0.1)

        alpha_label = MathTex(r"\alpha =")
        alpha_value = DecimalNumber(alpha_exp[0], num_decimal_places=2)
        alpha_group = VGroup(alpha_label, alpha_value).arrange(RIGHT, buff=0.1)

        all_texts = VGroup(tempo_group, alpha_group).arrange(DOWN, aligned_edge=LEFT).to_corner(DR, buff=0.5)

        self.add(all_texts)

        current_t = [0]
        current_alpha = [alpha_exp[0]]

        def update_texts(obj):
            tempo_value.set_value(current_t[0])
            alpha_value.set_value(current_alpha[0])

        tempo_value.add_updater(update_texts)
        alpha_value.add_updater(update_texts)



        # --- Animação dos pontos (mais lenta) ---
        dt_anim = 0.1  # mais lenta
        for i in range(len(t_exp)):
            new_point_vazao = axes_vazao.c2p(t_exp[i], m_exp[i])
            new_point_pressao = axes_pressao.c2p(t_exp[i], p_exp[i])
            current_t[0] = t_exp[i]
            current_alpha[0] = alpha_exp[i]
            self.play(
                point_vazao.animate.move_to(new_point_vazao),
                point_pressao.animate.move_to(new_point_pressao),
                run_time=dt_anim,
                rate_func=linear
            )
