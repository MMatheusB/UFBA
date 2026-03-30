# tank_manim.py
from manim import *
import numpy as np
from scipy.integrate import solve_ivp


def temp_to_color(T, Tmin, Tmax):
    # clamp and map to [0,1]
    if Tmax <= Tmin:
        alpha = 0.5
    else:
        alpha = float((T - Tmin) / (Tmax - Tmin))
        alpha = max(0.0, min(1.0, alpha))
    # interpolate blue (cold) -> yellow -> red (hot)
    if alpha < 0.5:
        # blue -> yellow
        return interpolate_color(BLUE, YELLOW, alpha * 2)
    else:
        # yellow -> red
        return interpolate_color(YELLOW, RED, (alpha - 0.5) * 2)


class IntroTanque(Scene):
    def construct(self):
        title = Text("Tanque Cônico com Entrada Bombeada", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(0.8)

        intro = VGroup(
            Text("Modelo físico: entrada bombeada (q_in)", font_size=26),
            Text("Saída dependente do nível (gravidade / descarga)", font_size=26),
            Text("Objetivo: modelar nível L(t) e temperatura T(t)", font_size=26)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).next_to(title, DOWN, buff=0.8)

        for line in intro:
            self.play(FadeIn(line, shift=RIGHT), run_time=0.7)
            self.wait(0.25)

        self.wait(1.2)
        self.play(FadeOut(intro), FadeOut(title))
        self.wait(0.5)


class PremissasTanque(Scene):
    def construct(self):
        title = Text("Premissas do Modelo", font_size=36, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(0.6)

        itens = [
            "Fluido incompressível e densidade constante",
            "Saída controlada pela altura (comportamento tipo Torricelli)",
            "Área de seção transversal constante (A)",
            "Entradas medidas: vazão q_in e temperatura T_in",
        ]

        # Apresenta um por um
        y = title.get_bottom() + DOWN * 0.6
        for txt in itens:
            t = Text(txt, font_size=28).next_to(title, DOWN, buff=0.6)
            self.play(FadeIn(t, shift=RIGHT))
            self.wait(0.9)
            title = t  # next item appears below previous
        self.wait(1.2)
        self.play(FadeOut(*self.mobjects))
        self.wait(0.3)


class EquacoesTanque(Scene):
    def construct(self):
        title = Text(
            "Derivação do Modelo Dinâmico",
            font_size=38,
            color=BLUE
        ).to_edge(UP)

        self.play(Write(title))
        self.wait(0.6)

        # --------------------------------------------------
        # 1) Aproximação da velocidade de saída
        # --------------------------------------------------
        s1 = Text(
            "1) Aproximação da velocidade na saída",
            font_size=30,
            color=YELLOW
        ).next_to(title, DOWN, buff=0.6)

        t1 = Text(
            "Desprezamos a dinâmica de momento no duto de saída,\n"
            "assumindo regime quase-estacionário.",
            font_size=26
        ).next_to(s1, DOWN, buff=0.4)

        eq_vp0 = MathTex(
            r"\frac{d v_p(t)}{dt} = 0"
        ).next_to(t1, DOWN, buff=0.4)

        self.play(FadeIn(s1, shift=RIGHT))
        self.play(FadeIn(t1, shift=RIGHT))
        self.play(Write(eq_vp0))
        self.wait(1.8)

        self.play(FadeOut(t1), FadeOut(eq_vp0))

        # --------------------------------------------------
        # 2) Balanço de momento → relação algébrica
        # --------------------------------------------------
        t2 = Text(
            "Aplicando essa hipótese ao balanço de momento:",
            font_size=26
        ).next_to(s1, DOWN, buff=0.4)

        eq_mom = MathTex(
            r"m_p \frac{d v_p}{dt} = \gamma A_p h(t) - k_f v_p^2(t)"
        ).next_to(t2, DOWN, buff=0.4)

        eq_mom_ss = MathTex(
            r"0 = \gamma A_p h(t) - k_f v_p^2(t)"
        ).next_to(eq_mom, DOWN, buff=0.4)

        eq_vp = MathTex(
            r"v_p(t) = \sqrt{\frac{\gamma A_p}{k_f}}\, h(t)"
        ).next_to(eq_mom_ss, DOWN, buff=0.4)

        self.play(FadeIn(t2, shift=RIGHT))
        self.play(Write(eq_mom))
        self.wait(1.0)
        self.play(Write(eq_mom_ss))
        self.wait(1.0)
        self.play(Write(eq_vp))
        self.wait(2.0)

        self.play(FadeOut(s1), FadeOut(t2), FadeOut(eq_mom), FadeOut(eq_mom_ss), FadeOut(eq_vp))

        # --------------------------------------------------
        # 3) Balanço de massa → equação do nível
        # --------------------------------------------------
        s2 = Text(
            "2) Balanço de massa do tanque",
            font_size=30,
            color=YELLOW
        ).to_edge(UP)

        eq_mass = MathTex(
            r"\frac{dL(t)}{dt} = \frac{Q_{in}(t) - A_p v_p(t)}{A}"
        )

        eq_subs = MathTex(
            r"\frac{dL(t)}{dt} = \frac{Q_{in}(t) - A_p \sqrt{\frac{\gamma A_p}{k_f}}\, L(t)}{A}"
        )

        eq_alpha = MathTex(
            r"\alpha = A_p \sqrt{\frac{\gamma A_p}{k_f}}"
        )

        eq_level = MathTex(
            r"\frac{dL(t)}{dt} = \frac{Q_{in}(t) - \alpha\, \sqrt{L(t)}}{A}"
        )

        # Organiza em duas colunas
        eqs = VGroup(
            eq_mass,
            eq_subs,
            eq_alpha,
            eq_level
        ).arrange_in_grid(
            rows=2,
            cols=2,
            buff=0.6,
            aligned_edge=LEFT
        ).next_to(s2, DOWN, buff=0.6)

        self.play(FadeIn(s2, shift=RIGHT))
        self.play(LaggedStart(
            Write(eq_mass),
            Write(eq_subs),
            Write(eq_alpha),
            Write(eq_level),
            lag_ratio=0.35
        ))
        self.wait(2.0)

        self.play(FadeOut(s2), FadeOut(eqs))

        # --------------------------------------------------
        # 4) Balanço de energia → equação da temperatura
        # --------------------------------------------------
        s3 = Text(
        "3) Balanço de energia do tanque",
        font_size=30,
        color=YELLOW
        ).next_to(title, DOWN, buff=0.6)

        self.play(FadeIn(s3, shift=RIGHT))
        self.wait(0.4)

        # -------------------------------
        # COLUNA ESQUERDA
        # -------------------------------
        eq_energy = MathTex(
            r"\frac{dU}{dt} = \rho q_{in} h_{in} - \rho q h + \dot{Q}"
        ).scale(0.9)

        eq_U = MathTex(
            r"U \approx H = m h = \rho A L h"
        ).scale(0.9)

        eq_prod = MathTex(
            r"\rho A L \frac{dh}{dt}"
            r"= \rho q_{in}(h_{in} - h) + \dot{Q}"
        ).scale(0.9)

        left_col = VGroup(eq_energy, eq_U, eq_prod)\
            .arrange(DOWN, aligned_edge=LEFT, buff=0.4)\
            .to_edge(LEFT, buff=1.0)\
            .shift(DOWN*0.6)

        # -------------------------------
        # COLUNA DIREITA
        # -------------------------------
        eq_cp = MathTex(
            r"h = c_p T,\quad h_{in} = c_p T_{in}"
        ).scale(0.9)

        eq_Q = MathTex(
            r"\dot{Q} = \rho_j q_j \lambda_j"
        ).scale(0.9)

        eq_T = MathTex(
            r"\frac{dT}{dt} = "
            r"\frac{\rho q_{in} c_p (T_{in} - T) + \rho_j q_j \lambda_j}"
            r"{\rho A L c_p}"
        ).scale(0.9)

        right_col = VGroup(eq_cp, eq_Q, eq_T)\
            .arrange(DOWN, aligned_edge=LEFT, buff=0.5)\
            .to_edge(RIGHT, buff=1.0)\
            .shift(DOWN*0.6)

        # -------------------------------
        # ANIMAÇÃO
        # -------------------------------
        self.play(Write(eq_energy))
        self.wait(0.6)
        self.play(Write(eq_U))
        self.wait(0.6)
        self.play(Write(eq_prod))
        self.wait(1.0)

        self.play(
            Write(eq_cp),
            Write(eq_Q)
        )
        self.wait(0.8)

        self.play(Write(eq_T))
        self.wait(2.5)



class SimulacaoTanque(Scene):
    def construct(self):
        # --- Título ---
        title = Text("Simulação: Nível e Temperatura", font_size=36, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(0.6)

        A = np.pi * (1.5**2)           # m^2  (área constante)
        k = 0.13            # parâmetro de descarga (unidade compatível)
        q_in = 0.2        # m^3/s (vazão de entrada constante)
        rho = 1000.0       # kg/m^3
        cp = 4180.0        # J/(kg K)
        rho_j = 958.0     # densidade do condensado (assumir água)
        q_j = 0.015         # vazão de condensado (pode ser zero)
        lambda_j = 2.256e6    # calor latente (zero se q_j = 0)
        T_in = 301.15      # K (20 °C)

        # Condições iniciais
        L0 = 0.5           # m (nível inicial)
        T0 = 301.15           # K (inicial igual à entrada)

        # --- EDOs ---
        def model(t, y):
            L, T = y
            dLdt = (q_in - k * np.sqrt(L))/ A
            numerator = rho * q_in * cp * (T_in - T) + rho_j * q_j * lambda_j
            dTdt = numerator / (rho * A * L * cp)
            return [dLdt, dTdt]

        # Tempo de simulação
        t_span = (0, 500)  # segundos (mais longo para observar regimes)
        t_eval = np.linspace(t_span[0], t_span[1], 500)

        sol = solve_ivp(model, t_span, [L0, T0], t_eval=t_eval, vectorized=False)
        t_vals = sol.t
        L_vals = sol.y[0]
        T_vals = sol.y[1]

        # --- Determine temperatura min/max para paleta ---
        Tmin = float(np.min(T_vals))
        Tmax = float(np.max(T_vals))
        if Tmax - Tmin < 1e-6:
            Tmax = Tmin + 1.0

        # --- Layout: tanque (à esquerda) + gráficos (à direita) ---
        # Tank frame: a rectangle representando o tanque (vertical)
        tank_width = 2.0
        tank_height = 4.0
        tank_bottom = DOWN * 1.0
        tank_rect = Rectangle(width=tank_width, height=tank_height, color=WHITE).shift(LEFT * 3 + DOWN * 0.2)
        # water initial polygon (fill)
        def water_polygon_for_level(level):  # level in meters -> map to polygon
            # map L (meters) to y coordinate inside tank_rect
            # we consider max physical level for mapping: pick max observed or a cap
            Lcap = max( np.max(L_vals), 0.5 )  # cap to something reasonable
            # normalized height
            norm = float(level / Lcap)
            norm = max(0.0, min(1.0, norm))
            top_y = tank_rect.get_bottom()[1] + norm * tank_height
            left_x = tank_rect.get_left()[0] + 0.02
            right_x = tank_rect.get_right()[0] - 0.02
            bottom_y = tank_rect.get_bottom()[1] + 0.02
            return Polygon(
                np.array([left_x, bottom_y, 0]),
                np.array([left_x, top_y, 0]),
                np.array([right_x, top_y, 0]),
                np.array([right_x, bottom_y, 0]),
            )

        # Axis for L and T on the right
        axes_L = Axes(
            x_range=[t_vals[0], t_vals[-1], (t_vals[-1]-t_vals[0])/5],
            y_range=[0, np.max(L_vals)*1.2, np.max(L_vals)*0.2 if np.max(L_vals)>0 else 0.1],
            x_length=6,
            y_length=2.2,
        ).shift(RIGHT * 2.5 + UP * 1.1)
        axes_L_labels = axes_L.get_axis_labels(x_label="t (s)", y_label="L (m)")

        axes_T = Axes(
            x_range=[t_vals[0], t_vals[-1], (t_vals[-1]-t_vals[0])/5],
            y_range=[Tmin - 1.0, Tmax + 1.0, (Tmax - Tmin)/4],
            x_length=6,
            y_length=2.2,
        ).shift(RIGHT * 2.5 + DOWN * 1.3)
        axes_T_labels = axes_T.get_axis_labels(x_label="t (s)", y_label="T (K)")

        # Full (background) curves (light gray)
        # To avoid heavy LaTeX in plotting, create lists for axes.plot_line_graph
        # Manim expects lists; convert to python lists
        L_list = list(L_vals)
        T_list = list(T_vals)
        t_list = list(t_vals)

        curve_L_bg = axes_L.plot_line_graph(x_values=t_list, y_values=L_list, line_color=GRAY, stroke_width=2)
        curve_T_bg = axes_T.plot_line_graph(x_values=t_list, y_values=T_list, line_color=GRAY, stroke_width=2)

        # initial water polygon + dot on curves + traces
        initial_water = water_polygon_for_level(L_vals[0])
        initial_color = temp_to_color(T_vals[0], Tmin, Tmax)
        initial_water.set_fill(initial_color, opacity=0.8)
        initial_water.set_stroke(width=0)

        dot_L = Dot(color=YELLOW).move_to(axes_L.c2p(t_vals[0], L_vals[0]))
        dot_T = Dot(color=RED).move_to(axes_T.c2p(t_vals[0], T_vals[0]))

        trace_L = VMobject(stroke_width=4)
        trace_T = VMobject(stroke_width=4)
        trace_L.set_points_as_corners([dot_L.get_center(), dot_L.get_center()])  # at least one point
        trace_T.set_points_as_corners([dot_T.get_center(), dot_T.get_center()])

        # Labels
        label_L = Text("Nível L(t)", font_size=24).next_to(axes_L, UP)
        label_T = Text("Temperatura T(t)", font_size=24).next_to(axes_T, UP)

        # Add static elements
        self.play(Create(tank_rect))
        self.play(FadeIn(initial_water))
        self.play(Create(axes_L), Create(axes_T))
        self.play(Write(axes_L_labels), Write(axes_T_labels))
        self.play(Create(curve_L_bg), Create(curve_T_bg))
        self.play(FadeIn(label_L), FadeIn(label_T))
        self.wait(0.6)

        # Add traces and dots
        self.add(trace_L, trace_T, dot_L, dot_T)

        # --- Dynamic update function used with UpdateFromAlphaFunc ---
        def update_anim(mob, alpha):
            # alpha in [0,1] -> index
            idx = int(alpha * (len(t_vals) - 1))
            idx = max(1, min(len(t_vals)-1, idx))

            # Update water polygon
            new_water = water_polygon_for_level(float(L_vals[idx]))
            new_col = temp_to_color(float(T_vals[idx]), Tmin, Tmax)
            # replace mob[0] which is the current polygon: do by removing & adding or transform
            # We'll simply update mob (which will be the initial_water) via move & set_points_by_anchors:
            # For simplicity, replace the polygon by transforming
            mob_water = mob[0]
            self.play(Transform(mob_water, new_water, run_time=0.05), run_time=0.05)
            mob_water.set_fill(new_col, opacity=0.8)
            mob_water.set_stroke(width=0)

            # Update traces/dots (we update VMobjects directly)
            new_point_L = axes_L.c2p(t_vals[idx], L_vals[idx])
            new_point_T = axes_T.c2p(t_vals[idx], T_vals[idx])

            # update trace_L: append point if it's sufficiently far
            current_pts_L = trace_L.get_points()
            # ensure at least two anchors for set_points_as_corners:
            # build list of points from start up to idx
            ptsL = [axes_L.c2p(t_vals[j], L_vals[j]) for j in range(max(0, idx-1), idx+1)]
            trace_L.set_points_as_corners(ptsL)
            dot_L.move_to(new_point_L)

            ptsT = [axes_T.c2p(t_vals[j], T_vals[j]) for j in range(max(0, idx-1), idx+1)]
            trace_T.set_points_as_corners(ptsT)
            dot_T.move_to(new_point_T)

        # Because we need to animate multiple transformations per frame (water transform + traces),
        # we will step through the time using small sub-animations rather than a single UpdateFromAlphaFunc.
        # Create a lightweight loop of small animations to progress the scene.
        n_frames = 200
        for k in range(n_frames):
            alpha0 = k / n_frames
            alpha1 = (k+1) / n_frames
            idx1 = int(alpha1 * (len(t_vals)-1))
            idx1 = max(1, idx1)
            # compute new water and color
            new_water = water_polygon_for_level(float(L_vals[idx1]))
            new_col = temp_to_color(float(T_vals[idx1]), Tmin, Tmax)
            new_point_L = axes_L.c2p(t_vals[idx1], L_vals[idx1])
            new_point_T = axes_T.c2p(t_vals[idx1], T_vals[idx1])

            # update water by Transform
            self.play(Transform(initial_water, new_water), run_time=0.08, rate_func=linear)
            initial_water.set_fill(new_col, opacity=0.8)
            initial_water.set_stroke(width=0)

            # update traces (set sublist of points)
            j0 = max(0, idx1-40)  # keep a sliding window for performance
            ptsL = [axes_L.c2p(t_vals[j], L_vals[j]) for j in range(j0, idx1+1)]
            ptsT = [axes_T.c2p(t_vals[j], T_vals[j]) for j in range(j0, idx1+1)]
            if len(ptsL) >= 2:
                trace_L.set_points_as_corners(ptsL)
            else:
                trace_L.set_points_as_corners([new_point_L, new_point_L])
            if len(ptsT) >= 2:
                trace_T.set_points_as_corners(ptsT)
            else:
                trace_T.set_points_as_corners([new_point_T, new_point_T])

            dot_L.move_to(new_point_L)
            dot_T.move_to(new_point_T)

        # finalize
        self.wait(0.8)
        self.play(FadeOut(initial_water), FadeOut(tank_rect),
                  FadeOut(axes_L), FadeOut(axes_T),
                  FadeOut(axes_L_labels), FadeOut(axes_T_labels),
                  FadeOut(curve_L_bg), FadeOut(curve_T_bg),
                  FadeOut(trace_L), FadeOut(trace_T),
                  FadeOut(dot_L), FadeOut(dot_T),
                  FadeOut(label_L), FadeOut(label_T), FadeOut(title))
        end = Text("Simulação concluída", font_size=34, color=GREEN)
        self.play(Write(end))
        self.wait(1.2)

# ===============================
# 5. Classificação
# ===============================
class ClassificacaoTanque(Scene):
    def construct(self):
        title = Text("Classificação do Modelo", font_size=38, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(0.6)

        props = [
            "Dinâmico × Estático",
            "Linear × Não linear",
            "SISO × SIMO × MIMO",
            "Contínuo × Discreto",
            "Invariante × Variante no tempo",
            "Concentrado × Distribuído",
            "Determinístico × Estocástico",
            "Forçado × Homogêneo"
        ]
        vals = [
            "Dinâmico",
            "Não linear",
            "MIMO",
            "Contínuo",
            "Invariante no tempo",
            "Concentrado",
            "Determinístico",
            "Forçado"
        ]
        colors = [GREEN, ORANGE, PURPLE, TEAL, GOLD, BLUE_B, RED, YELLOW]

        left_col = VGroup(*[Text(p, font_size=24) for p in props]).arrange(DOWN, aligned_edge=LEFT, buff=0.45).shift(LEFT*2)
        self.play(LaggedStart(*[Write(m) for m in left_col], lag_ratio=0.12))
        self.wait(0.6)

        for i in range(len(props)):
            arrow = Arrow(start=left_col[i].get_right(), end=left_col[i].get_right()+RIGHT*1.5, color=colors[i])
            val = Text(vals[i], font_size=24, color=colors[i]).next_to(arrow, RIGHT, buff=0.2)
            self.play(GrowArrow(arrow), FadeIn(val, shift=RIGHT))
            self.play(left_col[i].animate.set_color(colors[i]))
            self.wait(0.25)

        self.wait(1.4)
        self.play(FadeOut(*self.mobjects))
        self.wait(0.3)
