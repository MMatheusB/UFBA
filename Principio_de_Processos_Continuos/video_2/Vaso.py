from manim import *
import numpy as np
from scipy.integrate import solve_ivp

class PressurizedVesselAnimation(Scene):
    def construct(self):
        # --- Parâmetros físicos do sistema ---
        R = 8.314      # J/mol·K
        T = 300         # K
        V = 1.0         # m³
        MM = 0.0289     # kg/mol (ar aproximado)
        k1 = 0.05
        k2 = 0.04
        P1 = 10.0       # pressão de entrada (bar)
        P2 = 2.0        # pressão de saída (bar)
        P0 = 5.0        # pressão inicial (bar)

        # --- EDO: dP/dt = (RT/(V*MM)) * (k1*(P1 - P) - k2*(P - P2)) ---
        def dPdt(t, P):
            return (R*T/(V*MM)) * (k1*(P1 - P) - k2*(P - P2))

        # --- Solução numérica usando solve_ivp ---
        t_span = (0, 0.1)
        t_eval = np.linspace(*t_span, 200)
        sol = solve_ivp(dPdt, t_span, [P0], t_eval=t_eval)
        t_vals = sol.t
        P_vals = sol.y[0]

        # --- Texto de introdução ---
        intro_text = Text("Vaso pressurizado de gás", font_size=36).to_edge(UP)
        self.play(Write(intro_text))
        self.wait(3)

        # --- Desenho do vaso com entrada e saída ---
        vessel = Rectangle(width=3, height=4, color=BLUE).move_to(ORIGIN)
        inlet_arrow = Arrow(start=LEFT*3.5, end=LEFT*1.5, color=GREEN)
        outlet_arrow = Arrow(start=RIGHT*1.5, end=RIGHT*3.5, color=RED)
        label_in = Text("F₁(t)", font_size=24).next_to(inlet_arrow, UP)
        label_out = Text("F₂(t)", font_size=24).next_to(outlet_arrow, UP)

        self.play(Create(vessel))
        self.play(GrowArrow(inlet_arrow), Write(label_in))
        self.play(GrowArrow(outlet_arrow), Write(label_out))
        self.wait(3)

        # --- Passo 1: Balanço de massa ---
        eq_mass_balance = MathTex(r"\frac{dm(t)}{dt} = F_1(t) - F_2(t)").next_to(vessel, DOWN)
        self.play(Write(eq_mass_balance))
        self.wait(3)
        self.play(FadeOut(eq_mass_balance))
        self.wait(1)

        # --- Passo 2: Equações intermediárias ---
        eq_flows = MathTex(r"F_1(t) = k_1 (P_1 - P(t)),\quad F_2(t) = k_2 (P(t) - P_2)").next_to(vessel, DOWN)
        eq_mass_ideal = MathTex(r"m(t) = \frac{P(t)\,V\,M_M}{R\,T}").next_to(vessel, DOWN)
        eq_mass_derivative = MathTex(r"\frac{dm(t)}{dt} = \frac{V\,M_M}{R\,T}\frac{dP(t)}{dt}").next_to(vessel, DOWN)

        self.play(Write(eq_flows))
        self.wait(3)
        self.play(FadeOut(eq_flows), Write(eq_mass_ideal))
        self.wait(3)
        self.play(FadeOut(eq_mass_ideal), Write(eq_mass_derivative))
        self.wait(3)
        self.play(FadeOut(eq_mass_derivative))
        self.wait(2)

        # --- Passo 3: Equação final da pressão ---
        eq_final_pressure = MathTex(
            r"\frac{dP}{dt} = \frac{R\,T}{V\,M_M} \Big( k_1 (P_1 - P) - k_2 (P - P_2) \Big)"
        ).next_to(vessel, DOWN)
        self.play(Write(eq_final_pressure))
        self.wait(4)
        self.play(FadeOut(eq_final_pressure))
        self.wait(0.5)

        # --- Passo 4: Mover tanque para esquerda e mostrar gráfico à direita ---
        self.play(
            vessel.animate.move_to(LEFT*3.5),
            inlet_arrow.animate.shift(LEFT*3.5),
            outlet_arrow.animate.shift(LEFT*3.5),
            label_in.animate.shift(LEFT*3.5),
            label_out.animate.shift(LEFT*3.5),
            run_time=3
        )

        # Criação do gráfico à direita
        axes = Axes(
            x_range=[0, 0.1, 0.02],  # tempo de 0 a 0.1
            y_range=[min(P_vals)-0.5, max(P_vals)+0.5, 1],
            axis_config={"include_tip": True}
        ).scale(0.5).move_to(RIGHT*3)

        labels = axes.get_axis_labels("t", "P(t)")
        self.play(Create(axes), Write(labels))

        # --- Curva simulada (traçada progressivamente) ---
        pressure_curve = axes.plot(
            lambda t: np.interp(t, t_vals, P_vals),
            x_range=[t_vals[0], t_vals[-1]],
            color=YELLOW
        )

        # Animação: curva sendo desenhada no tempo
        self.play(Create(pressure_curve), run_time=6, rate_func=linear)
        self.wait(3)

        # --- Passo 5: Animação das setas de fluxo contínuo ---
        for _ in range(3):
            self.play(Indicate(inlet_arrow), Indicate(outlet_arrow), run_time=1.5)


        # --- Passo 5: Animação das setas de fluxo contínuo ---
        for _ in range(3):
            self.play(Indicate(inlet_arrow), Indicate(outlet_arrow), run_time=1.5)
