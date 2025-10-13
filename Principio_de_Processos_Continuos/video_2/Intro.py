from manim import *
import numpy as np

class PressurizedVesselAnimation(Scene):
    def construct(self):
        # --- Texto de introdução ---
        intro_text = Text("Vaso pressurizado de gás", font_size=36).to_edge(UP)
        self.play(Write(intro_text))
        self.wait(1)

        # --- Desenho do vaso com entrada e saída ---
        vessel = Rectangle(width=3, height=4, color=BLUE).move_to(ORIGIN)
        inlet_arrow = Arrow(start=LEFT*5, end=LEFT*1.5, color=GREEN)
        outlet_arrow = Arrow(start=RIGHT*1.5, end=RIGHT*5, color=RED)
        label_in = Text("F1(t)", font_size=24).next_to(inlet_arrow, UP)
        label_out = Text("F2(t)", font_size=24).next_to(outlet_arrow, UP)

        self.play(Create(vessel))
        self.play(GrowArrow(inlet_arrow), Write(label_in))
        self.play(GrowArrow(outlet_arrow), Write(label_out))
        self.wait(1)

        # --- Passo 1: Balanço de massa ---
        eq_mass_balance = MathTex(r"\frac{dm(t)}{dt} = F_1(t) - F_2(t)").to_edge(DOWN)
        self.play(Write(eq_mass_balance))
        self.wait(2)
        self.play(FadeOut(eq_mass_balance))
        self.wait(2)
        # --- Passo 2: Mover e reduzir o tanque ---
        self.play(
            vessel.animate.scale(0.5).move_to(UL*3),
            inlet_arrow.animate.scale(0.5).move_to(UL*3 + LEFT*0.5),
            outlet_arrow.animate.scale(0.5).move_to(UL*3 + RIGHT*0.5),
            label_in.animate.scale(0.7).move_to(UL*3 + LEFT*0.5 + UP*0.5),
            label_out.animate.scale(0.7).move_to(UL*3 + RIGHT*0.5 + UP*0.5),
            run_time=2
        )
        self.wait(0.5)

        # --- Passo 3: Equações intermediárias, uma por vez ---
        eq_flows = MathTex(
            r"F_1(t) = k_1 (P_1 - P(t)),\quad F_2(t) = k_2 (P(t) - P_2)"
        ).to_edge(DOWN)
        eq_mass_ideal = MathTex(
            r"m(t) = \frac{P(t) V M_M}{R T}"
        ).to_edge(DOWN)
        eq_mass_derivative = MathTex(
            r"\frac{dm(t)}{dt} = \frac{V M_M}{R T} \frac{dP(t)}{dt}"
        ).to_edge(DOWN)

        # Mostrar e sumir as equações intermediárias
        self.play(Write(eq_flows))
        self.wait(1.5)
        self.play(FadeOut(eq_flows), Write(eq_mass_ideal))
        self.wait(1.5)
        self.play(FadeOut(eq_mass_ideal), Write(eq_mass_derivative))
        self.wait(1.5)
        self.play(FadeOut(eq_mass_derivative))
        self.wait(0.5)

        # --- Passo 4: Voltar o tanque para o centro ---
        self.play(
            vessel.animate.scale(2).move_to(ORIGIN),
            inlet_arrow.animate.scale(2).move_to(LEFT*1.5),
            outlet_arrow.animate.scale(2).move_to(RIGHT*1.5),
            label_in.animate.scale(1.43).move_to(LEFT*1.5 + UP*0.5),
            label_out.animate.scale(1.43).move_to(RIGHT*1.5 + UP*0.5),
            run_time=2
        )
        self.wait(0.5)

        # --- Passo 5: Equação final da pressão ---
        eq_final_pressure = MathTex(
            r"\frac{dP}{dt} = \frac{R T}{V M_M} \Big( k_1 (P_1 - P) - k_2 (P - P_2) \Big)"
        ).to_edge(DOWN)
        self.play(Write(eq_final_pressure))
        self.wait(2)

        # --- Passo 6: Gráfico de P(t) surgindo ---
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_axis_config={"include_tip": True},
            y_axis_config={"include_tip": True}
        ).to_edge(DOWN)
        label_axes = axes.get_axis_labels(x_label="t", y_label="P(t)")
        self.play(Create(axes), Write(label_axes))

        # Exemplo qualitativo de P(t)
        def pressure(t):
            return 5 + 2*np.exp(-0.2*t)*np.sin(t)

        graph = axes.plot(pressure, color=YELLOW)
        self.play(Create(graph), run_time=4)
        self.wait(2)

        # --- Animação das setas de fluxo contínuo ---
        for _ in range(3):
            self.play(Indicate(inlet_arrow), Indicate(outlet_arrow), run_time=1.5)
