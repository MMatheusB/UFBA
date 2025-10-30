from manim import *
import numpy as np
from scipy.integrate import solve_ivp

class RCCircuitPremises(Scene):
    def construct(self):
        # --- Título ---
        title = Text("Premissas do Modelo RC", font_size=42, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(1.5)

        # --- Lista de premissas ---
        premissas = BulletedList(
            "Os fios e componentes são ideais (sem perdas de energia)",
            "A fonte de tensão é constante no tempo",
            "O capacitor está inicialmente descarregado",
            "A resistência e a capacitância não variam com o tempo",
            "As grandezas variam continuamente, sem saltos instantâneos",
            font_size=30
        ).next_to(title, DOWN, buff=0.8)

        # Exibir uma a uma
        for item in premissas:
            self.play(FadeIn(item, shift=DOWN), run_time=1.2)
            self.wait(1.3)

        self.wait(2)

        # --- Encerramento / transição ---
        outro = Text(
            "Com essas premissas, conseguimos descrever a carga do capacitor ao longo do tempo.",
            font_size=18
        ).next_to(premissas, DOWN, buff=0.8)

        self.play(Write(outro))
        self.wait(3)

        # Fade para limpar a tela antes da próxima animação
        self.play(FadeOut(title), FadeOut(premissas), FadeOut(outro))
        self.wait(1)

        
class RCCircuitGraphOnly1(Scene):
    def construct(self):
        # --- Parâmetros do circuito ---
        R = 1000      # ohms
        C = 1e-6      # farads
        E = 5         # tensão da fonte [V]
        V0 = 0        # tensão inicial no capacitor [V]

        # --- EDO: dVc/dt = 1/(R*C) * (E - Vc) ---
        def dVcdt(t, Vc):
            return (1/(R*C)) * (E - Vc)

        # --- Solução numérica ---
        t_span = (0, 0.01)
        t_eval = np.linspace(*t_span, 300)
        sol = solve_ivp(dVcdt, t_span, [V0], t_eval=t_eval)
        t_vals = sol.t
        Vc_vals = sol.y[0]

        # --- Título ---
        intro_text = Text("Circuito Série RC", font_size=36).to_edge(UP)
        self.play(Write(intro_text))
        self.wait(3)

        # --- Passo 1: Lei de Kirchhoff ---
        eq_kvl = MathTex(r"\varepsilon - V_R - V_C = 0").to_edge(DOWN, buff=3)
        self.play(Write(eq_kvl))
        self.wait(3)
        self.play(FadeOut(eq_kvl))

        # --- Passo 2: Relações intermediárias ---
        eq_resistor = MathTex(r"V_R = R \cdot i").to_edge(DOWN, buff=3)
        eq_current = MathTex(r"i = C \frac{dV_C}{dt}").to_edge(DOWN, buff=3)
        
        for eq in [eq_resistor, eq_current]:
            self.play(Write(eq))
            self.wait(3)
            self.play(FadeOut(eq))
            self.wait(2)
        # --- Passo 3: Equação final ---
        eq_final = MathTex(r"\frac{dV_C}{dt} = \frac{1}{RC} (\varepsilon - V_C)").to_edge(DOWN, buff=3)
        self.play(Write(eq_final))
        self.wait(3)
        self.play(FadeOut(eq_final))
        self.wait(3)
        # --- Passo 4: Gráfico ---
        axes = Axes(
            x_range=[0, t_span[1], t_span[1]/5],
            y_range=[0, 5.5, 1],  
            axis_config={"include_tip": True}
        ).scale(0.6).move_to(RIGHT * 3)
        labels = axes.get_axis_labels("t (s)", "V_C(t) (V)")
        self.play(Create(axes), Write(labels))

        # Curva simulada
        vc_curve = axes.plot(
            lambda t: np.interp(t, t_vals, Vc_vals),
            x_range=[t_vals[0], t_vals[-1]],
            color=YELLOW
        )
        self.play(Create(vc_curve), run_time=4, rate_func=linear)
        self.wait(3)
