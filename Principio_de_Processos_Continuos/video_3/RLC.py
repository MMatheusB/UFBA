from manim import *
import numpy as np
from scipy.integrate import solve_ivp

# -------------------------------
# 1. Premissas do Modelo RLC
# -------------------------------
class RLCCircuitPremises(Scene):
    def construct(self):
        # --- Título ---
        title = Text("Premissas do Modelo RLC", font_size=42, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(1.5)

        # --- Lista de premissas ---
        premissas = BulletedList(
            "Os fios e componentes são ideais (sem perdas adicionais)",
            "A fonte de tensão é conhecida e contínua no tempo",
            "O capacitor e o indutor são lineares e de parâmetros constantes",
            "As grandezas elétricas variam continuamente no tempo",
            "O circuito é de parâmetros concentrados (sem efeitos distribuídos)",
            font_size=30
        ).next_to(title, DOWN, buff=0.8)

        for item in premissas:
            self.play(FadeIn(item, shift=DOWN), run_time=1.2)
            self.wait(1.3)

        self.wait(2)

        outro = Text(
            "Com essas premissas, podemos modelar a dinâmica da tensão no capacitor ao longo do tempo.",
            font_size=18
        ).next_to(premissas, DOWN, buff=0.8)

        self.play(Write(outro))
        self.wait(3)

        self.play(FadeOut(title), FadeOut(premissas), FadeOut(outro))
        self.wait(1)


# -------------------------------
# 2. Derivação e Gráfico
# -------------------------------
class RLCCircuitGraph(Scene):
    def construct(self):
        # --- Parâmetros ---
        R = 10      # ohms
        L = 2      # henry
        C = 5000e-6     # farads
        E = 5        # tensão da fonte [V]
        Vc0 = 0      # tensão inicial no capacitor [V]
        dVc0 = 0     # derivada inicial (corrente inicial no capacitor)

        # --- Sistema de 1ª ordem ---
        # x1 = Vc
        # x2 = dVc/dt
        def rlc_system(t, y):
            Vc, dVc = y
            d2Vc = (1/(L*C)) * E - (R/L)*dVc - (1/(L*C))*Vc
            return [dVc, d2Vc]

        # --- Solução numérica ---
        t_span = (0, 3)
        t_eval = np.linspace(*t_span, 600)
        sol = solve_ivp(rlc_system, t_span, [Vc0, dVc0], t_eval=t_eval)
        t_vals = sol.t
        Vc_vals = sol.y[0]

        # --- Introdução ---
        intro_text = Text("Circuito Série RLC", font_size=36).to_edge(UP)
        self.play(Write(intro_text))
        self.wait(2)

        # --- Passo 1: Lei das tensões ---
        eq_kvl = MathTex(r"\varepsilon - V_R - V_L - V_C = 0").to_edge(DOWN, buff=3)
        self.play(Write(eq_kvl))
        self.wait(3)
        self.play(FadeOut(eq_kvl))

        # --- Passo 2: Relações constitutivas ---
        eq_relations = [
            MathTex(r"V_R = R \, i"),
            MathTex(r"V_L = L \frac{di}{dt}"),
            MathTex(r"i = C \frac{dV_C}{dt}")
        ]
        for eq in eq_relations:
            eq.to_edge(DOWN, buff=3)
            self.play(Write(eq))
            self.wait(2)
            self.play(FadeOut(eq))
            self.wait(0.5)

        # --- Passo 3: Equação diferencial ---
        eq_final = MathTex(
            r"\frac{d^2 V_C}{dt^2} = \frac{1}{LC} \varepsilon "
            r"- \frac{R}{L} \frac{dV_C}{dt} "
            r"- \frac{1}{LC} V_C"
        ).to_edge(DOWN, buff=3)
        self.play(Write(eq_final))
        self.wait(3)
        self.play(FadeOut(eq_final))
        self.wait(1.5)

        # --- Passo 4: Gráfico ---
        axes = Axes(
            x_range=[0, t_span[1], t_span[1]/5],
            y_range=[-1, 10, 1],
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

        outro = Text("Tensão no capacitor variando com o tempo", font_size=24).next_to(axes, DOWN)
        self.play(Write(outro))
        self.wait(3)

class RCCircuitClassification(Scene):
    def construct(self):
        # --- Título ---
        title = Text("Classificação do Modelo RLC", font_size=44, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # --- Lista de propriedades e valores ---
        props = [
            "Linear × Não linear",
            "SISO × SIMO × MISO × MIMO",
            "Tempo contínuo × Tempo discreto",
            "Invariante × Variante no tempo",
            "Parâmetros concentrados × Distribuídos",
            "Determinístico × Estocástico",
            "Forçado × Homogêneo"
        ]

        vals = [
            "Linear",
            "SISO",
            "Contínuo no tempo",
            "Invariante no tempo",
            "Parâmetros concentrados",
            "Determinístico",
            "Forçado"
        ]

        colors = [GREEN, TEAL, ORANGE, PURPLE, BLUE_B, RED, GOLD]

        # --- Coluna esquerda (propriedades) ---
        left_col = VGroup(*[
            Text(p, font_size=26, color=WHITE).align_to(LEFT, LEFT)
            for p in props
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(LEFT * 3)

        self.play(LaggedStart(*[FadeIn(m, shift=LEFT) for m in left_col], lag_ratio=0.15))
        self.wait(1)

        # --- Anima cada seta + resposta ---
        right_group = VGroup()

        for i in range(len(props)):
            # Cria seta e texto da resposta
            val_text = Text(vals[i], font_size=26, color=colors[i])
            arrow = Arrow(
                start=left_col[i].get_right(),
                end=left_col[i].get_right() + RIGHT * 1.5,
                buff=0.1,
                color=colors[i],
                stroke_width=3
            )
            val_text.next_to(arrow, RIGHT, buff=0.2)

            # Mostra seta + texto juntos
            self.play(GrowArrow(arrow), FadeIn(val_text, shift=RIGHT), run_time=0.8)
            self.play(
                left_col[i].animate.set_color(colors[i]),
                val_text.animate.scale(1.1),
                run_time=0.6
            )
            self.wait(0.5)
            self.play(val_text.animate.scale(1/1.1))
            right_group.add(arrow, val_text)

        self.wait(1.5)
