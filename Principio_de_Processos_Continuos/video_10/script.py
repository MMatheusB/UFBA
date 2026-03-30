from manim import *
import numpy as np


class TanqueMisturadorCSTR(Scene):

    def construct(self):

        # ======================================================
        # TÍTULO GERAL
        # ======================================================
        title = Text("Tanque com Misturador (CSTR)", font_size=44, color=BLUE)
        subtitle = Text("Modelagem de Sistema Químico", font_size=28, color=YELLOW)
        VGroup(title, subtitle).arrange(DOWN, buff=0.4).to_edge(UP)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ======================================================
        # PREMISSAS
        # ======================================================
        self.section_transition("Premissas do Modelo")

        prem_title = Text("Premissas do Modelo", font_size=36, color=GREEN).to_edge(UP)
        self.play(FadeIn(prem_title))

        premissas = [
            Text("• Mistura perfeita (CSTR ideal)", font_size=26),
            Text("• Volume do tanque constante", font_size=26),
            Text("• Vazão de entrada igual à vazão de saída", font_size=26),
            Text("• Fluido incompressível", font_size=26),
            Text("• Processo isotérmico", font_size=26),
            Text("• Reação química de primeira ordem", font_size=26, color=RED),
        ]

        prem_group = VGroup(*premissas).arrange(DOWN, buff=0.4)
        prem_group.next_to(prem_title, DOWN, buff=1)

        for item in premissas:
            self.play(FadeIn(item))
            self.wait(0.8)

        self.wait(2)
        self.play(FadeOut(prem_title), FadeOut(prem_group))

        # ======================================================
        # VARIÁVEIS
        # ======================================================
        self.section_transition("Variáveis do Sistema")

        vars_title = Text("Variáveis do Sistema", font_size=36, color=BLUE).to_edge(UP)
        self.play(FadeIn(vars_title))

        vars_group = VGroup(
            Text("Estado:", font_size=26, color=GREEN),
            MathTex(r"C(t)"),
            Text("Entradas:", font_size=26, color=ORANGE),
            MathTex(r"C_{in}(t), \; Q"),
            Text("Parâmetros:", font_size=26, color=PURPLE),
            MathTex(r"V, \; k"),
        ).arrange(DOWN, buff=0.45, aligned_edge=LEFT)

        vars_group.next_to(vars_title, DOWN, buff=1)

        self.play(LaggedStart(*[FadeIn(v) for v in vars_group], lag_ratio=0.3))
        self.wait(3)
        self.play(FadeOut(vars_title), FadeOut(vars_group))

        # ======================================================
        # LEIS FUNDAMENTAIS
        # ======================================================
        self.section_transition("Leis Fundamentais")

        laws_title = Text("Leis Fundamentais", font_size=36, color=BLUE).to_edge(UP)
        self.play(FadeIn(laws_title))

        laws = VGroup(
            Text("Conservação de massa do componente químico", font_size=26, color=GREEN),
            MathTex(r"\text{Acúmulo} = \text{Entrada} - \text{Saída} + \text{Reação}"),
            Text("Lei cinética da reação (1ª ordem)", font_size=26, color=RED),
            MathTex(r"r(C) = k\,C"),
        ).arrange(DOWN, buff=0.5)

        laws.next_to(laws_title, DOWN, buff=1)

        self.play(LaggedStart(*[FadeIn(l) for l in laws], lag_ratio=0.4))
        self.wait(3)
        self.play(FadeOut(laws_title), FadeOut(laws))

        # ======================================================
        # DEDUÇÃO PASSO A PASSO
        # ======================================================
        self.section_transition("Dedução do Modelo")

        deriv_title = Text("Dedução do Modelo Dinâmico", font_size=36, color=BLUE).to_edge(UP)
        self.play(FadeIn(deriv_title))

        eq1 = MathTex(
            r"\frac{d(VC)}{dt} =",
            r"\underbrace{Q\,C_{in}}_{\text{entrada}}",
            r"- \underbrace{Q\,C}_{\text{saída}}",
            r"- \underbrace{V\,k\,C}_{\text{reação}}"
        ).scale(1.0)

        eq1.next_to(deriv_title, DOWN, buff=1)

        self.play(Write(eq1))
        self.wait(3)

        eq2 = MathTex(
            r"V\,\frac{dC}{dt} = Q(C_{in} - C) - V k C"
        ).scale(1.1)

        self.play(ReplacementTransform(eq1, eq2))
        self.wait(3)

        eq3 = MathTex(
            r"\frac{dC}{dt} = \frac{Q}{V}(C_{in} - C) - k C"
        ).scale(1.2)

        self.play(ReplacementTransform(eq2, eq3))
        self.wait(3)

        box = SurroundingRectangle(eq3, color=YELLOW, buff=0.3)
        model_title = Text("Modelo Dinâmico Final do CSTR", font_size=32, color=BLUE)
        model_title.next_to(box, UP)

        self.play(Create(box), FadeIn(model_title))
        self.wait(3)

        self.play(FadeOut(deriv_title), FadeOut(eq3), FadeOut(box), FadeOut(model_title))

        # ======================================================
        # SIMULAÇÃO
        # ======================================================
        self.section_transition("Resposta Dinâmica")

        sim_title = Text("Resposta ao Degrau em $C_{in}$", font_size=36, color=BLUE).to_edge(UP)
        self.play(FadeIn(sim_title))

        time, C = self.simulate_cstr()

        axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, max(C) * 1.2, 0.5],
            tips=False,
            axis_config={"include_numbers": True}
        ).scale(0.7)

        axes.next_to(sim_title, DOWN, buff=1)

        curve = axes.plot_line_graph(time, C, add_vertex_dots=False)

        self.play(Create(axes))
        self.play(Create(curve), run_time=3)
        self.wait(4)

    # ======================================================
    # SIMULAÇÃO ANALÍTICA
    # ======================================================
    def simulate_cstr(self):
        V, Q, k = 1.0, 0.1, 0.05
        Cin = 2.0
        t = np.linspace(0, 100, 500)
        C = Cin * (Q/V) / (Q/V + k) * (1 - np.exp(-(Q/V + k)*t))
        return t, C

    # ======================================================
    # TRANSIÇÃO DE SEÇÕES
    # ======================================================
    def section_transition(self, text):
        title = Text(text, font_size=44, color=BLUE)
        box = SurroundingRectangle(title, color=BLUE, buff=0.5)
        group = VGroup(title, box)
        self.play(FadeIn(group))
        self.wait(1.5)
        self.play(FadeOut(group))
