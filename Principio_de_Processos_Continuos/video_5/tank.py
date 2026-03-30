from manim import *
import numpy as np
from scipy.integrate import solve_ivp

# =================================================
# 0. PARÂMETROS GLOBAIS (ajuste se quiser)
# =================================================
PI = np.pi

# =================================================
# 1. INTRODUÇÃO (curta, sobre o tanque)
# =================================================
class IntroTanque(Scene):
    def construct(self):
        title = Text("Tanque Cônico", font_size=42, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(0.6)

        bullets = BulletedList(
    r"Vaso em forma de cone -- volume varia com \(h^{3}\)",
    r"Entrada bombeada (\(q_{\text{in}}\)) e saída dependente da altura",
    r"Queremos a dinâmica do nível \(h(t)\)",).next_to(title, DOWN, buff=0.8)


        for item in bullets:
            self.play(FadeIn(item, shift=DOWN), run_time=0.8)
            self.wait(0.4)

        outro = Text("Vamos às premissas do modelo.", font_size=24).next_to(bullets, DOWN, buff=0.8)
        self.play(Write(outro))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(bullets), FadeOut(outro))
        self.wait(0.5)

# =================================================
# 2. PREMISSAS (curtas, simples)
# =================================================
class PremissasTanque(Scene):
    def construct(self):
        title = Text("Premissas do Modelo", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(0.6)

        itens = [
            "Fluido incompressível e densidade constante",
            "Sem vazamentos, evaporação ou reações",
            "Saída por gravidade dependente do nível (Torricelli)",
            "Entrada q_in conhecida e mensurável"
        ]

        lista = VGroup()
        y_cursor = title.get_bottom() + DOWN * 0.6
        for txt in itens:
            line = Text("•  " + txt, font_size=28).next_to(ORIGIN, UP)  # position fixado em seguida
            if len(lista) == 0:
                line.move_to(y_cursor)
            else:
                line.next_to(lista[-1], DOWN, aligned_edge=LEFT, buff=0.5)
            lista.add(line)

        # aparece um por um
        for item in lista:
            self.play(FadeIn(item, shift=RIGHT), run_time=0.7)
            self.wait(0.6)

        self.wait(1.0)
        self.play(FadeOut(lista), FadeOut(title))
        self.wait(0.3)

# =================================================
# 3. DERIVAÇÃO (passo-a-passo: balanço, volume, substituição)
# =================================================
class DerivacaoTanque(Scene):
    def construct(self):
        title = Text("Derivação da EDO do Nível", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(0.6)

        # 1: balanço volumétrico
        step1 = MathTex(r"\frac{dV}{dt} = q_{in}(t) - q_{out}(t)").next_to(title, DOWN, buff=0.8)
        self.play(Write(step1))
        self.wait(1.2)
        self.play(FadeOut(step1))

        # 2: volume do cone em função de h
        step2a = MathTex(r"V(h) = \frac{\pi R^{2}}{3 H^{2}} \, h^{3}").next_to(title, DOWN, buff=0.8)
        self.play(Write(step2a))
        self.wait(1.2)
        self.play(FadeOut(step2a))

        # 3: derivada de V(h)
        step3 = MathTex(r"\frac{dV}{dt} = \frac{\pi R^{2}}{H^{2}} \, h^{2}(t) \, \frac{dh}{dt}").next_to(title, DOWN, buff=0.8)
        self.play(Write(step3))
        self.wait(1.4)
        self.play(FadeOut(step3))

        # 4: trocar no balanço e isolar dh/dt
        step4 = MathTex(
            r"\pi \frac{R^{2}}{H^{2}} h^{2}\, \frac{dh}{dt} = q_{in}(t) - q_{out}(t)"
        ).next_to(title, DOWN, buff=0.8)
        self.play(Write(step4))
        self.wait(1.2)
        self.play(FadeOut(step4))

        # 5: modelo da saída (Torricelli-like) e equação final
        step5a = MathTex(r"q_{out}(t) = k \, h^{\alpha}").next_to(title, DOWN, buff=0.8)
        # para seu caso: α = 3/2 ou simplificação apresentada; aqui deixamos genérico e depois a versão usada
        self.play(Write(step5a))
        self.wait(1.0)
        self.play(FadeOut(step5a))

        # versão final conforme enunciado (usando q_out = k h^3 para compatibilidade com seu enunciado)
        final = MathTex(
            r"\frac{dh}{dt} = \frac{H^{2}}{\pi R^{2}}\left(q_{in}(t)\,h^{2} - k\,h^{3}\right)"
        ).scale(0.95).next_to(title, DOWN, buff=0.8)
        self.play(Write(final))
        self.wait(2.0)

        outro = Text("Esta é a EDO não linear que descreve o nível do tanque.", font_size=22).next_to(final, DOWN, buff=0.8)
        self.play(Write(outro))
        self.wait(2.0)

        self.play(FadeOut(title), FadeOut(final), FadeOut(outro))
        self.wait(0.3)

# =================================================
# 4. SIMULAÇÃO NUMÉRICA E ANIMAÇÃO DO GRÁFICO (real)
# =================================================
class SimulacaoTanque(Scene):
    def construct(self):
        title = Text("Simulação do Tanque Cônico", font_size=38, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(0.6)

        # --- Parâmetros (ajuste se quiser) ---
        H = 3.5       # altura máxima [m]
        R = 1.2      # raio no topo [m]
        k = 0.8       # parâmetro de descarga (usar unidades compatíveis)
        q_in = 1   # vazão de entrada volumétrica (m^3/s) - constante neste exemplo

        # EDO: dh/dt = (H^2/(pi R^2)) * ( q_in * h^2 - k * h^3 )
        coef = (H**2) / (PI * R**2)

        def ode(t, h):
            # h é vetor com shape (1,) ou escalar
            hh = h[0] if hasattr(h, "__len__") else h
            dh = coef * (q_in * hh**2 - k * hh**3)
            return [dh]

        # kondisi e solução numérica
        h0 = 0.05
        t_span = (0.0, 120.0)   # 120 s para ver convergência
        t_eval = np.linspace(t_span[0], t_span[1], 800)
        sol = solve_ivp(ode, t_span, [h0], t_eval=t_eval, rtol=1e-6)

        t = sol.t
        h = sol.y[0]

        # --- Eixos ---
        axes = Axes(
            x_range=[t_span[0], t_span[1], 30],
            y_range=[0, max(1.0, np.max(h)*1.2), max(0.1, np.max(h)/5+0.01)],
            x_length=10,
            y_length=4,
            axis_config={"include_tip": True}
        ).shift(DOWN * 0.2)
        labels = axes.get_axis_labels("t (s)", "h(t) (m)")

        self.play(Create(axes), Write(labels))
        self.wait(0.5)

        # Desenhar curva completa em cinza leve (opcional)
        full_curve = axes.plot_line_graph(
            x_values=t,
            y_values=h,
            line_color=GRAY,
            stroke_width=2
        )
        self.play(Create(full_curve), run_time=1.0)
        self.wait(0.4)

        # --- Trace animado ---
        trace = VMobject(stroke_width=4)
        trace.set_points_as_corners([axes.c2p(t[0], h[0])])
        dot = Dot(color=YELLOW).move_to(axes.c2p(t[0], h[0]))
        self.add(trace, dot)

        def update_trace(mob, alpha):
            # calcula índice e atualiza pontos (garantir ao menos 2 pontos)
            idx = int(alpha * (len(t) - 1))
            idx = max(idx, 1)
            pts = [axes.c2p(t[j], h[j]) for j in range(idx+1)]
            mob.set_points_as_corners(pts)
            dot.move_to(pts[-1])

        self.play(UpdateFromAlphaFunc(trace, update_trace), run_time=10, rate_func=linear)
        self.wait(0.5)

        # Mostrar Estado de Regime (valor final aproximado)
        h_ss = h[-1]
        txt = Text(f"Altura final aproximada: {h_ss:.3f} m", font_size=24).next_to(axes, DOWN, buff=0.6)
        self.play(Write(txt))
        self.wait(1.5)

        self.play(FadeOut(trace), FadeOut(dot), FadeOut(full_curve), FadeOut(axes), FadeOut(labels), FadeOut(txt), FadeOut(title))
        self.wait(0.3)

# =================================================
# 5. CLASSIFICAÇÃO DO MODELO (estilo seta aparecendo)
# =================================================
class ClassificacaoTanque(Scene):
    def construct(self):
        title = Text("Classificação do Modelo", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(0.6)

        props = [
            "Estático × Dinâmico",
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
            "SISO",
            "Contínuo",
            "Invariante no tempo",
            "Concentrado",
            "Determinístico",
            "Forçado"
        ]
        colors = [TEAL, ORANGE, GREEN, PURPLE, BLUE_B, GOLD, RED, YELLOW]

        left_col = VGroup(*[Text(p, font_size=26, color=WHITE) for p in props]).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(LEFT*3)
        self.play(LaggedStart(*[FadeIn(m, shift=LEFT) for m in left_col], lag_ratio=0.12))
        self.wait(0.6)

        for i, prop in enumerate(left_col):
            arrow = Arrow(start=prop.get_right(), end=prop.get_right() + RIGHT*1.8, color=colors[i], stroke_width=3)
            val = Text(vals[i], font_size=26, color=colors[i]).next_to(arrow, RIGHT, buff=0.2)
            self.play(GrowArrow(arrow), FadeIn(val, shift=RIGHT))
            self.play(prop.animate.set_color(colors[i]), val.animate.scale(1.05), run_time=0.35)
            self.wait(0.4)
            self.play(val.animate.scale(1/1.05))
        self.wait(1.0)
        self.play(FadeOut(title), *[FadeOut(m) for m in self.mobjects])
        self.wait(0.3)
