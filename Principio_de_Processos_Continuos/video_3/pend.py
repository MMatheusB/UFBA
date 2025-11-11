from manim import *
import numpy as np
from scipy.integrate import solve_ivp

# --------------------------------------
# 1. Premissas do Modelo do Pêndulo
# --------------------------------------
class PenduloPremissas(Scene):
    def construct(self):
        title = Text("Premissas do Modelo do Pêndulo Simples", font_size=42, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(1.5)

        premissas = BulletedList(
            "A haste é rígida e sem massa.",
            "A massa está concentrada em um ponto na extremidade.",
            "O ponto de apoio é fixo e sem atrito.",
            "O movimento ocorre em um plano vertical.",
            "A única força atuante é a gravidade.",
            "Despreza-se a inércia rotacional da haste.",
            font_size=30
        ).next_to(title, DOWN, buff=0.8)

        for item in premissas:
            self.play(FadeIn(item, shift=DOWN), run_time=1.2)
            self.wait(1)

        outro = Text(
            "Com essas premissas, podemos modelar o movimento angular do pêndulo.",
            font_size=22
        ).next_to(premissas, DOWN, buff=0.8)
        self.play(Write(outro))
        self.wait(3)

        self.play(FadeOut(title), FadeOut(premissas), FadeOut(outro))
        self.wait(1)


# --------------------------------------
# 2. Animação do Pêndulo
# --------------------------------------
class Pendulo(Scene):
    def construct(self):
        # Parâmetros físicos
        L = 2        # Comprimento da haste (em unidades de cena)
        g = 9.81     # Gravidade
        theta0 = 0.4 # Ângulo inicial (rad)
        omega = np.sqrt(g / L)  # Frequência natural (aproximação para pequenos ângulos)

        # Posição do pivô (fixo)
        pivot = UP * 2

        # Cria a haste e o peso inicial
        theta = theta0
        mass_pos = pivot + L * np.array([np.sin(theta), -np.cos(theta), 0])
        corda = Line(pivot, mass_pos, color=WHITE)
        massa = Dot(mass_pos, color=YELLOW, radius=0.1)

        # Adiciona o pêndulo à cena
        self.add(corda, massa)

        # Define função que atualiza a posição do pêndulo no tempo
        def atualizar_pendulo(mob, dt):
            t = self.time
            theta_t = theta0 * np.cos(omega * t)  # Solução analítica para pequenos ângulos
            new_pos = pivot + L * np.array([np.sin(theta_t), -np.cos(theta_t), 0])
            corda.put_start_and_end_on(pivot, new_pos)
            massa.move_to(new_pos)

        # Animação contínua
        corda.add_updater(atualizar_pendulo)
        massa.add_updater(atualizar_pendulo)

        # Executa por 10 segundos
        self.wait(10)


# --------------------------------------
# 3. Classificação do Modelo
# --------------------------------------
class PenduloClassificacao(Scene):
    def construct(self):
            # --- Título ---
            title = Text("Classificação do Modelo Pendulo Simples", font_size=44, color=BLUE).to_edge(UP)
            self.play(Write(title))
            self.wait(1)

            # --- Lista de propriedades e valores ---
            props = [
                "Linear × Não linear",
                "Tempo contínuo × Tempo discreto",
                "Invariante × Variante no tempo",
                "Parâmetros concentrados × Distribuídos",
                "Determinístico × Estocástico",
                "Forçado × Homogêneo"
            ]

            vals = [
                "Não linear",
                "Contínuo no tempo",
                "Invariante no tempo",
                "Parâmetros concentrados",
                "Determinístico",
                "Homogêneo"
            ]

            colors = [GREEN, ORANGE, PURPLE, BLUE_B, RED, GOLD]

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
        
class PenduloEquacoes(Scene):
    def construct(self):
        # --- Título ---
        title = Text("Equações do Modelo do Pêndulo Simples", font_size=40, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(1.5)

        # --- Passo 1: Definição do sistema ---
        eq_def = MathTex(
            r"\text{Para um pêndulo simples de comprimento } L,",
            r"\text{a variável de interesse é o ângulo } \theta(t)."
        ).scale(0.8).next_to(title, DOWN, buff=0.8)
        self.play(Write(eq_def))
        self.wait(3)
        self.play(FadeOut(eq_def))

        # --- Passo 2: Expressões das energias ---
        eq_energias = [
            MathTex(r"T = \frac{1}{2} m (L\dot{\theta})^2", color=YELLOW),  # Energia cinética
            MathTex(r"V = mgL(1 - \cos(\theta))", color=YELLOW)              # Energia potencial
        ]
        for eq in eq_energias:
            eq.next_to(title, DOWN, buff=1)
            self.play(Write(eq))
            self.wait(2)
            self.play(FadeOut(eq))

        # --- Passo 3: Lagrangiano ---
        eq_lagr = MathTex(
            r"\mathcal{L} = T - V = \frac{1}{2} mL^2 \dot{\theta}^2 - mgL(1 - \cos(\theta))"
        ).scale(0.9).next_to(title, DOWN, buff=0.8)
        self.play(Write(eq_lagr))
        self.wait(3)
        self.play(FadeOut(eq_lagr))

        # --- Passo 4: Equação de Lagrange ---sss
        eq_lagrange = MathTex(
            r"\frac{d}{dt}\left(\frac{\partial \mathcal{L}}{\partial \dot{\theta}}\right)"
            r"- \frac{\partial \mathcal{L}}{\partial \theta} = 0"
        ).scale(0.9).next_to(title, DOWN, buff=0.8)
        self.play(Write(eq_lagrange))
        self.wait(3)
        self.play(FadeOut(eq_lagrange))

        # --- Passo 5: Resultado da equação de movimento ---
        eq_motion = MathTex(
            r"mL^2 \ddot{\theta} + mgL \sin(\theta) = 0"
        ).scale(0.9).next_to(title, DOWN, buff=0.8)
        self.play(Write(eq_motion))
        self.wait(3)
        self.play(FadeOut(eq_motion))

        # --- Passo 6: Forma final simplificada ---
        eq_final = MathTex(
            r" \ddot{\theta} = - \frac{g}{L}\sin(\theta)"
        ).scale(1.1).set_color(YELLOW).next_to(title, DOWN, buff=0.8)
        self.play(Write(eq_final))
        self.wait(3)

        outro = Text(
            "Essa é a equação diferencial que descreve o movimento do pêndulo simples.",
            font_size=22
        ).next_to(eq_final, DOWN, buff=0.8)
        self.play(Write(outro))
        self.wait(3)

        self.play(FadeOut(title), FadeOut(eq_final), FadeOut(outro))
        self.wait(1)


class PenduloGraficoNaoLinear(Scene):
    def construct(self):
        # --- Parâmetros físicos ---
        L = 2.0       # comprimento (m)
        g = 9.81      # gravidade (m/s²)
        theta0 = 0.8  # ângulo inicial (rad)
        omega0 = 0.0  # velocidade angular inicial

        # --- Equação diferencial ---
        def pendulo(t, y):
            theta, omega = y
            dtheta_dt = omega
            domega_dt = -(g / L) * np.sin(theta)
            return [dtheta_dt, domega_dt]

        # --- Solução numérica ---
        t_span = (0, 10)
        t_eval = np.linspace(*t_span, 300)
        sol = solve_ivp(pendulo, t_span, [theta0, omega0], t_eval=t_eval)
        t_vals = sol.t
        theta_vals = sol.y[0]

        # --- Título ---
        title = Text("Evolução Temporal do Ângulo", font_size=38, color=BLUE).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # --- Eixos ---
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[-1.0, 1.0, 0.5],
            x_length=8,
            y_length=4,
            axis_config={"include_tip": True},
        ).shift(DOWN * 0.5)
        labels = axes.get_axis_labels("t (s)", r"\theta(t) (rad)")
        self.play(Create(axes), Write(labels))
        self.wait(0.5)

        # --- Gráfico base ---
        full_graph = axes.plot_line_graph(
            x_values=t_vals,
            y_values=theta_vals,
            line_color=GRAY,
            stroke_width=2
        )
        self.play(Create(full_graph), run_time=1.5)
        self.wait(0.5)

        # --- Gráfico em tempo real ---
        trace = VMobject(color=YELLOW, stroke_width=4)
        trace.set_points_as_corners([axes.c2p(t_vals[0], theta_vals[0])])
        dot = Dot(color=RED).move_to(axes.c2p(t_vals[0], theta_vals[0]))

        self.add(trace, dot)

        def update_trace(mob, alpha):
            idx = int(alpha * (len(t_vals) - 1))
            if idx < 1:
                idx = 1
            new_points = [axes.c2p(t_vals[i], theta_vals[i]) for i in range(idx)]
            trace.set_points_as_corners(new_points)
            dot.move_to(new_points[-1])

        # --- Animação principal ---
        self.play(UpdateFromAlphaFunc(trace, update_trace), run_time=8, rate_func=linear)
        self.wait(0.5)

        outro = Text(
            "Modelo simulado",
            font_size=24
        ).next_to(axes, DOWN, buff=0.8)
        self.play(Write(outro))
        self.wait(3)

        # --- Encerramento ---
        self.play(*[FadeOut(mob) for mob in [dot, trace, axes, labels, title, outro, full_graph]])
        self.wait(1)
