from manim import *
import numpy as np

class FuncaoTransferenciaEstabilidade(Scene):
    def construct(self):
        # ======================================================
        # TÍTULO PRINCIPAL (mantém visível durante todo o vídeo)
        # ======================================================
        title = Text(
            "Função de Transferência e Resposta Dinâmica", 
            font_size=36,  # Reduzido de 44
            color=BLUE
        ).to_edge(UP, buff=0.5)  # Posicionado no topo com espaço
        
        subtitle = Text(
            "Ganho, constante de tempo e sistemas de 2ª ordem",
            font_size=24  # Reduzido de 30
        ).next_to(title, DOWN, buff=0.3)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)

        intro_text = VGroup(
            Text("A função de transferência descreve sistemas", font_size=22),  # Reduzido
            Text("Relação entrada-saída no domínio s", font_size=22),
            Text("Permite análise de estabilidade e desempenho", font_size=22, color=YELLOW)
        ).arrange(DOWN, buff=0.3).next_to(subtitle, DOWN, buff=0.6)  # Espaço reduzido

        for t in intro_text:
            self.play(FadeIn(t, shift=RIGHT))
            self.wait(0.5)  # Reduzido

        self.wait(1)
        
        # Remover apenas introdução, manter título principal
        self.play(
            FadeOut(intro_text),
            FadeOut(subtitle)
        )

        # ======================================================
        # FUNÇÃO DE TRANSFERÊNCIA SISO
        # ======================================================
        s1_title = Text("Função de Transferência SISO", font_size=32, color=YELLOW)  # Reduzido
        s1_title.next_to(title, DOWN, buff=0.6)  # Posicionado abaixo do título principal
        
        self.play(FadeIn(s1_title))
        self.wait(0.5)  # Reduzido

        eq_tf = MathTex(
            r"G(s) = \frac{Y(s)}{U(s)}"
        ).scale(1.2).next_to(s1_title, DOWN, buff=0.4)  # Reduzido espaçamento

        self.play(Write(eq_tf))
        self.wait(0.5)

        comment_tf = VGroup(
            Text("• Relação saída/entrada no domínio s", font_size=20),  # Reduzido
            Text("• Propriedade do sistema", font_size=20),
            Text("• Representação entrada-saída", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(eq_tf, DOWN, buff=0.3)  # Reduzido

        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT) for c in comment_tf], lag_ratio=0.15))
        self.wait(1)

        # Forma polinomial (mais compacta)
        poly_tf = MathTex(
            r"G(s)=\frac{b_0 s^n+\cdots+b_n}{s^n+a_1 s^{n-1}+\cdots+a_n}"
        ).scale(0.8).next_to(comment_tf, DOWN, buff=0.3)  # Reduzido tamanho e espaçamento

        self.play(Write(poly_tf))
        self.wait(1.5)

        # Limpar seção (mas manter título principal)
        self.play(
            FadeOut(s1_title),
            FadeOut(eq_tf),
            FadeOut(comment_tf),
            FadeOut(poly_tf),
            FadeOut(title)
        )

        # ======================================================
        # SISTEMAS DE PRIMEIRA ORDEM
        # ======================================================
        s2_title = Text("Sistemas de Primeira Ordem", font_size=32, color=GREEN)
        s2_title.next_to(title, DOWN, buff=0.6)
        
        self.play(FadeIn(s2_title))
        self.wait(0.5)

        eq_1st = MathTex(
            r"G(s)=\frac{K_p}{\tau s + 1}"
        ).scale(1.2).next_to(s2_title, DOWN, buff=0.4)

        self.play(Write(eq_1st))
        self.wait(0.5)

        defs_1st = VGroup(
            MathTex(r"K_p \rightarrow \text{ganho estático}").scale(0.8),
            MathTex(r"\tau \rightarrow \text{constante de tempo}").scale(0.8),
            Text("Resposta exponencial ao degrau", font_size=20, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(eq_1st, DOWN, buff=0.3)

        self.play(LaggedStart(*[FadeIn(p) for p in defs_1st], lag_ratio=0.2))
        self.wait(1.5)

        # ======================================================
        # RESPOSTA AO DEGRAU - 1ª ORDEM
        # ======================================================
        # Agrupar conteúdo à esquerda
        left_content = VGroup(s2_title, eq_1st, defs_1st)
        
        # Mover conteúdo para a esquerda e reduzir
        self.play(
            left_content.animate.scale(0.8).to_edge(LEFT, buff=0.5).shift(UP * 0.3)  # Ajustado
        )
        self.wait(0.3)

        # Adicionar título do gráfico à direita
        graph_title = Text("Resposta ao Degrau", font_size=26, color=BLUE)
        graph_title.to_edge(UP).shift(RIGHT * 2.5)  # Movido mais para direita
        
        self.play(FadeIn(graph_title))
        self.wait(0.3)

        # Configurar eixos - lado direito
        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1.4, 0.2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "font_size": 16
            }
        ).scale(0.55)  # Reduzido
        
        # Posicionar eixos no lado direito
        axes.next_to(graph_title, DOWN, buff=0.3).shift(RIGHT * 1)

        labels = axes.get_axis_labels(
            x_label=MathTex("t").scale(0.55),
            y_label=MathTex("y(t)").scale(0.55)
        )

        self.play(Create(axes), Write(labels))
        self.wait(0.3)

        # Parâmetros
        tau = 2.0
        Kp = 1.0

        # Plotar resposta
        response = axes.plot(
            lambda t: Kp * (1 - np.exp(-t / tau)),
            x_range=[0, 10],
            color=BLUE
        )

        self.play(Create(response), run_time=1.2)
        self.wait(0.3)

        # Linha da constante de tempo
        tau_line = axes.get_vertical_line(
            axes.c2p(tau, Kp * (1 - np.exp(-1))),
            color=YELLOW
        )

        tau_text = MathTex(r"t=\tau").scale(0.55).next_to(tau_line, UP, buff=0.1)

        self.play(Create(tau_line), FadeIn(tau_text))
        
        # Texto explicativo
        tau_explain = MathTex(r"y(\tau) = 0.632K_p").scale(0.65)
        tau_explain.next_to(axes, DOWN, buff=0.15)

        self.play(FadeIn(tau_explain))
        self.wait(1.5)

        # Limpar seção (manter título principal)
        self.play(
            FadeOut(left_content),
            FadeOut(graph_title),
            FadeOut(axes), FadeOut(labels), FadeOut(response),
            FadeOut(tau_line), FadeOut(tau_text), FadeOut(tau_explain)
        )

        # ======================================================
        # SISTEMAS DE SEGUNDA ORDEM
        # ======================================================
        s3_title = Text("Sistemas de Segunda Ordem", font_size=32, color=PURPLE)
        s3_title.next_to(title, DOWN, buff=0.6)
        
        self.play(FadeIn(s3_title))
        self.wait(0.5)

        eq_2nd = MathTex(
            r"G(s)=\frac{K_p}{\tau^2 s^2 + 2\xi\tau s + 1}"
        ).scale(1.2).next_to(s3_title, DOWN, buff=0.4)

        self.play(Write(eq_2nd))
        self.wait(0.5)

        params_2nd = VGroup(
            MathTex(r"\xi \rightarrow \text{fator de amortecimento}").scale(0.8),
            MathTex(r"\tau \rightarrow \text{escala temporal}").scale(0.8),
            Text("Comportamento depende de ξ", font_size=20, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(eq_2nd, DOWN, buff=0.3)

        self.play(LaggedStart(*[FadeIn(p) for p in params_2nd], lag_ratio=0.2))
        self.wait(1.5)

        # ======================================================
        # CASOS DE AMORTECIMENTO
        # ======================================================
        # Agrupar conteúdo à esquerda
        left_content2 = VGroup(s3_title, eq_2nd, params_2nd)
        
        # Mover conteúdo para a esquerda e reduzir
        self.play(
            left_content2.animate.scale(0.8).to_edge(LEFT, buff=0.5).shift(UP * 0.3)
        )
        self.wait(0.3)

        # Casos de amortecimento
        cases = [
            (r"\xi = 0 : \text{Não amortecido}", 0, RED, "Oscilação sustentada"),
            (r"0 < \xi < 1 : \text{Subamortecido}", 0.3, GREEN, "Oscilações decrescentes"),
            (r"\xi = 1 : \text{Criticamente amortecido}", 1.0, BLUE, "Resposta mais rápida sem oscilação"),
            (r"\xi > 1 : \text{Superamortecido}", 1.5, PURPLE, "Resposta lenta, sem oscilação")
        ]

        for case_tex, zeta, color, description in cases:
            # Título do gráfico à direita
            case_title = MathTex(case_tex, font_size=22, color=color)  # Reduzido
            case_title.to_edge(UP).shift(RIGHT * 2.5)
            
            desc_text = Text(description, font_size=18).next_to(case_title, DOWN, buff=0.15)  # Reduzido
            
            self.play(FadeIn(case_title), FadeIn(desc_text))
            self.wait(0.3)

            # Configurar eixos no lado direito
            axes = Axes(
                x_range=[0, 10, 2],
                y_range=[-0.2, 2.0, 0.2],
                tips=False,
                axis_config={
                    "include_numbers": True,
                    "font_size": 14
                }
            ).scale(0.48)  # Reduzido
            axes.next_to(desc_text, DOWN, buff=0.2).shift(RIGHT * 0.5)

            # Função de resposta ao degrau
            if zeta == 0:
                func = lambda t: 1 - np.cos(t)
            elif 0 < zeta < 1:
                wd = np.sqrt(1 - zeta**2)
                func = lambda t: 1 - np.exp(-zeta * t) * (
                    np.cos(wd * t) + zeta / wd * np.sin(wd * t)
                )
            elif zeta == 1:
                func = lambda t: 1 - np.exp(-t) * (1 + t)
            else:
                r1 = -zeta + np.sqrt(zeta**2 - 1)
                r2 = -zeta - np.sqrt(zeta**2 - 1)
                func = lambda t: 1 + (r2*np.exp(r1*t)-r1*np.exp(r2*t))/(r1-r2)

            # Plotar curva
            curve = axes.plot(func, x_range=[0, 10], color=color)
            
            self.play(Create(axes))
            self.play(Create(curve), run_time=1.2)
            self.wait(1.5)

            # Limpar para próximo caso
            self.play(
                FadeOut(axes), 
                FadeOut(curve), 
                FadeOut(case_title),
                FadeOut(desc_text)
            )

        self.play(FadeOut(left_content2))

        # ======================================================
        # SOBREELEVAÇÃO (OVERSHOOT)
        # ======================================================
        s4_title = Text("Sobreelevação (Zero no Numerador)", font_size=30, color=ORANGE)  # Reduzido
        s4_title.next_to(title, DOWN, buff=0.6)
        
        self.play(FadeIn(s4_title))
        self.wait(0.5)

        eq_over = MathTex(
            r"G(s)=\frac{\tau_1 s + 1}{\tau^2 s^2 + 2\xi\tau s + 1}"
        ).scale(1.1).next_to(s4_title, DOWN, buff=0.4)  # Reduzido

        self.play(Write(eq_over))
        self.wait(0.5)

        explain_over = VGroup(
            MathTex(r"\text{Zero no semiplano esquerdo}").scale(0.7),
            MathTex(r"\text{Acelera resposta inicial}").scale(0.7),
            Text("Pode causar sobreelevação", font_size=20, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(eq_over, DOWN, buff=0.3)

        self.play(LaggedStart(*[FadeIn(e) for e in explain_over], lag_ratio=0.2))
        self.wait(1.5)

        # Agrupar conteúdo à esquerda
        left_content3 = VGroup(s4_title, eq_over, explain_over)
        
        # Mover conteúdo para a esquerda e reduzir
        self.play(
            left_content3.animate.scale(0.8).to_edge(LEFT, buff=0.5).shift(UP * 0.3)
        )
        self.wait(0.3)

        # Gráfico de sobreelevação à direita
        graph_title2 = Text("Resposta com Sobreelevação", font_size=24, color=ORANGE)  # Reduzido
        graph_title2.to_edge(UP).shift(RIGHT * 2.5)
        
        self.play(FadeIn(graph_title2))

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 2.0, 0.2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "font_size": 14
            }
        ).scale(0.5)  # Reduzido
        axes.next_to(graph_title2, DOWN, buff=0.3).shift(RIGHT * 0.5)

        overshoot_curve = axes.plot(
            lambda t: 1 - np.exp(-0.5*t)*(np.cos(1.5*t)-0.3*np.sin(1.5*t)),
            x_range=[0, 10],
            color=ORANGE
        )

        self.play(Create(axes))
        self.play(Create(overshoot_curve), run_time=1.2)
        
        # Destacar máximo
        max_point = axes.c2p(1.5, 1.4)
        max_dot = Dot(max_point, color=RED)
        max_label = MathTex(r"y_{\max}").scale(0.55).next_to(max_dot, UP, buff=0.1)
        
        self.play(FadeIn(max_dot), FadeIn(max_label))
        
        overshoot_math = MathTex(r"\text{Overshoot} \approx 40\%").scale(0.65)
        overshoot_math.next_to(axes, DOWN, buff=0.15)
        
        self.play(FadeIn(overshoot_math))
        self.wait(1.5)

        # Limpar seção
        self.play(
            FadeOut(left_content3),
            FadeOut(graph_title2),
            FadeOut(axes), FadeOut(overshoot_curve),
            FadeOut(max_dot), FadeOut(max_label),
            FadeOut(overshoot_math)
        )

        # ======================================================
        # RESPOSTA INVERSA
        # ======================================================
        s5_title = Text("Resposta Inversa (Zero no SPD)", font_size=30, color=RED)  # Reduzido
        s5_title.next_to(title, DOWN, buff=0.6)
        
        self.play(FadeIn(s5_title))
        self.wait(0.5)

        eq_inv = MathTex(
            r"G(s)=\frac{1-\tau_1 s}{\tau^2 s^2 + 2\xi\tau s + 1}"
        ).scale(1.1).next_to(s5_title, DOWN, buff=0.4)

        self.play(Write(eq_inv))
        self.wait(0.5)

        explain_inv = VGroup(
            MathTex(r"\text{Zero no semiplano direito}").scale(0.7),
            MathTex(r"\text{Resposta inicial oposta}").scale(0.7),
            Text("Comportamento não mínimo de fase", font_size=20, color=YELLOW)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).next_to(eq_inv, DOWN, buff=0.3)

        self.play(LaggedStart(*[FadeIn(e) for e in explain_inv], lag_ratio=0.2))
        self.wait(1.5)

        # Agrupar conteúdo à esquerda
        left_content4 = VGroup(s5_title, eq_inv, explain_inv)
        
        # Mover conteúdo para a esquerda e reduzir
        self.play(
            left_content4.animate.scale(0.8).to_edge(LEFT, buff=0.5).shift(UP * 0.3)
        )
        self.wait(0.3)

        # Gráfico de resposta inversa à direita
        graph_title3 = Text("Resposta Inversa", font_size=24, color=RED)
        graph_title3.to_edge(UP).shift(RIGHT * 2.5)
        
        self.play(FadeIn(graph_title3))

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[-0.6, 1.4, 0.2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "font_size": 14
            }
        ).scale(0.5)
        axes.next_to(graph_title3, DOWN, buff=0.3).shift(RIGHT * 0.5)

        inverse_curve = axes.plot(
            lambda t: 1 - np.exp(-t) - 0.6*np.exp(-3*t),
            x_range=[0, 10],
            color=RED
        )

        self.play(Create(axes))
        self.play(Create(inverse_curve), run_time=1.2)
        
        # Destacar resposta inicial negativa
        init_point = axes.c2p(0.5, -0.3)
        init_dot = Dot(init_point, color=BLUE)
        init_label = MathTex(r"y(0^+) < 0").scale(0.55).next_to(init_dot, DOWN, buff=0.1)
        
        self.play(FadeIn(init_dot), FadeIn(init_label))
        self.wait(1.5)

        # Limpar seção
        self.play(
            FadeOut(left_content4),
            FadeOut(graph_title3),
            FadeOut(axes), FadeOut(inverse_curve),
            FadeOut(init_dot), FadeOut(init_label)
        )

        # ======================================================
        # ENCERRAMENTO
        # ======================================================
        # Fade out do título principal
        self.play(FadeOut(title))
        
        conclusion_title = Text("Conclusão", font_size=36, color=BLUE).to_edge(UP)
        
        self.play(FadeIn(conclusion_title))
        self.wait(0.5)

        conclusion_points = VGroup(
            Text("Função de transferência:", font_size=30, color=BLUE).next_to(conclusion_title, DOWN, buff=0.5),
            MathTex(r"\bullet \text{ Descreve dinâmica do sistema}").scale(0.85),
            MathTex(r"\bullet \text{ Polos determinam estabilidade}").scale(0.85),
            MathTex(r"\bullet \text{ Zeros influenciam resposta transitória}").scale(0.85),
            Text("Análise no domínio s simplifica projeto", font_size=26, color=GREEN),
            Text("de controladores", font_size=26, color=GREEN)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT)
        
        conclusion_points[0].next_to(conclusion_title, DOWN, buff=0.5)
        for i in range(1, len(conclusion_points)):
            conclusion_points[i].next_to(conclusion_points[i-1], DOWN, buff=0.25)

        for point in conclusion_points:
            self.play(FadeIn(point, shift=UP * 0.3))
            self.wait(0.3)

        self.wait(2.5)

        # Fade out final
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)