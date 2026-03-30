from manim import *
import numpy as np

class ModelagemEmpiricaIdentificacao(Scene):
    def construct(self):
        # ======================================================
        # TÍTULO PRINCIPAL
        # ======================================================
        title = Text(
            "Modelagem Empírica e Identificação de Sistemas", 
            font_size=36,
            color=BLUE
        ).to_edge(UP, buff=0.5)
        
        subtitle = Text(
            "Abordagens Caixa Preta e Análise de Resposta",
            font_size=24
        ).next_to(title, DOWN, buff=0.3)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)

        intro_text = VGroup(
            Text("Modelagem de processos contínuos:", font_size=22),
            Text("Três abordagens principais", font_size=22, color=YELLOW),
            Text("Baseadas no nível de conhecimento disponível", font_size=22)
        ).arrange(DOWN, buff=0.3).next_to(subtitle, DOWN, buff=0.6)

        for t in intro_text:
            self.play(FadeIn(t, shift=RIGHT))
            self.wait(0.5)

        self.wait(1)
        self.play(
            FadeOut(intro_text),
            FadeOut(subtitle)
        )

        # ======================================================
        # ABORDAGENS DE MODELAGEM
        # ======================================================
        sec1_title = Text("Abordagens de Modelagem", font_size=32, color=YELLOW)
        sec1_title.next_to(title, DOWN, buff=0.6)
        
        self.play(FadeIn(sec1_title))
        self.wait(0.5)

        # Caixa Transparente
        transparente = VGroup(
            Text("Caixa Transparente", font_size=26, color=GREEN),
            Text("• Modelagem Fenomenológica", font_size=20),
            Text("• Baseada em leis físicas", font_size=20),
            Text("• Equações fundamentais", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(sec1_title, DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(t) for t in transparente], lag_ratio=0.2))
        self.wait(1.5)

        # Caixa Cinza
        cinza = VGroup(
            Text("Caixa Cinza", font_size=26, color=ORANGE),
            Text("• Abordagem Mista", font_size=20),
            Text("• Leis físicas + dados", font_size=20),
            Text("• Parâmetros ajustáveis", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(transparente, DOWN, buff=0.3)

        self.play(LaggedStart(*[FadeIn(t) for t in cinza], lag_ratio=0.2))
        self.wait(1.5)

        # Caixa Preta
        preta = VGroup(
            Text("Caixa Preta", font_size=26, color=RED),
            Text("• Identificação de Sistemas", font_size=20),
            Text("• Baseada apenas em dados", font_size=20),
            Text("• Relação entrada-saída", font_size=20)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(cinza, DOWN, buff=0.3)

        self.play(LaggedStart(*[FadeIn(t) for t in preta], lag_ratio=0.2))
        self.wait(2)

        # Mover tudo para esquerda para criar espaço para o gráfico
        self.play(
            transparente.animate.scale(0.8).to_edge(LEFT, buff=0.8).shift(UP * 0.5),
            cinza.animate.scale(0.8).next_to(transparente, DOWN, buff=0.2, aligned_edge=LEFT),
            preta.animate.scale(0.8).next_to(cinza, DOWN, buff=0.2, aligned_edge=LEFT),
            sec1_title.animate.scale(0.9).to_edge(LEFT, buff=0.8),
            title.animate.scale(0.9).to_edge(UP, buff=0.3).shift(LEFT * 0.5)
        )

        # Gráfico ilustrativo
        graph_title = Text("Exemplo: Resposta ao Degrau", font_size=26, color=BLUE)
        graph_title.to_edge(UP).shift(RIGHT * 1.8)
        
        self.play(FadeIn(graph_title))

        axes = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1.4, 0.2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "font_size": 16
            }
        ).scale(0.5)
        axes.next_to(graph_title, DOWN, buff=0.3).shift(RIGHT * 1.5)

        # Degrau de entrada
        entrada = axes.plot(lambda t: 1 if t > 2 else 0, x_range=[0, 10], color=GREEN)
        # Resposta do sistema
        resposta = axes.plot(lambda t: 0 if t < 2 else (1 - np.exp(-(t-2)/2)), x_range=[0, 10], color=RED)

        labels = axes.get_axis_labels(
            x_label=MathTex("t").scale(0.5),
            y_label=MathTex("y(t)").scale(0.5)
        )

        self.play(Create(axes), Write(labels))
        self.play(Create(entrada), run_time=1)
        self.play(Create(resposta), run_time=2)

        # Legenda
        legenda_entrada = Text("Entrada (u)", font_size=16, color=GREEN)
        legenda_saida = Text("Saída (y)", font_size=16, color=RED)
        legenda = VGroup(legenda_entrada, legenda_saida).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legenda.next_to(axes, DOWN, buff=0.2).shift(LEFT * 0.5)
        
        self.play(FadeIn(legenda))
        self.wait(2)

        # Limpar seção completamente
        tudo_sec1 = VGroup(transparente, cinza, preta, sec1_title, graph_title, axes, labels, entrada, resposta, legenda)
        self.play(FadeOut(tudo_sec1), FadeOut(title))
        
        # ======================================================
        # MOTIVAÇÃO PARA MODELAGEM EMPÍRICA
        # ======================================================
        # Título da seção
        sec2_main_title = Text("Motivação para Modelagem Empírica", font_size=38, color=GREEN).to_edge(UP, buff=0.5)
        self.play(FadeIn(sec2_main_title))
        self.wait(0.5)

        sec2_title = Text("Desafios da Modelagem Fenomenológica", font_size=28, color=YELLOW)
        sec2_title.next_to(sec2_main_title, DOWN, buff=0.6)
        
        self.play(FadeIn(sec2_title))
        self.wait(0.5)

        motivacoes = VGroup(
            Text("• Pode ser excessivamente complexa", font_size=22),
            Text("• Demorada de implementar", font_size=22),
            Text("• Dificuldade com perturbações", font_size=22),
            Text("• Incertezas do processo", font_size=22)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(sec2_title, DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(m) for m in motivacoes], lag_ratio=0.2))
        self.wait(2)

        # Vantagens da Identificação
        vantagens_title = Text("Vantagens da Identificação:", font_size=28, color=GREEN)
        vantagens_title.next_to(motivacoes, DOWN, buff=0.5)
        vantagens = VGroup(
            Text("✓ Estrutura conhecida de modelos", font_size=22),
            Text("✓ Fácil inclusão de perturbações", font_size=22),
            Text("✓ Método analítico e numérico", font_size=22)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(vantagens_title, DOWN, buff=0.4)

        self.play(FadeIn(vantagens_title))
        self.play(LaggedStart(*[FadeIn(v) for v in vantagens], lag_ratio=0.2))
        self.wait(2)

        # Limitações
        limitacoes_title = Text("Limitações:", font_size=28, color=RED)
        limitacoes_title.next_to(vantagens, DOWN, buff=0.5)
        limitacoes = VGroup(
            Text("• Conhecimento limitado do sistema", font_size=22),
            Text("• Capacidade preditiva restrita", font_size=22),
            Text("• Dependente dos dados experimentais", font_size=22)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(limitacoes_title, DOWN, buff=0.4)

        self.play(FadeIn(limitacoes_title))
        self.play(LaggedStart(*[FadeIn(l) for l in limitacoes], lag_ratio=0.2))
        self.wait(2)

        # Mover tudo para a esquerda para criar espaço para o diagrama
        left_content = VGroup(sec2_title, motivacoes, vantagens_title, vantagens, limitacoes_title, limitacoes)
        self.play(
            left_content.animate.scale(0.8).to_edge(LEFT, buff=0.8).shift(UP * 0.3),
            sec2_main_title.animate.scale(0.9).to_edge(LEFT, buff=0.8).shift(DOWN * 0.5)
        )

        # Diagrama ilustrativo
        diagram_title = Text("Comparação de Abordagens", font_size=26, color=BLUE)
        diagram_title.to_edge(UP).shift(RIGHT * 1.8)
        
        self.play(FadeIn(diagram_title))

        # Criar diagrama simples
        # Caixa transparente
        transparent_box = Rectangle(width=2.2, height=1.2, color=GREEN, fill_opacity=0.2)
        transparent_box.move_to(RIGHT * 3.5 + UP * 1.5)
        transparent_label = Text("Caixa\nTransparente", font_size=16).move_to(transparent_box)
        
        # Caixa cinza
        gray_box = Rectangle(width=2.2, height=1.2, color=ORANGE, fill_opacity=0.2)
        gray_box.move_to(RIGHT * 3.5 + UP * 0.2)
        gray_label = Text("Caixa\nCinza", font_size=16).move_to(gray_box)
        
        # Caixa preta
        black_box = Rectangle(width=2.2, height=1.2, color=RED, fill_opacity=0.2)
        black_box.move_to(RIGHT * 3.5 + DOWN * 1.1)
        black_label = Text("Caixa\nPreta", font_size=16).move_to(black_box)
        
        # Setas
        arrow_transparent = Arrow(
            transparent_box.get_bottom(),
            gray_box.get_top(),
            color=GREEN,
            buff=0.1
        )
        
        arrow_gray = Arrow(
            gray_box.get_bottom(),
            black_box.get_top(),
            color=ORANGE,
            buff=0.1
        )
        
        # Labels das setas
        less_physics = Text("Menos física", font_size=14, color=WHITE).next_to(arrow_transparent, RIGHT, buff=0.1)
        more_data = Text("Mais dados", font_size=14, color=WHITE).next_to(arrow_gray, RIGHT, buff=0.1)
        
        diagram = VGroup(
            transparent_box, transparent_label,
            gray_box, gray_label,
            black_box, black_label,
            arrow_transparent, arrow_gray,
            less_physics, more_data
        ).scale(0.9)
        
        self.play(Create(diagram), run_time=2)
        self.wait(2)

        # Limpar seção completamente
        tudo_sec2 = VGroup(left_content, sec2_main_title, diagram_title, diagram)
        self.play(FadeOut(tudo_sec2))

                # ======================================================
        # MÉTODO GRÁFICO: RESPOSTA AO DEGRAU
        # ======================================================
        sec3_main_title = Text("Método Gráfico: Resposta ao Degrau", font_size=38, color=PURPLE).to_edge(UP, buff=0.5)
        self.play(FadeIn(sec3_main_title))
        self.wait(0.5)

        sec3_title = Text("Curva de Reação do Processo", font_size=28, color=YELLOW)
        sec3_title.next_to(sec3_main_title, DOWN, buff=0.4)
        
        self.play(FadeIn(sec3_title))
        self.wait(0.5)

        # Primeira Ordem (lado esquerdo)
        primeira_ordem = VGroup(
            Text("Para sistemas de Primeira Ordem:", font_size=22, color=GREEN),
            MathTex(r"K_p = \frac{\Delta y}{\Delta u}").scale(0.8),
            MathTex(r"\tau_p \rightarrow \text{Constante de Tempo}").scale(0.8),
            Text("", font_size=10),
            Text("Onde:", font_size=18),
            Text("• Kₚ = Ganho do Processo", font_size=18),
            Text("• τₚ = Constante de tempo", font_size=18)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        # Segunda Ordem (lado direito)
        segunda_ordem = VGroup(
            Text("Para sistemas de Segunda Ordem:", font_size=22, color=BLUE),
            Text("• Tempo de Subida (tₛ)", font_size=18),
            Text("• Sobre-sinal Máximo (Mₚ)", font_size=18),
            Text("• Instante de Pico (tₚ)", font_size=18),
            Text("• Tempo de Acomodação (tₐ)", font_size=18),
            Text("", font_size=10),
            Text("Permitem determinar:", font_size=18),
            Text("• Fator de amortecimento (ζ)", font_size=16),
            Text("• Frequência natural (ωₙ)", font_size=16)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.15)

        # Organizar em duas colunas
        colunas = VGroup(primeira_ordem, segunda_ordem).arrange(RIGHT, buff=1.5)
        colunas.next_to(sec3_title, DOWN, buff=0.4).to_edge(LEFT, buff=1.0)

        # Descrição geral acima das colunas
        descricao = Text("Eficaz para obter parâmetros de funções de transferência simples", 
                        font_size=20, color=YELLOW)
        descricao.next_to(sec3_title, DOWN, buff=0.2)

        self.play(FadeIn(descricao))
        self.wait(0.5)
        self.play(FadeIn(colunas))
        self.wait(2)

        # Mover tudo para esquerda (mais compacto)
        left_group = VGroup(sec3_main_title, sec3_title, descricao, colunas)
        self.play(
            left_group.animate.scale(0.75).to_edge(LEFT, buff=0.4).shift(UP * 0.1)
        )

        # Gráfico de resposta ao degrau com parâmetros
        graph_title2 = Text("Resposta ao Degrau - 1ª Ordem", font_size=22, color=BLUE)
        graph_title2.to_edge(UP, buff=0.5).shift(RIGHT * 3.5)
        
        self.play(FadeIn(graph_title2))

        axes2 = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1.4, 0.2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "font_size": 14
            }
        ).scale(0.4)
        axes2.next_to(graph_title2, DOWN, buff=0.3).shift(RIGHT * 2.8)

        # Plotar resposta
        Kp = 1.0
        tau = 2.0
        resposta_plot = axes2.plot(
            lambda t: 0 if t < 2 else Kp * (1 - np.exp(-(t-2)/tau)),
            x_range=[0, 10],
            color=BLUE
        )

        # Degrau de entrada
        entrada_plot = axes2.plot(
            lambda t: 0.2 if t < 2 else 0.7,
            x_range=[0, 10],
            color=GREEN
        )

        labels2 = axes2.get_axis_labels(
            x_label=MathTex("t").scale(0.4),
            y_label=MathTex("y(t)").scale(0.4)
        )

        self.play(Create(axes2), Write(labels2))
        self.play(Create(entrada_plot), run_time=1)
        self.play(Create(resposta_plot), run_time=2)

        # Mostrar Kp
        delta_y_line = DashedLine(
            start=axes2.c2p(0, 0.2),
            end=axes2.c2p(0, 0.7),
            color=YELLOW
        )
        delta_u_line = DashedLine(
            start=axes2.c2p(8, 0.2),
            end=axes2.c2p(8, 1.0),
            color=YELLOW
        )
        
        delta_y_label = MathTex(r"\Delta u").scale(0.45).next_to(delta_y_line, LEFT, buff=0.1)
        delta_u_label = MathTex(r"\Delta y").scale(0.45).next_to(delta_u_line, RIGHT, buff=0.1)
        
        self.play(Create(delta_y_line), Create(delta_u_line))
        self.play(FadeIn(delta_y_label), FadeIn(delta_u_label))
        
        # Mostrar constante de tempo
        tau_line = axes2.get_vertical_line(
            axes2.c2p(2 + tau, Kp * (1 - np.exp(-1)) + 0.2),
            color=RED
        )
        tau_text = MathTex(r"\tau_p").scale(0.45).next_to(tau_line, UP, buff=0.1)
        
        self.play(Create(tau_line), FadeIn(tau_text))
        self.wait(2)

        # Limpar seção completamente
        tudo_sec3 = VGroup(left_group, graph_title2, axes2, labels2, entrada_plot, resposta_plot,
                          delta_y_line, delta_u_line, delta_y_label, delta_u_label, tau_line, tau_text)
        self.play(FadeOut(tudo_sec3))

        # ======================================================
        # PROCEDIMENTO DE IDENTIFICAÇÃO
        # ======================================================
        sec4_main_title = Text("Procedimento de Identificação", font_size=38, color=ORANGE).to_edge(UP, buff=0.5)
        self.play(FadeIn(sec4_main_title))
        self.wait(0.5)

        # Etapas em colunas para melhor organização
        etapa1 = VGroup(
            Text("1. Projeto do Experimento", font_size=22, color=GREEN),
            Text("• Sinais de excitação", font_size=18),
            Text("  - PRBS", font_size=16),
            Text("  - Pulsos", font_size=16),
            Text("  - Ondas", font_size=16)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        etapa2 = VGroup(
            Text("2. Escolha da Estrutura", font_size=22, color=BLUE),
            Text("• Funções de transferência", font_size=18),
            Text("• Equações diferenciais", font_size=18),
            Text("• Modelos ARX/ARMAX", font_size=16)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        etapa3 = VGroup(
            Text("3. Estimação de Parâmetros", font_size=22, color=ORANGE),
            Text("• Minimização de erros", font_size=18),
            Text("• Métodos estatísticos", font_size=18),
            Text("• Algoritmos de otimização", font_size=16)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        etapa4 = VGroup(
            Text("4. Validação do Modelo", font_size=22, color=RED),
            Text("• Dados independentes", font_size=18),
            Text("• Testes de desempenho", font_size=18),
            Text("• Análise de resíduos", font_size=16)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.1)

        # Organizar em 2x2 grid
        row1 = VGroup(etapa1, etapa2).arrange(RIGHT, buff=1.5)
        row2 = VGroup(etapa3, etapa4).arrange(RIGHT, buff=1.5)
        etapas_grid = VGroup(row1, row2).arrange(DOWN, buff=0.8)
        etapas_grid.next_to(sec4_main_title, DOWN, buff=0.6)

        self.play(LaggedStart(*[FadeIn(e) for e in [etapa1, etapa2, etapa3, etapa4]], lag_ratio=0.2))
        self.wait(2)

        # Mover para esquerda para criar espaço para o diagrama
        left_content4 = VGroup(sec4_main_title, etapas_grid)
        self.play(
            left_content4.animate.scale(0.8).to_edge(LEFT, buff=0.6).shift(UP * 0.2)
        )

        # Diagrama de fluxo
        fluxo_title = Text("Fluxo de Identificação", font_size=24, color=BLUE)
        fluxo_title.to_edge(UP).shift(RIGHT * 2.0)
        
        self.play(FadeIn(fluxo_title))

        # Criar diagrama de fluxo simplificado
        pos_x = 4
        
        # Caixas do fluxo
        caixa1 = Rectangle(width=2.0, height=0.8, color=GREEN, fill_opacity=0.2)
        caixa1.move_to(RIGHT * pos_x + UP * 1.2)
        label1 = Text("Experimento", font_size=12).move_to(caixa1)
        
        caixa2 = Rectangle(width=2.0, height=0.8, color=BLUE, fill_opacity=0.2)
        caixa2.move_to(RIGHT * pos_x + UP * 0.2)
        label2 = Text("Estrutura", font_size=12).move_to(caixa2)
        
        caixa3 = Rectangle(width=2.0, height=0.8, color=ORANGE, fill_opacity=0.2)
        caixa3.move_to(RIGHT * pos_x + DOWN * 0.8)
        label3 = Text("Estimação", font_size=12).move_to(caixa3)
        
        caixa4 = Rectangle(width=2.0, height=0.8, color=RED, fill_opacity=0.2)
        caixa4.move_to(RIGHT * pos_x + DOWN * 1.8)
        label4 = Text("Validação", font_size=12).move_to(caixa4)
        
        # Setas
        seta1 = Arrow(caixa1.get_bottom(), caixa2.get_top(), buff=0.1)
        seta2 = Arrow(caixa2.get_bottom(), caixa3.get_top(), buff=0.1)
        seta3 = Arrow(caixa3.get_bottom(), caixa4.get_top(), buff=0.1)
        
        # Setas de retorno (feedback)
        seta_feedback = Arrow(
            caixa4.get_left() + DOWN * 0.2,
            caixa1.get_left() + UP * 0.2,
            buff=0.1,
            color=YELLOW
        )
        feedback_label = Text("Revisar", font_size=10, color=YELLOW).next_to(seta_feedback, LEFT, buff=0.1)
        
        fluxo = VGroup(
            caixa1, label1, caixa2, label2, caixa3, label3, caixa4, label4,
            seta1, seta2, seta3, seta_feedback, feedback_label
        ).scale(0.8)
        
        self.play(Create(fluxo), run_time=2)
        
        # Sinais de excitação
        sinais_title = Text("Sinais de Excitação Comuns:", font_size=18, color=WHITE)
        sinais_title.next_to(fluxo, DOWN, buff=0.3).shift(RIGHT * 0.5)
        
        # Exemplos de sinais PRBS
        prbs_signal = VGroup()
        for i in range(8):
            value = 0.5 if i % 2 == 0 else 1.0
            prbs_signal.add(Line(
                start=RIGHT * (i * 0.4) + DOWN * 2.2,
                end=RIGHT * ((i+1) * 0.4) + DOWN * 2.2,
                stroke_width=2,
                color=GREEN
            ).shift(UP * value * 0.25))
        
        prbs_label = Text("PRBS", font_size=12, color=GREEN).next_to(prbs_signal, DOWN, buff=0.1)
        
        self.play(FadeIn(sinais_title), FadeIn(prbs_signal), FadeIn(prbs_label))
        self.wait(2)

        # Limpar seção completamente
        tudo_sec4 = VGroup(left_content4, fluxo_title, fluxo, sinais_title, prbs_signal, prbs_label)
        self.play(FadeOut(tudo_sec4))

        # ======================================================
        # VALIDAÇÃO E CRITÉRIOS DE SELEÇÃO
        # ======================================================
        sec5_main_title = Text("Validação e Critérios de Seleção", font_size=36, color=RED).to_edge(UP, buff=0.5)
        self.play(FadeIn(sec5_main_title))
        self.wait(0.5)

        sec5_title = Text("Critérios de Seleção", font_size=28, color=YELLOW)
        sec5_title.next_to(sec5_main_title, DOWN, buff=0.6)
        
        self.play(FadeIn(sec5_title))
        self.wait(0.5)

        validacao = VGroup(
            Text("• Critério de Akaike (AIC)", font_size=22),
            Text("• MDL/BIC", font_size=22),
            Text("• Erro de Predição Final (FPE)", font_size=22),
            Text("• Índice de Ajuste (fit%)", font_size=22, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).next_to(sec5_title, DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(v) for v in validacao], lag_ratio=0.2))
        self.wait(1.5)

        # Fórmulas
        formulas = VGroup(
            MathTex(r"\text{AIC} = -2\ln(L) + 2k").scale(0.9),
            MathTex(r"\text{fit\%} = 100 \times \left(1 - \frac{\|y - \hat{y}\|}{\|y - \bar{y}\|}\right)").scale(0.8)
        ).arrange(DOWN, buff=0.3).next_to(validacao, DOWN, buff=0.3)

        self.play(LaggedStart(*[Write(f) for f in formulas], lag_ratio=0.3))
        self.wait(2)

        # Mover para esquerda para criar espaço para o gráfico
        left_content3 = VGroup(sec5_title, validacao, formulas)
        self.play(
            left_content3.animate.scale(0.8).to_edge(LEFT, buff=0.8).shift(UP * 0.5),
            sec5_main_title.animate.scale(0.9).to_edge(LEFT, buff=0.8).shift(DOWN * 0.5)
        )

        # Gráfico de validação
        val_title = Text("Comparação: Real vs Modelo", font_size=26, color=BLUE)
        val_title.to_edge(UP).shift(RIGHT * 1.8)
        
        self.play(FadeIn(val_title))

        axes_val = Axes(
            x_range=[0, 10, 2],
            y_range=[0, 1.4, 0.2],
            tips=False,
            axis_config={
                "include_numbers": True,
                "font_size": 16
            }
        ).scale(0.5)
        axes_val.next_to(val_title, DOWN, buff=0.3).shift(RIGHT * 1.5)

        # Dados "reais" (com ruído)
        t_vals = np.linspace(0, 10, 100)
        y_real = 1 - np.exp(-t_vals/2)
        # Adicionar ruído
        np.random.seed(42)
        ruido = np.random.normal(0, 0.05, len(t_vals))
        y_real_ruido = y_real + ruido
        
        # Modelo ajustado
        y_model = 0.95 * (1 - np.exp(-t_vals/1.8))

        # Converter para pontos do Manim
        real_points = [axes_val.c2p(t, max(0, y)) for t, y in zip(t_vals, y_real_ruido)]
        model_points = [axes_val.c2p(t, y) for t, y in zip(t_vals, y_model)]
        
        real_curve = VMobject()
        real_curve.set_points_smoothly(real_points)
        real_curve.set_color(RED)
        real_curve.set_stroke(width=2)
        
        model_curve = VMobject()
        model_curve.set_points_smoothly(model_points)
        model_curve.set_color(BLUE)
        model_curve.set_stroke(width=2, opacity=0.8)

        labels_val = axes_val.get_axis_labels(
            x_label=MathTex("t").scale(0.5),
            y_label=MathTex("y").scale(0.5)
        )

        self.play(Create(axes_val), Write(labels_val))
        self.play(Create(real_curve), run_time=2)
        self.play(Create(model_curve), run_time=2)

        # Legenda
        legenda_real = Text("Dados Reais", font_size=16, color=RED)
        legenda_model = Text("Modelo Ajustado", font_size=16, color=BLUE)
        legenda_val = VGroup(legenda_real, legenda_model).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        legenda_val.next_to(axes_val, DOWN, buff=0.2).shift(LEFT * 0.5)
        
        self.play(FadeIn(legenda_val))
        
        # Mostrar fit%
        fit_value = MathTex(r"\text{fit\%} = 92.3\%").scale(0.8)
        fit_value.next_to(axes_val, DOWN, buff=0.2).shift(RIGHT * 0.5)
        
        self.play(FadeIn(fit_value))
        self.wait(2)

        # Limpar seção completamente
        tudo_sec5 = VGroup(left_content3, sec5_main_title, val_title, axes_val, labels_val, real_curve, model_curve, legenda_val, fit_value)
        self.play(FadeOut(tudo_sec5))

        # ======================================================
        # ANALOGIA E CONCLUSÃO
        # ======================================================
        analogia_title = Text("Analogia para Compreensão", font_size=36, color=BLUE).to_edge(UP, buff=0.8)
        
        self.play(FadeIn(analogia_title))
        self.wait(0.5)

        analogia_text = VGroup(
            Text("Modelagem 'Caixa Preta' é como:", font_size=28, color=YELLOW),
            Text("Aprender a dirigir um carro novo", font_size=26),
            Text("sem nunca abrir o capô.", font_size=26),
        ).arrange(DOWN, buff=0.2).scale(0.9).next_to(analogia_title, DOWN, buff=0.4)
        
        self.play(LaggedStart(*[FadeIn(t, shift=UP*0.3) for t in analogia_text], lag_ratio=0.15))
        self.wait(3)

        # Conclusão final
        conclusao_title = Text("Conclusão", font_size=34, color=BLUE)
        conclusao_title.next_to(analogia_text, DOWN, buff=0.8)
        
        self.play(FadeIn(conclusao_title))
        self.wait(0.5)

        conclusao_points = VGroup(
            Text("A identificação de sistemas é:", font_size=26),
            Text("• Uma ferramenta prática e eficaz", font_size=24, color=GREEN),
            Text("• Baseada em dados de entrada-saída", font_size=24),
            Text("• Complementar à modelagem física", font_size=24),
            Text("• Essencial para controle moderno", font_size=24, color=YELLOW)
        ).arrange(DOWN, buff=0.3, aligned_edge=LEFT).next_to(conclusao_title, DOWN, buff=0.4)

        self.play(LaggedStart(*[FadeIn(p, shift=UP*0.2) for p in conclusao_points], lag_ratio=0.2))
        self.wait(3)

        # Fade out final
        tudo_final = VGroup(analogia_title, analogia_text, conclusao_title, conclusao_points)
        self.play(FadeOut(tudo_final))
        self.wait(1)