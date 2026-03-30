from manim import *

class ModelagemDadosAvancada(Scene):
    def construct(self):

        # ======================================================
        # INTRODUÇÃO
        # ======================================================
        title = Text("Modelagem de Dados Dinâmicos", font_size=44, color=BLUE)
        subtitle = Text(
            "Linearidade e Transformada de Laplace",
            font_size=30
        ).next_to(title, DOWN)

        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)

        intro_text = VGroup(
            Text("Sistemas reais possuem dinâmica", font_size=28),
            Text("Estados evoluem no tempo", font_size=28),
            Text("Precisamos de modelos matemáticos", font_size=28, color=YELLOW)
        ).arrange(DOWN, buff=0.4).next_to(subtitle, DOWN, buff=0.8)

        for t in intro_text:
            self.play(FadeIn(t, shift=RIGHT))
            self.wait(0.6)

        self.wait(1.5)
        self.play(FadeOut(intro_text), FadeOut(subtitle))

        # ======================================================
        # SISTEMAS NÃO LINEARES
        # ======================================================
        s1 = Text("Sistemas Não Lineares", font_size=34, color=RED).to_edge(UP)

        eq_nl = MathTex(
            r"\dot{x}(t) = f(x(t), u(t), t)"
        ).scale(1.2)

        comment_nl = VGroup(
            Text("• Representam fielmente a física", font_size=26),
            Text("• Difíceis de analisar", font_size=26),
            Text("• Solução geralmente numérica", font_size=26)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(eq_nl, DOWN)

        self.play(ReplacementTransform(title, s1))
        self.play(Write(eq_nl))
        self.play(LaggedStart(*[FadeIn(c, shift=RIGHT) for c in comment_nl], lag_ratio=0.3))
        self.wait(2)

        self.play(FadeOut(eq_nl), FadeOut(comment_nl))

        # ======================================================
        # SISTEMAS LINEARES
        # ======================================================
        s2 = Text("Sistemas Lineares", font_size=34, color=GREEN).to_edge(UP)

        eq_lin = MathTex(
            r"\dot{x}(t) = A x(t) + B u(t)"
        ).scale(1.2)

        props_lin = VGroup(
            Text("✔ Superposição", font_size=26, color=GREEN),
            Text("✔ Solução analítica", font_size=26, color=GREEN),
            Text("✔ Base do controle clássico", font_size=26, color=GREEN)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(eq_lin, DOWN)

        self.play(ReplacementTransform(s1, s2))
        self.play(Write(eq_lin))
        self.play(LaggedStart(*[FadeIn(p, shift=LEFT) for p in props_lin], lag_ratio=0.3))
        self.wait(2)

        self.play(FadeOut(eq_lin), FadeOut(props_lin))

        # ======================================================
        # LINEARIZAÇÃO – SÉRIE DE TAYLOR (PARTE NOVA)
        # ======================================================
        s3 = Text("Linearização via Série de Taylor", font_size=34, color=YELLOW).to_edge(UP)
        self.play(ReplacementTransform(s2, s3))

        eq_taylor_full = MathTex(
            r"f(x,u) \approx f(x^\star,u^\star)"
            r"+ \frac{\partial f}{\partial x}\Big|_\star (x-x^\star)"
            r"+ \frac{\partial f}{\partial u}\Big|_\star (u-u^\star)"
            r"+ \frac{1}{2}\frac{\partial^2 f}{\partial x^2}\Big|_\star (x-x^\star)^2"
            r"+ \cdots"
        ).scale(0.85).next_to(s3, DOWN, buff=0.8)

        self.play(Write(eq_taylor_full))
        self.wait(2)

        explain = Text(
            "Expansão em torno do ponto de operação (equilíbrio)",
            font_size=26
        ).next_to(eq_taylor_full, DOWN)

        self.play(FadeIn(explain))
        self.wait(2)

        # Destaque termos de alta ordem
        high_order_box = SurroundingRectangle(
            eq_taylor_full[3:], color=RED
        )
        high_order_txt = Text(
            "Termos de ordem superior",
            font_size=24,
            color=RED
        ).next_to(high_order_box, DOWN)

        self.play(Create(high_order_box), FadeIn(high_order_txt))
        self.wait(2)

        # Remove termos de alta ordem
        eq_taylor_trunc = MathTex(
            r"f(x,u) \approx f(x^\star,u^\star)"
            r"+ \frac{\partial f}{\partial x}\Big|_\star (x-x^\star)"
            r"+ \frac{\partial f}{\partial u}\Big|_\star (u-u^\star)"
        ).scale(1.0).move_to(eq_taylor_full)

        self.play(
            FadeOut(high_order_box),
            FadeOut(high_order_txt),
            Transform(eq_taylor_full, eq_taylor_trunc),
            FadeOut(explain)
        )
        self.wait(2)

        trunc_txt = Text(
            "Aproximação linear (primeira ordem)",
            font_size=26,
            color=YELLOW
        ).next_to(eq_taylor_full, DOWN)

        self.play(FadeIn(trunc_txt))
        self.wait(2)

        self.play(FadeOut(eq_taylor_full), FadeOut(trunc_txt))

        # ======================================================
        # FORMA MATRICIAL FINAL
        # ======================================================
        eq_final = MathTex(
            r"\dot{\bar{x}} = A\bar{x} + B\bar{u}"
        ).scale(1.4)

        defs = VGroup(
            MathTex(r"A = \frac{\partial f}{\partial x}\Big|_\star"),
            MathTex(r"B = \frac{\partial f}{\partial u}\Big|_\star")
        ).arrange(DOWN, buff=0.4).next_to(eq_final, DOWN)

        self.play(Write(eq_final))
        self.play(LaggedStart(*[Write(d) for d in defs], lag_ratio=0.4))
        self.wait(3)

        self.play(FadeOut(eq_final), FadeOut(defs))

                # ======================================================
        # TRANSFORMADA DE LAPLACE (VERSÃO EXPANDIDA)
        # ======================================================
        s4 = Text("Transformada de Laplace", font_size=34, color=BLUE).to_edge(UP)
        self.play(ReplacementTransform(s3, s4))

        # Definição
        eq_lap = MathTex(
            r"\mathcal{L}\{f(t)\} = F(s) = \int_0^\infty f(t)e^{-st}dt"
        ).scale(1.2)

        self.play(Write(eq_lap))
        self.wait(2)

        # ------------------------------------------------------
        # TRANSFORMAÇÃO LINEAR
        # ------------------------------------------------------
        lin_title = Text(
            "Laplace é uma transformação linear",
            font_size=28,
            color=YELLOW
        ).next_to(eq_lap, DOWN, buff=0.6)

        lin_prop = MathTex(
            r"\mathcal{L}\{a f(t) + b g(t)\} = aF(s) + bG(s)"
        ).next_to(lin_title, DOWN)

        self.play(FadeIn(lin_title))
        self.play(Write(lin_prop))
        self.wait(2)

        self.play(FadeOut(lin_title), FadeOut(lin_prop))

        # ------------------------------------------------------
        # TEOREMA DA DIFERENCIAÇÃO
        # ------------------------------------------------------
        diff_title = Text(
            "Teorema da Diferenciação no Tempo",
            font_size=28,
            color=GREEN
        ).next_to(eq_lap, DOWN, buff=0.6)

        diff_rule = MathTex(
            r"\mathcal{L}\{\dot{f}(t)\} = sF(s) - f(0)"
        ).next_to(diff_title, DOWN)

        self.play(FadeIn(diff_title))
        self.play(Write(diff_rule))
        self.wait(2)

        diff_comment = Text(
            "Derivadas viram multiplicação por s",
            font_size=24
        ).next_to(diff_rule, DOWN)

        self.play(FadeIn(diff_comment))
        self.wait(2)

        self.play(FadeOut(diff_title), FadeOut(diff_rule), FadeOut(diff_comment))

        # ------------------------------------------------------
        # TEOREMA DA INTEGRAÇÃO
        # ------------------------------------------------------
        int_title = Text(
            "Teorema da Integração no Tempo",
            font_size=28,
            color=GREEN
        ).next_to(eq_lap, DOWN, buff=0.6)

        int_rule = MathTex(
            r"\mathcal{L}\left\{\int_0^t f(\tau)d\tau\right\} = \frac{F(s)}{s}"
        ).next_to(int_title, DOWN)

        self.play(FadeIn(int_title))
        self.play(Write(int_rule))
        self.wait(2)

        int_comment = Text(
            "Integrações viram divisão por s",
            font_size=24
        ).next_to(int_rule, DOWN)

        self.play(FadeIn(int_comment))
        self.wait(2)

        self.play(FadeOut(int_title), FadeOut(int_rule), FadeOut(int_comment))

        # ------------------------------------------------------
        # EXEMPLOS DE SINAIS EXÓGENOS (UM POR VEZ)
        # ------------------------------------------------------
        sig_title = Text(
            "Exemplos de sinais exógenos",
            font_size=30,
            color=YELLOW
        ).next_to(eq_lap, DOWN, buff=0.6)

        self.play(FadeIn(sig_title))
        self.wait(1)

        sinais = [
            MathTex(
                r"u(t) = 1 \quad \Rightarrow \quad U(s) = \frac{1}{s}"
            ),
            MathTex(
                r"u(t) = t \quad \Rightarrow \quad U(s) = \frac{1}{s^2}"
            ),
            MathTex(
                r"u(t) = e^{-a t} \quad \Rightarrow \quad U(s) = \frac{1}{s+a}"
            ),
            MathTex(
                r"u(t) = \sin(\omega t) \quad \Rightarrow \quad U(s) = \frac{\omega}{s^2+\omega^2}"
            )
        ]

        comentarios = [
            Text("Entrada degrau", font_size=26),
            Text("Entrada rampa", font_size=26),
            Text("Entrada exponencial decrescente", font_size=26),
            Text("Entrada senoidal", font_size=26)
        ]

        for sinal, comentario in zip(sinais, comentarios):
            sinal.scale(1.2).next_to(sig_title, DOWN, buff=0.6)
            comentario.next_to(sinal, DOWN, buff=0.4)

            self.play(Write(sinal))
            self.play(FadeIn(comentario))
            self.wait(2)

            self.play(FadeOut(sinal), FadeOut(comentario))

        self.play(FadeOut(sig_title), FadeOut(eq_lap))

        # ------------------------------------------------------
        # RESOLUÇÃO DE UMA EDO COM LAPLACE
        # ------------------------------------------------------
        # ------------------------------------------------------
        # RESOLUÇÃO DE UMA EDO COM LAPLACE (PASSO A PASSO)
        # ------------------------------------------------------
        edo_title = Text(
            "Resolvendo uma EDO com Laplace",
            font_size=30,
            color=BLUE
        ).next_to(s4, DOWN, buff=0.6)

        self.play(FadeIn(edo_title))
        self.wait(1)

        # Passo 1 — EDO no tempo
        edo = MathTex(
            r"\dot{y}(t) + a y(t) = u(t), \quad y(0)=0"
        ).scale(1.2).next_to(edo_title, DOWN, buff=0.8)

        edo_txt = Text(
            "Sistema linear no domínio do tempo",
            font_size=26
        ).next_to(edo, DOWN, buff=0.4)

        self.play(Write(edo))
        self.play(FadeIn(edo_txt))
        self.wait(2)

        self.play(FadeOut(edo), FadeOut(edo_txt))

        # Passo 2 — Aplicando Laplace
        lap_step1 = MathTex(
            r"\mathcal{L}\{\dot{y}(t)\} + a\mathcal{L}\{y(t)\} = \mathcal{L}\{u(t)\}"
        ).scale(1.1).next_to(edo_title, DOWN, buff=0.8)

        lap_txt1 = Text(
            "Aplicando a transformada de Laplace",
            font_size=26
        ).next_to(lap_step1, DOWN, buff=0.4)

        self.play(Write(lap_step1))
        self.play(FadeIn(lap_txt1))
        self.wait(2)

        self.play(FadeOut(lap_step1), FadeOut(lap_txt1))

        # Passo 3 — Usando o teorema da diferenciação
        lap_step2 = MathTex(
            r"sY(s) - y(0) + aY(s) = U(s)"
        ).scale(1.2).next_to(edo_title, DOWN, buff=0.8)

        lap_txt2 = Text(
            "Teorema da diferenciação no tempo",
            font_size=26,
            color=YELLOW
        ).next_to(lap_step2, DOWN, buff=0.4)

        self.play(Write(lap_step2))
        self.play(FadeIn(lap_txt2))
        self.wait(2)

        self.play(FadeOut(lap_step2), FadeOut(lap_txt2))

        # Passo 4 — Condição inicial
        lap_step3 = MathTex(
            r"sY(s) + aY(s) = U(s)"
        ).scale(1.2).next_to(edo_title, DOWN, buff=0.8)

        lap_txt3 = Text(
            "Como y(0)=0, o termo inicial desaparece",
            font_size=26
        ).next_to(lap_step3, DOWN, buff=0.4)

        self.play(Write(lap_step3))
        self.play(FadeIn(lap_txt3))
        self.wait(2)

        self.play(FadeOut(lap_step3), FadeOut(lap_txt3))

        # Passo 5 — Isolando Y(s)
        lap_step4 = MathTex(
            r"Y(s) = \frac{1}{s+a} \, U(s)"
        ).scale(1.4).next_to(edo_title, DOWN, buff=0.8)

        lap_txt4 = Text(
            "EDO vira uma equação algébrica",
            font_size=26,
            color=GREEN
        ).next_to(lap_step4, DOWN, buff=0.4)

        self.play(Write(lap_step4))
        self.play(FadeIn(lap_txt4))
        self.wait(3)

        self.play(
            FadeOut(lap_step4),
            FadeOut(lap_txt4),
            FadeOut(edo_title)
        )

                # ------------------------------------------------------
        # Passo 6 — Aplicando a Transformada Inversa de Laplace
        # ------------------------------------------------------
        lap_inv_title = Text(
            "Aplicando a Laplace Inversa",
            font_size=28,
            color=BLUE
        ).next_to(edo_title, DOWN, buff=0.8)

        self.play(FadeIn(lap_inv_title))
        self.wait(1)

        # Expressão no domínio de s
        lap_inv_eq = MathTex(
            r"Y(s) = \frac{1}{s+a}\,U(s)"
        ).scale(1.2).next_to(lap_inv_title, DOWN, buff=0.4)

        self.play(Write(lap_inv_eq))
        self.wait(2)

        # Caso particular: entrada degrau
        step_input = MathTex(
            r"u(t) = 1 \;\;\Rightarrow\;\; U(s) = \frac{1}{s}"
        ).scale(1.1).next_to(lap_inv_eq, DOWN, buff=0.4)

        self.play(Write(step_input))
        self.wait(2)

        # Substituição
        lap_subs = MathTex(
            r"Y(s) = \frac{1}{s(s+a)}"
        ).scale(1.3).next_to(step_input, DOWN, buff=0.4)

        self.play(Write(lap_subs))
        self.wait(2)

        self.play(
            FadeOut(lap_inv_eq),
            FadeOut(step_input),
            FadeOut(lap_subs)
        )

        # Resultado no tempo
        time_sol = MathTex(
            r"y(t) = 1 - e^{-at}"
        ).scale(1.5).next_to(lap_inv_title, DOWN, buff=0.6)

        time_txt = Text(
            "Resposta temporal do sistema",
            font_size=26,
            color=GREEN
        ).next_to(time_sol, DOWN, buff=0.4)

        self.play(Write(time_sol))
        self.play(FadeIn(time_txt))
        self.wait(3)

        self.play(
            FadeOut(time_sol),
            FadeOut(time_txt),
            FadeOut(lap_inv_title),
            FadeOut(edo_title)
        )

        # ======================================================
        # ENCERRAMENTO
        # ======================================================
        end = Text(
            "Laplace simplifica análise de sistemas dinâmicos",
            font_size=32,
            color=YELLOW
        )

        self.play(FadeIn(end))
        self.wait(3)
