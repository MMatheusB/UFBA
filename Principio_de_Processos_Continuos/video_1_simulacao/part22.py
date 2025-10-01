from manim import *

class ModelagemFinal(Scene):
    def construct(self):
        # Cena 1 – Precisão do modelo
        text1 = Text("Qual a precisão do modelo?", font_size=40).shift(UP * 1.5)
        bullets1 = VGroup(
            Text("• Nível de detalhamento × benefícios", font_size=30),
            Text("• Aceitável para uma aplicação, inadequado para outra", font_size=30),
            Text("• Avaliação contínua: ajustado ou descartado", font_size=30)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(text1, DOWN)

        self.play(Write(text1))
        for item in bullets1:
            self.play(FadeIn(item, shift=RIGHT))
            self.wait(3.3)
        self.wait(0.5)
        self.play(FadeOut(text1), *[FadeOut(item) for item in bullets1])

        # Cena 2 – Importância da análise dinâmica
        text2 = Text("Análise dinâmica", font_size=36).shift(UP * 1.5)
        bullets2 = VGroup(
            Text("• Simulação temporal do sistema", font_size=28),
            Text("• Integração numérica das equações de balanço", font_size=28),
            Text("• Avaliação da estabilidade", font_size=28),
            Text("• Resposta a diferentes estímulos", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).next_to(text2, DOWN)

        self.play(Write(text2))
        for item in bullets2:
            self.play(FadeIn(item, shift=RIGHT))
            self.wait(2)
        self.wait(0.5)
        self.play(FadeOut(text2), *[FadeOut(item) for item in bullets2])

        # Cena 3 – Formas de obtenção de modelos
        title3 = Text("Como obter um modelo matemático?", font_size=36).shift(UP * 1.5)
        col1 = VGroup(
            Text("Teórico:", font_size=30, color=YELLOW),
            Text("- Leis de Newton, Kirchhoff", font_size=26),
            Text("- Conservação de massa, energia", font_size=26),
            Text("- Relações constitutivas", font_size=26)
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.9)

        col2 = VGroup(
            Text("Empírico:", font_size=30, color=YELLOW),
            Text("- Dados experimentais", font_size=26),
            Text("- Ajuste de curvas", font_size=26),
            Text("- Métodos estatísticos", font_size=26)
        ).arrange(DOWN, aligned_edge=LEFT).scale(0.9)

        group3 = VGroup(col1, col2).arrange(RIGHT, buff=1).next_to(title3, DOWN)

        self.play(Write(title3))
        self.wait(1)
        for item in col1:
            self.play(FadeIn(item, shift=LEFT))
            self.wait(2)
        self.wait(1)
        for item in col2:
            self.play(FadeIn(item, shift=RIGHT))
            self.wait(1)
        self.wait(2)
        self.play(FadeOut(title3), FadeOut(group3))

        # Cena 4 – Classificação dos modelos
        title4 = Text("Classificação dos modelos", font_size=36).shift(UP * 1.5)
        bullets4 = VGroup(
            Text("• Estático × Dinâmico", font_size=28),
            Text("• Linear × Não linear", font_size=28),
            Text("• Determinístico × Estocástico", font_size=28),
            Text("• Contínuo no tempo × Discreto no tempo", font_size=28),
            Text("• MISO × MIMO", font_size=28),
            Text("• Parâmetros concentrados × distribuídos", font_size=28)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).next_to(title4, DOWN)

        self.play(Write(title4))
        for item in bullets4:
            self.play(FadeIn(item, shift=RIGHT))
            self.wait(6)
        self.wait(2)
        self.play(FadeOut(title4), *[FadeOut(item) for item in bullets4])

        # Cena 5 – Encerramento com logo
        logo = Text("Math2Sim", gradient=(BLUE, GREEN)).scale(1.2)
        slogan = Text("Matemática e simulação visual", font_size=30).next_to(logo, DOWN)

        self.play(Write(logo))
        self.play(FadeIn(slogan, shift=UP))
        self.wait(2)
        self.play(logo.animate.scale(1.1), run_time=2, rate_func=there_and_back)
        self.wait(5)
