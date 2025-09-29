from manim import *

class IntroCena(Scene):
    def construct(self):
        # --- Cena 1: Introdução com cubo girando ---
        titulo1 = Text("O que é um modelo?", font_size=40)
        cubo = Cube(side_length=1).set_color(BLUE).scale(0.5).next_to(titulo1, DOWN, buff=0.5)
        
        self.play(FadeIn(titulo1))
        self.play(FadeIn(cubo))
        self.play(Rotate(cubo, angle=TAU, axis=UP), run_time=2)
        self.wait(1.5)
        self.play(FadeOut(titulo1), FadeOut(cubo))
        
        
        # --- Cena 2: Tipos de modelos ---
        titulo2 = Text("Modelos podem ser classificados em dois grupos principais:", font_size=32)
        fisico = Paragraph(
            "• Modelos físicos: protótipos ou réplicas físicas",
            alignment="center"
        ).scale(0.55)
        matematico = Paragraph(
            "• Modelos matemáticos: equações que descrevem sistemas",
            alignment="center"
        ).scale(0.55)
        
        fisico.next_to(titulo2, DOWN, buff=0.3)
        matematico.next_to(fisico, DOWN, buff=0.2)
        
        # Exibir texto
        self.play(Write(titulo2))
        self.wait(0.5)
        self.play(FadeIn(fisico, shift=UP*0.2))
        self.wait(4)
        self.play(FadeIn(matematico, shift=UP*0.2))
        self.wait(1)
        
        # --- Gráfico posicionado abaixo do texto "matemático" ---
# --- Gráfico posicionado abaixo do texto "matemático" ---
        axes = Axes(
         x_range=[0, 3, 1],
        y_range=[0, 3, 0.5],  # menor limite no eixo Y
        x_length=3,           # eixo menor
        y_length=1.5
    )
# Posicionar logo abaixo do texto "matemático"
        axes.next_to(matematico, DOWN, buff=0.4)

# Linha do gráfico mais “achatada” para caber
        graph = axes.plot(lambda x: 0.15*x**2, x_range=[0,5], color=YELLOW)

        self.play(FadeIn(axes))
        self.play(DrawBorderThenFill(graph))
        self.wait(3) # coeficiente menor
        
        # Limpar a cena
        self.play(FadeOut(titulo2), FadeOut(fisico), FadeOut(matematico))
        self.play(FadeOut(axes), FadeOut(graph))
        
        # --- Cena 3: Foco do vídeo ---
        foco = Paragraph(
            "Neste vídeo, focaremos nos modelos matemáticos,",
            "mostrando como eles podem ser usados para simulação",
            "e análise de sistemas complexos.",
            alignment="center",
            line_spacing=0.35
        ).scale(0.65)
        self.play(Write(foco))
        self.wait(5)
        self.play(FadeOut(foco))
        
        # --- Cena 4: Citaçao de pensadores ---
        citacao = Paragraph(
            "“Um modelo não é a realidade; ele é uma forma de entender a realidade.” – Edward Tufte",
            alignment="center"
        ).scale(0.4)

        self.play(FadeIn(citacao, shift=UP*0.2))