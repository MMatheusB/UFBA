from manim import *

# -----------------------------
# Classe 1: Gráficos Estatísticos
# -----------------------------
class EstatisticasGIF(Scene):
    def construct(self):
        # Título
        titulo = Text("Gráficos Estatísticos", font_size=40).to_edge(UP)
        self.play(FadeIn(titulo))
        
        # Eixos
        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0, 10, 1],
            x_length=6,
            y_length=4
        ).to_edge(DOWN)
        self.play(Create(axes))
        
        # Gráfico 1: Linha crescente
        graph1 = axes.plot(lambda x: 0.5*x + 1, color=BLUE)
        self.play(Create(graph1))
        self.wait(1)
        
        # Gráfico 2: Curva quadrática
        graph2 = axes.plot(lambda x: 0.2*x**2, color=YELLOW)
        self.play(Create(graph2))
        self.wait(1)
        
        # Gráfico 3: Senoidal
        graph3 = axes.plot(lambda x: 3*np.sin(0.5*x)+5, color=RED)
        self.play(Create(graph3))
        self.wait(2)
        
        # Fade out final
        self.play(FadeOut(titulo), FadeOut(axes), FadeOut(graph1), FadeOut(graph2), FadeOut(graph3))


# -----------------------------
# Classe 2: EDO e EDP
# -----------------------------
class EDO_EDP_GIF(Scene):
    def construct(self):
        # Título
        titulo = Text("EDO e EDP", font_size=40).to_edge(UP)
        self.play(FadeIn(titulo))
        
        # EDO simples
        edo = MathTex(r"\frac{dy}{dt} = -ky", font_size=50).to_edge(LEFT).shift(UP*0.5)
        self.play(Write(edo))
        
        # EDP simples
        edp = MathTex(r"\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}", font_size=50)
        edp.next_to(edo, RIGHT, buff=2)
        self.play(Write(edp))
        
        # Espera para visualização
        self.wait(3)
        
        # Fade out
        self.play(FadeOut(titulo), FadeOut(edo), FadeOut(edp))
