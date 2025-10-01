from manim import *


class LogoIntro(Scene):
    def construct(self):
        # --- Criar partes do nome ---
        math = Text("Math", font="Arial", color=BLUE)
        two = Text("2", font="Arial", color=WHITE)
        sim = Text("Sim", font="Arial", color=GREEN)

        # Agrupar e centralizar
        logo = VGroup(math, two, sim).arrange(RIGHT, buff=0.2)
        logo.move_to(ORIGIN)

        # Animação de surgimento
        self.play(Write(math))
        self.wait(0.5)
        self.play(FadeIn(two, shift=UP))
        self.wait(0.5)
        self.play(Write(sim))
        self.wait(1)

        # Efeito final
        self.play(logo.animate.scale(1.1), rate_func=there_and_back, run_time=1)
        self.wait(1)