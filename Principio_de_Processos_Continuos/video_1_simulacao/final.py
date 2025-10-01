from manim import *

class final(Scene):
    def construct(self):    
        logo = Text("Math2Sim", gradient=(BLUE, GREEN)).scale(1.2)
        slogan = Text("Matemática e simulação visual", font_size=30).next_to(logo, DOWN)

        self.play(Write(logo))
        self.play(FadeIn(slogan, shift=UP))
        self.wait(2)
        self.play(logo.animate.scale(1.1), run_time=2, rate_func=there_and_back)
        self.wait(5)