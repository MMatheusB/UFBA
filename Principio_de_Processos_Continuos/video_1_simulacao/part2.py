from manim import *
import numpy as np
from scipy.integrate import solve_ivp

class TanqueEnchendo(Scene):
    def construct(self):
        # --- Frases iniciais formatadas ---
        frase1 = Paragraph(
            "Imagine um tanque, teremos a vazão de saída e de entrada,",
            "como foi mostrado na EDO",
            alignment="center",
            line_spacing=0.4
        ).scale(0.6)
        
        frase2 = Paragraph(
            "Dependendo da condição inicial e dos valores das entradas do sistema,",
            "teremos diferentes comportamentos",
            alignment="center",
            line_spacing=0.4
        ).scale(0.6)
        
        frase3 = VGroup(
            Paragraph("Caso Q_in > Q_out", alignment="center").scale(0.6),
            Paragraph(
                "Naturalmente, estaria entrando mais água no tanque do que saindo,",
                "por isso o nível iria aumentar",
                alignment="center",
                line_spacing=0.4
            ).next_to(Paragraph("Caso Q_in > Q_out"), DOWN*0.5)
        )

        # --- Exibir frases sequenciais ---
        self.play(Write(frase1))
        self.wait(3)
        self.play(FadeOut(frase1))
        
        self.play(Write(frase2))
        self.wait(3)
        self.play(FadeOut(frase2))
        
        self.play(Write(frase3))
        self.wait(3)
        self.play(FadeOut(frase3))

        # --- Parâmetros da EDO ---
        Qin = 1.0
        k = 0.3
        h0 = 0.1
        t_span = [0, 8]
        t_eval = np.linspace(t_span[0], t_span[1], 200)

        def dhdt(t, h):
            return Qin - k*h

        sol = solve_ivp(dhdt, t_span, [h0], t_eval=t_eval)
        h_vals = sol.y[0]
        h_max = max(h_vals)

        # --- Tanque ---
        altura_tanque = 3
        tanque = Rectangle(width=2, height=altura_tanque, color=WHITE).shift(DOWN*0.5)
        self.play(DrawBorderThenFill(tanque))

        # --- Líquido inicial ---
        liquido = Rectangle(
            width=1.9,
            height=(h0/h_max)*altura_tanque,
            fill_color=BLUE,
            fill_opacity=0.6,
            stroke_width=0
        )
        liquido.move_to(tanque.get_bottom() + UP*liquido.height/2)
        self.add(liquido)

        # --- Label nível ---
        nivel_texto = MathTex(r"h(t)").next_to(tanque, UP)
        self.add(nivel_texto)

        # --- Animação do tanque enchendo com EDO ---
        t_tracker = ValueTracker(0)

        def update_liquido(mob):
            t = t_tracker.get_value()
            h = np.interp(t, t_eval, h_vals)
            altura_liquido = (h / h_max) * altura_tanque
            mob.become(
                Rectangle(width=1.9, height=altura_liquido, fill_color=BLUE, fill_opacity=0.6, stroke_width=0)
                .move_to(tanque.get_bottom() + UP*altura_liquido/2)
            )
        liquido.add_updater(update_liquido)

        self.play(t_tracker.animate.set_value(t_span[1]), run_time=6)
        self.wait(2)
