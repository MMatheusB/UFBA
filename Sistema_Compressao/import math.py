import math

def refracao_do_Vidro (N, theta) :
    t = 4 * 1e-3 #transformando de mm para m
    lamb = 633 * 1e-9 #transformando de nm para m
    ang = math.radians(theta - 0.6) # pegando o valor em radianos
    return (2*t - N*lamb)*(1 - math.cos(ang) + (((N**2)*(lamb)**2)/4*t))/(2*t * (1 - math.cos(ang)) - N * lamb)

def desvio_relativo (valor_exp, valor_teo):
    desvio = ((valor_exp - valor_teo)/valor_teo) * 1e2
    return abs(desvio)


print(refracao_do_Vidro(50, 8.6))
print(desvio_relativo(1.58, 1.52))
