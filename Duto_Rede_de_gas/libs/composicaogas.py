from casadi import *
from numpy import exp, log, array, roots, isnan, fromstring
from scipy.optimize import fsolve
from libs.species_builder import Species, Mixture
from libs.eos_database import *
from libs.gc_eos_soave import *
from builtins import sum

def sub_vis_chung(T, V, gas):
    V = V  


    Tc = np.array([sp.Tc for sp in gas.mixture.list_of_species])
    Vc = np.array([sp.Vc for sp in gas.mixture.list_of_species])
    w = np.array([sp.omega for sp in gas.mixture.list_of_species])       
    PM = np.sum([species.x*species.MM for species in gas.mixture.list_of_species])
    dip = np.array([0 for sp in gas.mixture.list_of_species])
    x = np.array([sp.x for sp in gas.mixture.list_of_species])  

    # Valores especificos do método de Chung
    ai = np.array([6.324, 1.210e-3, 5.283, 6.623, 19.745, -1.9, 24.275, 0.7972, -0.2382, 0.06863])
    bi = np.array([50.412, -1.154e-3, 254.209, 38.096, 7.630, -12.537, 3.45, 1.117, 0.0677, 0.3479])
    ci = np.array([-51.68, -6.257e-3, -168.48, -8.464, -14.354, 4.985, -11.291, 0.01235, -0.8163, 0.5926])


    eps_k = Tc / 1.2593
    sigma = 0.809 * Vc**(1/3)

    sigma_ij = np.sqrt(np.outer(sigma, sigma))
    eps_k_ij = np.sqrt(np.outer(eps_k, eps_k))
    w_ij = (np.add.outer(w, w)) / 2
    PM_ij = 2 * np.outer(PM, PM) / (np.add.outer(PM, PM))

    sigma_m3 = x @ (sigma_ij**3) @ x
    sigma_m = sigma_m3**(1/3)

    eps_k_m = x @ (eps_k_ij * sigma_ij**3) @ x / sigma_m**3
    Tea_m = T / eps_k_m

    PM_m = (x @ (eps_k_ij * sigma_ij**2 * np.sqrt(PM_ij)) @ x / (eps_k_m * sigma_m**2))**2
    w_m = x @ (w_ij * sigma_ij**3) @ x / sigma_m**3

    dip_vec = dip * x
    dip_m = (sigma_m**3 * dip_vec @ (sigma_ij**-3) @ dip_vec)**0.25

    Omega_m = (
        1.16145 * Tea_m**-0.14874 +
        0.52487 * np.exp(-0.77320 * Tea_m) +
        2.16178 * np.exp(-2.43787 * Tea_m)
    )

    Tc_m = 1.2593 * eps_k_m
    Vc_m = (sigma_m / 0.809)**3
    dip_rm = 131.3 * dip_m / np.sqrt(Vc_m * Tc_m)

    Fc_m = 1 - 0.2756 * w_m + 0.059035 * dip_rm**4
    y_m = Vc_m / V / 6
    G1_m = (1 - 0.5 * y_m) / (1 - y_m)**3

    E = ai + bi * w_m + ci * dip_rm**4 #simplificacao da conta da matriz

    E1, E2, E3, E4, E5, E6, E7, E8, E9, E10 = E

    G2_m = (
        E1 * (1 - np.exp(-E4 * y_m)) / y_m +
        E2 * G1_m * np.exp(E5 * y_m) +
        E3 * G1_m
    ) / (E1 * E4 + E2 + E3)

    eta_aa_m = E7 * y_m**2 * G2_m * np.exp(E8 + E9 / Tea_m + E10**2 / Tea_m**2)

    eta_a_m = np.sqrt(Tea_m) / Omega_m * (Fc_m * (1/G2_m + E6 * y_m)) + eta_aa_m

    eta_m = 36.344 * np.sqrt(PM_m * Tc_m) / Vc_m**(2/3) * eta_a_m * 1e-7

    return eta_m, Tc_m, Vc_m, PM_m, y_m, G1_m, dip_rm, w_m, dip_m

#mixture.tcm e mixture.Vcm
def coef_con_ter(gas):
    
    gas.ci_real()

    Cvt = gas.Cvt 

    eta_m, Tc_m, Vc_m, PM_m, y_m, G1_m, dip_rm, w_m, dip_m = sub_vis_chung(gas.T, gas.V, gas)

    Tr = gas.T / Tc_m

    q = 3.586e-3 * (Tc_m / PM_m)**0.5 / (Vc_m**(2/3))

    # Coeficientes Chung
    abcd = np.array([
        [2.4166,   0.74824,  -0.91858, 121.72],
        [-0.50924, -1.5094, -49.991,   69.983],
        [6.6107,   5.6207,   64.76,    27.039],
        [14.543,  -8.9139,  -5.6379,   74.344],
        [0.79274,  0.82019, -0.69369,   6.3173],
        [-5.8634, 12.801,    9.5893,   65.529],
        [91.089, 128.11,   -54.217,   523.81]
    ])

    B = [abcd[i, 0] + abcd[i, 1] * w_m + abcd[i, 2] * dip_rm**4 for i in range(7)]

    G2_m = (
        B[0]/y_m * (1 - np.exp(-B[3]*y_m)) +
        B[1]*G1_m*np.exp(B[4]*y_m) +
        B[2]*G1_m
    ) / (B[0]*B[3] + B[1] + B[2])


    alpha = Cvt/R - 1.5
    beta = 0.7862 - 0.7109*w_m + 1.3168*w_m**2
    zeta = 2 + 10.5*Tr**2

    Phi = 1 + alpha * (0.215 + 0.28288*alpha + 1.061*beta + 0.26665*zeta) / \
          (0.6366 + beta*zeta + 1.061*alpha*beta)

    
    kappa = (
        31.2 * eta_m * Phi / (PM_m * 1e-3) * (1/G2_m + B[5]*y_m) +
        q * B[6] * y_m**2 * Tr**0.5 * G2_m
    )

    return kappa