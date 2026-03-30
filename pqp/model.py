import numpy as np

# Define as constantes do modelo*
# *Os ajustes de escala são usados para tornar o modelo adimensional.

g = 9.81  # gravidade [m/s²]

# --- Parâmetros do petróleo liquido (Tabela 1) ---
β_T = 1 / 1.5e9  # constante de compressibilidade isotérmica [1/Pa]
p0_β = 1e5  # 1 bar em Pa, pressão de referência para a compressibilidade [Pa]
ρ_o = 900.0  # Densidade do óleo [kg/m³]
ρ_w = 1000.0  # Densidade da água [kg/m³]
χ_w = 0.35  # fração volumétrica de água (water cut)
ρ0 = χ_w * ρ_w + (1 - χ_w) * ρ_o  # densidade do fluido total [kg/m³]
χ_mw = 0.5  # fração volumétrica de água no manifold
ρ_m0 = χ_mw * ρ_w + (1 - χ_mw) * ρ_o  # densidade do fluido no manifold [kg/m³]
ν_o = 100e-6  #  viscosidade cinemática do óleo [m²/s]
ν_w = 1e-6  #  viscosidade cinemática da água [m²/s]

ν = χ_w * ν_w + (1 - χ_w) * ν_o  # viscosidade cinemática do fluido total [m²/s]
# μ = ρ0 * ν  # viscosidade dinâmica do fluido total [Pa.s]

# --- Parâmetros do tubo vertical (Tabela 2) ---
ℓ_plus = 2000.0  # comprimento tubo vertical (parte entre o ESP e o manifold) [m]
ℓ_minus = 100.0  # comprimento do tubo vertical (parte entre o reservatório e a ESP)[m]
h = ℓ_plus + ℓ_minus  # altura para o cálculo gravitacional
d = 0.1569  # diâmetro do tubo vertical [m]
A = np.pi * (d / 2) ** 2  # área do tubo vertical [m²]
ε = 45.7e-6  # altura da rugosidade do tubo vertical [m]
hςp = 1210.6  # ajuste de escala da cabeça da bomba [m]
fp0 = 60.0  # frequência de rotação nominal da ESP [Hz]
V_ς = 1.0  # ajuste de escala da vazão [m³/s]
a1, a2, a3 = -37.57, 2.864e3, -8.668e4  # coeficientes da bomba submersa [adimensionais]
mcv = 25.9e3 / 3600  # capacidade de vazão mássica da válvula [kg/s]
# f(uv)
p_ς = 1e5  # ajuste de escala da pressão [Pa]
ρ_ς = 1000.0  # ajuste de escala da densidade (scaling density) [kg/m³]
Vpi = 7e-4  # capacidade de produção do poço (productivity index) [m³/s]

# --- Parâmetros do manifold + tubo horizontal (Tabela 3) ---
# parâmetros do manifold
l_m = 500.0  # comprimento do manifold [m]
d_m = d  # diâmetro do manifold [m]
Vm = (np.pi * (d_m / 2) ** 2) * l_m  # volume da manifold [m³]
# parâmetros do tubo horizontal de transporte do manifold até o separador.
ℓ_t = 4000.0  # comprimento do tubo horizontal [m]
d_t = d  # diâmetro do tubo horizontal [m]
A_t = np.pi * (d_t / 2) ** 2  # área do tubo horizontal [m²]
ε_t = ε  # altura da rugosidade do tubo horizontal [m]
# parâmetros da bomba horizontal
dp_ς_bp = 1e6  # ajuste de escala do aumento pressão dado uma frequência da bomba [Pa]
fbp0 = 60.0  # frequência de rotação nominal da bomba horizontal [Hz]


def pf(t):
    """Pressão do reservatório [Pa]"""
    return 220e5


def ps(t):
    """Pressão no separador [Pa]"""
    return 30e5


def fbp(t):
    """Frequência de rotação da bomba horizontal [Hz]"""
    return 60.0


# Equações Diferenciais Ordinárias do modelo


def EDOs(t, y, args):
    Vv, pm, Vt = y
    fp = args

    # Tubo vertical
    vv = Vv / A
    ρ_v = ρ0 * np.exp(β_T * (pm - p0_β))
    Re_v = ρ_v * vv * d / ν
    fD_v = 1 / (4 * (np.log10(5.74 / (Re_v**0.9) + ε / (3.7 * d))) ** 2)
    dpf = ℓ_plus * fD_v * ρ_v / 2 * vv**2 / d
    dpg = ρ_v * g * hςp

    hp = hςp * (
        (fp / fp0) ** 2
        + a1 * fp / fp0 * Vv / V_ς
        + a2 * (Vv / V_ς) ** 2
        + a3 * fp0 / fp * (Vv / V_ς) ** 3
    )
    dpp = ρ_v * g * hp

    ph = pf(t) - p_ς * Vv / Vpi
    pci = pm + (p_ς / ρ_v) * (ρ_v / ρ_ς) * (ρ_v * Vv / mcv) ** 2
    dVv_dt = (ph - pci + dpp - dpf - dpg) / (ρ_v * ℓ_plus / A)

    # Manifold
    ρ_m = ρ_m0 * np.exp(β_T * (pm - p0_β))
    Vw = (χ_mw - χ_w) / (1 - χ_mw) * Vv
    m_in = ρ_v * Vv + ρ_w * Vw
    m_out = ρ_m * Vt
    dpm_dt = (m_in - m_out) / (ρ_m * Vm * β_T)

    # Tubo horizontal
    vt = Vt / A_t
    ρ_t = ρ_m0 * np.exp(β_T * (pm - p0_β))
    Re_t = ρ_t * vt * d_t / ν
    fD_t = 1 / (4 * (np.log10(5.74 / (Re_t**0.9) + ε_t / (3.7 * d_t))) ** 2)
    dpft = ℓ_t * fD_t * ρ_t / 2 * vt**2 / d_t
    dpp_boost = dp_ς_bp * (fbp(t) / fbp0) ** 2
    dVt_dt = (pm - ps(t) + dpp_boost - dpft) / (ρ_t * ℓ_t / A_t)

    return [dVv_dt, dpm_dt, dVt_dt]

# --- Funções novas: simulação por trajetória e geração de dataset ---
def _simulate_fp_trajectory(fp_time, t_span, n_points, y0):
    """
    Integra EDOs para uma trajetória de fp(t) definida em fp_time (array length n_points).
    Retorna (t_grid, sol.y) onde sol.y tem shape (3, n_points).
    """
    from scipy.integrate import solve_ivp
    t_grid = np.linspace(t_span[0], t_span[1], n_points)

    def fun(t, y):
        fp = float(np.interp(t, t_grid, fp_time))
        return EDOs(t, y, fp)

    sol = solve_ivp(fun, t_span, y0, t_eval=t_grid, method="LSODA")
    if not sol.success:
        raise RuntimeError("Integração falhou: " + (sol.message or "unknown"))
    return t_grid, sol.y


def generate_dataset(
    num_trajectories=1000,
    t_span=(0.0, 200.0),
    n_points=2000,
    save_path="dataset.npz",
    y0=None,
    fp_bounds=(40.0, 80.0),
    max_segments=10,
    seed=None,
    train_ratio=0.7,
):
    """
    Gera um banco de dados dinâmico para treino/validação de rede neural.

    Principais alterações:
    - num_trajectories: padrão maior para "simulação bem grande".
    - n_points: pontos por trajetória (maior por padrão).
    - max_segments: mais segmentos por trajetória para maior variabilidade.
    - train_ratio: fração de exemplos destinados ao treino (ex.: 0.7).
    - Salva em save_path com chaves: t, inputs_train, outputs_train, inputs_val, outputs_val, y0, meta.
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)

    if y0 is None:
        y0 = [
            0.03379577221408489,  # Vv0 [m3/s]
            2674866.220840107,    # pm0 [Pa]
            0.033795772214084814, # Vt0 [m3/s]
        ]

    # alocar arrays
    inputs = _np.zeros((num_trajectories, n_points), dtype=_np.float32)
    outputs = _np.zeros((num_trajectories, 3, n_points), dtype=_np.float32)

    for i in range(num_trajectories):
        # número de segmentos aleatórios por trajetória (pelo menos 1)
        n_seg = int(rng.integers(1, max_segments + 1))
        if n_seg == 1:
            indices = [0, n_points]
        else:
            cuts = rng.choice(_np.arange(1, n_points - 1), n_seg - 1, replace=False)
            cuts.sort()
            indices = _np.concatenate(([0], cuts, [n_points]))
        fp_traj = _np.empty(n_points, dtype=_np.float32)
        for s in range(len(indices) - 1):
            val = float(rng.uniform(fp_bounds[0], fp_bounds[1]))
            fp_traj[indices[s] : indices[s + 1]] = val

        try:
            t_grid, sol_y = _simulate_fp_trajectory(fp_traj, t_span, n_points, y0)
        except Exception:
            # em caso de falha, tentar pequena suavização e re-integrar
            fp_traj = _np.clip(fp_traj + rng.normal(0, 0.5, size=n_points), fp_bounds[0], fp_bounds[1])
            t_grid, sol_y = _simulate_fp_trajectory(fp_traj, t_span, n_points, y0)

        inputs[i, :] = fp_traj
        outputs[i, :, :] = sol_y.astype(_np.float32)

    # embaralhar e dividir em treino / validação
    perm = rng.permutation(num_trajectories)
    n_train = int(_np.round(train_ratio * num_trajectories))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    inputs_train = inputs[train_idx]
    outputs_train = outputs[train_idx]
    inputs_val = inputs[val_idx]
    outputs_val = outputs[val_idx]

    meta = {
        "num_trajectories": int(num_trajectories),
        "n_points": int(n_points),
        "train_ratio": float(train_ratio),
        "seed": None if seed is None else int(seed),
    }

    _np.savez_compressed(
        save_path,
        t=t_grid,
        inputs_train=inputs_train,
        outputs_train=outputs_train,
        inputs_val=inputs_val,
        outputs_val=outputs_val,
        y0=_np.array(y0, dtype=_np.float32),
        meta=meta,
    )

    return {
        "path": save_path,
        "meta": meta,
        "n_train": inputs_train.shape[0],
        "n_val": inputs_val.shape[0],
    }
