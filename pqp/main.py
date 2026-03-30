import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import model as m
import matplotlib as mpl
import argparse

mpl.rcParams["lines.linewidth"] = 2

# --- Condições iniciais ---
Vv0 = 0.03379577221408489  # vazão volumétrica no tubo vertical [m³/s]
pm0 = 2674866.220840107  # Pressão no manifold [Pa]
Vt0 = 0.033795772214084814  # vazão volumétrica no tubo horizontal [m³/s]
fp0 = 60.0  # Frequência de rotação da ESP [Hz]
y0 = [Vv0, pm0, Vt0, fp0]

t_span = (0, 200)
t = np.linspace(*t_span, 1000)

def model(t, y):
    Vv, pm, Vt = y
    fp = np.where(t > 100, 80, 60)
    dVv_dt, dpm_dt, dVt_dt = m.EDOs(t, [Vv, pm, Vt], fp)

    return [dVv_dt, dpm_dt, dVt_dt]

sol = solve_ivp(model, t_span, y0[:3], t_eval=t, method="LSODA")
Vv, pm, Vt = sol.y

print("Vv:", Vv[-1])
print("pm:", pm[-1])
print("Vt:", Vt[-1])

# # --- Plotagem das saídas do sistema---
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)
# plt.plot(t, Vv, label="Vazão no tubo horizontal ($\\dot{V}_v$)")
# plt.plot(t, Vt, label="Vazão no tubo vertical ($\\dot{V}_t$)")
# plt.ylabel("Vazão / m$^3\\cdot$s$^{-1}$")
# plt.legend()
#
# plt.subplot(2, 1, 2)
# plt.plot(t, pm, label="Pressão no Manifold ($p_m$) sem controlador")
# plt.ylabel("Pressão / Pa")
# plt.xlabel("Tempo / s")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # --- Gráficos com variável controlada e variável manipulada ---
# _, axs = plt.subplots(2, figsize=(12, 5), layout="constrained")
#
# axs[0].plot(t, pm / 1e5, label="$p_m$ sem controlador")
# axs[0].set_xlabel("Tempo / s")
# axs[0].set_ylabel("Pressão no manifold / bar")
# axs[0].legend()
# axs[0].grid()
#
# axs[1].plot(t, manual_fp(t), label="$f_p$(t) sem controlador")
# axs[1].set_xlabel("Tempo / s ")
# axs[1].set_ylabel("Frequência da ESP / Hz")
# axs[1].legend()
# axs[1].grid()
#
# plt.show()

# python
import matplotlib.ticker as ticker

# Converter vazões de m³/s para m³/h
Vv_h = Vv * 3600.0
Vt_h = Vt * 3600.0

pm_bar = pm/1e5
# --- Plotagem das saídas do sistema (m³/h) ---
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(t, Vt_h, label="Vazão no tubo vertical (m³/h) ")
plt.ylabel("Vazão / (m$^3\\cdot$h$^{-1})$")
plt.legend()

ax2 = plt.subplot(2, 1, 2)
ax2.plot(t, pm_bar, label="Pressão no Manifold (Pa)")
ax2.set_ylabel("Pressão / Pa")
ax2.set_xlabel("Tempo / s")
ax2.legend()

# Desativa a notação científica / offset no eixo y (pressão)
ax2.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
ax2.yaxis.get_major_formatter().set_scientific(False)

plt.tight_layout()
plt.show()

print(Vv[-1], pm[-1], Vt[-1], fp0)

# --- CLI: geração de dataset (opcional) ---
def _print_generate_instructions():
    print("Como gerar o dataset:")
    print("1) Via terminal (exemplo):")
    print("   python main.py --gen --n 2000 --npoints 2000 --out dataset.npz --train-ratio 0.7 --seed 0")
    print("")
    print("2) Programaticamente (exemplo em Python):")
    print("   import model as m")
    print("   res = m.generate_dataset(num_trajectories=2000, n_points=2000, save_path='dataset.npz', seed=0, train_ratio=0.7)")
    print("   print('Dataset salvo em:', res['path'])")
    print("")
    print("Parâmetros importantes:")
    print("  --n       : número de trajetórias")
    print("  --npoints : pontos por trajetória")
    print("  --out     : arquivo de saída (.npz)")
    print("  --train-ratio : fração treino (ex.: 0.7)")
    print("  --seed    : semente para reprodutibilidade")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulação e utilitários do modelo.")
    parser.add_argument("--gen", action="store_true", help="(Ignorado) Mantido por compatibilidade; dataset é gerado automaticamente ao rodar o script.")
    parser.add_argument("--show-gen", action="store_true", help="Mostrar instruções rápidas para gerar o dataset.")
    parser.add_argument("--n", type=int, default=1000, help="Número de trajetórias a gerar (quando gerar dataset).")
    parser.add_argument("--npoints", type=int, default=2000, help="Pontos por trajetória (quando gerar dataset).")
    parser.add_argument("--out", type=str, default="dataset.npz", help="Caminho do arquivo de saída .npz.")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Fração de exemplos para treino (0..1).")
    parser.add_argument("--seed", type=int, default=0, help="Semente para reprodutibilidade.")
    args = parser.parse_args()

    if args.show_gen:
        _print_generate_instructions()
        exit(0)

    # Gera dataset automaticamente ao rodar main.py (usa parâmetros da CLI)
    print("Gerando dataset automaticamente com", args.n, "trajetórias,", args.npoints, "pontos cada ->", args.out)
    res = m.generate_dataset(
        num_trajectories=args.n,
        n_points=args.npoints,
        save_path=args.out,
        seed=args.seed,
        train_ratio=args.train_ratio,
    )
    print("Dataset salvo em:", res["path"], "| treino:", res["n_train"], "val:", res["n_val"])
