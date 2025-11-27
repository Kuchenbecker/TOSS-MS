import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

# Argumentos de entrada
parser = argparse.ArgumentParser(description="Plota espectros de massas em 2D ou 3D a partir de um CSV.")
parser.add_argument("--csv", required=True, help="Caminho para o arquivo CSV gerado pelo extrair_ms2.py")
parser.add_argument("--out", help="Caminho do arquivo de saída (.png ou .svg). Se omitido, mostra na tela.")
parser.add_argument("--3d", action="store_true", help="Ativa modo de plotagem 3D com barras")
parser.add_argument("--labemin", type=float, default=float("inf"), help="Intensidade mínima para rotular picos (em % ou unidades absolutas)")
parser.add_argument("--clean", type=float, default=0.0, help="Intensidade mínima para exibir no gráfico")
args = parser.parse_args()

# Carregar CSV
df = pd.read_csv(args.csv)

# Verificação de colunas obrigatórias
required_cols = {"collision_energy", "mz", "intensity"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"O CSV deve conter as colunas: {required_cols}")

# Aplicar filtro clean
df = df[df["intensity"] >= args.clean]

# PLOTAGEM 3D
if args.__dict__["3d"]:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Define largura das barras
    dx = dy = 0.2

    # Plotar barras 3D
    for _, row in df.iterrows():
        x, y, z = row["mz"], row["collision_energy"], row["intensity"]
        ax.bar3d(x, y, 0, dx, dy, z, shade=True)

        # Rótulo se acima de labemin
        if z >= args.labemin:
            ax.text(x, y, z, f"{x:.1f}", fontsize=7, ha="center", va="bottom")

    ax.set_xlabel("m/z")
    ax.set_ylabel("Collision Energy (eV)")
    ax.set_zlabel("Intensidade")
    ax.set_title("Espectro de massas 3D por CE")
else:
    # PLOTAGEM 2D
    fig, ax = plt.subplots(figsize=(12, 6))

    for ce, group in df.groupby("collision_energy"):
        ax.plot(group["mz"], group["intensity"], label=f"CE {ce:.1f}")
        # Rótulo dos picos
        for _, row in group.iterrows():
            if row["intensity"] >= args.labemin:
                ax.text(row["mz"], row["intensity"], f"{row['mz']:.1f}", fontsize=7, rotation=45, ha="left")

    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensidade")
    ax.set_title("Espectros sobrepostos por energia de colisão")
    ax.legend(title="Collision Energy", fontsize=8)

plt.tight_layout()

# Salvar ou mostrar
if args.out:
    ext = os.path.splitext(args.out)[1].lower()
    if ext not in [".png", ".svg"]:
        raise ValueError("Formato de saída inválido. Use .png ou .svg.")
    plt.savefig(args.out, dpi=300)
    print(f"✔ Gráfico salvo em: {args.out}")
else:
    plt.show()
