import xml.etree.ElementTree as ET
import base64
import zlib
import numpy as np
import pandas as pd
import re
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Argumentos
parser = argparse.ArgumentParser(description="Extrai espectros médios por CE de arquivo mzML.")
parser.add_argument("--path", required=True, help="Caminho para o arquivo mzML")
parser.add_argument("--plot", action="store_true", help="Gera plot 3D (mz vs CE vs intensidade)")
args = parser.parse_args()

mzml_path = args.path

# Decodificador binário
def decode_binary_array(encoded_text):
    decoded = base64.b64decode(encoded_text)
    decompressed = zlib.decompress(decoded)
    return np.frombuffer(decompressed, dtype=np.float64)

# Parse do XML
tree = ET.parse(mzml_path)
root = tree.getroot()
ns = {'mzml': 'http://psi.hupo.org/ms/mzml'}

data = []

# Loop pelos espectros
for spectrum in root.findall('.//mzml:spectrum', ns):
    spectrum_id = spectrum.attrib.get("id", "")
    match = re.search(r"scan=(\d+)", spectrum_id)
    scan_no = int(match.group(1)) if match else None

    # Collision Energy
    ce = None
    for cv in spectrum.findall(".//mzml:activation/mzml:cvParam", ns):
        if cv.attrib.get("name") == "collision energy":
            ce = float(cv.attrib.get("value", "nan"))

    # Arrays binários
    mz_array = None
    intensity_array = None
    for array in spectrum.findall(".//mzml:binaryDataArray", ns):
        array_type = None
        for cv in array.findall("mzml:cvParam", ns):
            if cv.attrib.get("accession") == "MS:1000514":
                array_type = "mz"
            elif cv.attrib.get("accession") == "MS:1000515":
                array_type = "intensity"

        binary_elem = array.find("mzml:binary", ns)
        if binary_elem is not None:
            try:
                decoded_array = decode_binary_array(binary_elem.text)
                if array_type == "mz":
                    mz_array = decoded_array
                elif array_type == "intensity":
                    intensity_array = decoded_array
            except Exception as e:
                print(f"Erro no scan {scan_no}: {e}")

    if mz_array is not None and intensity_array is not None and ce is not None:
        for mz, intensity in zip(mz_array, intensity_array):
            data.append({
                "collision_energy": ce,
                "mz": mz,
                "intensity": intensity
            })

# Criar DataFrame e média
df = pd.DataFrame(data)
df_avg = df.groupby(["collision_energy", "mz"], as_index=False)["intensity"].mean()
df_avg.to_csv("espectros_media_por_CE.csv", index=False)
print("✔ Arquivo 'espectros_media_por_CE.csv' gerado.")

# Gráfico 3D
if args.plot:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    x = df_avg["mz"]
    y = df_avg["collision_energy"]
    z = df_avg["intensity"]

    ax.plot_trisurf(x, y, z, cmap='viridis', linewidth=0.2, antialiased=True)

    ax.set_xlabel("m/z")
    ax.set_ylabel("Collision Energy (eV)")
    ax.set_zlabel("Intensidade média")
    ax.set_title("Espectro de massas 3D por CE")

    plt.tight_layout()
    plt.savefig("plot_3D_spectros_por_ce.png", dpi=300)
    print("✔ Gráfico 3D 'plot_3D_spectros_por_ce.png' salvo.")
