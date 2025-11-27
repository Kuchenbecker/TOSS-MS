import xml.etree.ElementTree as ET
import base64
import zlib
import numpy as np
import pandas as pd
import re
import argparse

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description="Extrai espectros médios por CE de um arquivo mzML.")
parser.add_argument("--path", required=True, help="Caminho para o arquivo mzML")
parser.add_argument("--r", action="store_true", help="Transforma intensidades em relativas por CE (pico base = 100%)")
parser.add_argument("--cutoff", type=float, default=0.0, help="Intensidade mínima relativa (%) para manter (usar com --r)")
parser.add_argument("--out", default="espectros_media_por_CE.csv", help="Nome do arquivo CSV de saída")
args = parser.parse_args()

mzml_path = args.path
saida_csv = args.out

# Função para decodificar arrays binários base64 + zlib → float64
def decode_binary_array(encoded_text):
    decoded = base64.b64decode(encoded_text)
    decompressed = zlib.decompress(decoded)
    return np.frombuffer(decompressed, dtype=np.float64)

# Parse do arquivo mzML
tree = ET.parse(mzml_path)
root = tree.getroot()
ns = {'mzml': 'http://psi.hupo.org/ms/mzml'}

data = []

# Loop pelos espectros
for spectrum in root.findall('.//mzml:spectrum', ns):
    spectrum_id = spectrum.attrib.get("id", "")
    match = re.search(r"scan=(\d+)", spectrum_id)
    scan_no = int(match.group(1)) if match else None

    # Extrair energia de colisão
    ce = None
    for cv in spectrum.findall(".//mzml:activation/mzml:cvParam", ns):
        if cv.attrib.get("name") == "collision energy":
            ce = float(cv.attrib.get("value", "nan"))

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
                print(f"Erro ao decodificar scan {scan_no}: {e}")

    # Armazena os dados do espectro
    if mz_array is not None and intensity_array is not None and ce is not None:
        for mz, intensity in zip(mz_array, intensity_array):
            data.append({
                "collision_energy": ce,
                "mz": mz,
                "intensity": intensity
            })

# Agrupamento e média
df = pd.DataFrame(data)
df_avg = df.groupby(["collision_energy", "mz"], as_index=False)["intensity"].mean()

# Se for intensidade relativa
if args.r:
    df_out = []

    # Normalização por CE
    for ce, group in df_avg.groupby("collision_energy"):
        max_int = group["intensity"].max()
        group["intensity"] = 100 * group["intensity"] / max_int
        df_out.append(group)

    df_avg = pd.concat(df_out, ignore_index=True)

    # Aplicar cutoff depois da normalização
    if args.cutoff > 0:
        df_avg = df_avg[df_avg["intensity"] >= args.cutoff]

# Exporta resultado
df_avg.to_csv(saida_csv, index=False)
print(f"✔ Arquivo '{saida_csv}' gerado com sucesso.")
