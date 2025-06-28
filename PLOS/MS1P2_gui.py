import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import matplotlib.pyplot as plt
from pyteomics import mzml
import numpy as np
import os

DEFAULT_MZ_TOLERANCE = 0.5
SOLVENT_EXCLUSION_TOLERANCE = 0.2

def get_average_ms1_spectrum(mzml_file, mz_tolerance=DEFAULT_MZ_TOLERANCE):
    all_mz = []
    all_intensities = []

    with mzml.read(mzml_file) as reader:
        ms1_scans = [s for s in reader if s['ms level'] == 1]
        if not ms1_scans:
            raise ValueError(f"No MS1 scans found in {mzml_file}")

        for scan in ms1_scans:
            mz_array = scan['m/z array']
            intensity_array = scan['intensity array']
            all_mz.extend(mz_array)
            all_intensities.extend(intensity_array)

    all_mz = np.array(all_mz)
    all_intensities = np.array(all_intensities)

    bins = np.arange(min(all_mz), max(all_mz) + mz_tolerance, mz_tolerance)
    digitized = np.digitize(all_mz, bins)

    avg_mz = []
    avg_intensity = []

    for i in range(1, len(bins)):
        bin_indices = np.where(digitized == i)[0]
        if len(bin_indices) > 0:
            avg_mz.append(np.mean(all_mz[bin_indices]))
            avg_intensity.append(np.mean(all_intensities[bin_indices]))

    return np.array(avg_mz), np.array(avg_intensity)

def prune_close_peaks(mz_array, intensity_array, tolerance=DEFAULT_MZ_TOLERANCE):
    if len(mz_array) == 0:
        return mz_array, intensity_array

    sort_idx = np.argsort(mz_array)
    mz_sorted = mz_array[sort_idx]
    intensity_sorted = intensity_array[sort_idx]

    filtered_mz, filtered_intensity = [], []
    current_cluster_mz, current_cluster_intensity = [mz_sorted[0]], [intensity_sorted[0]]

    for mz, intensity in zip(mz_sorted[1:], intensity_sorted[1:]):
        if mz - current_cluster_mz[-1] <= tolerance:
            current_cluster_mz.append(mz)
            current_cluster_intensity.append(intensity)
        else:
            max_idx = np.argmax(current_cluster_intensity)
            filtered_mz.append(current_cluster_mz[max_idx])
            filtered_intensity.append(current_cluster_intensity[max_idx])
            current_cluster_mz, current_cluster_intensity = [mz], [intensity]

    max_idx = np.argmax(current_cluster_intensity)
    filtered_mz.append(current_cluster_mz[max_idx])
    filtered_intensity.append(current_cluster_intensity[max_idx])

    return np.array(filtered_mz), np.array(filtered_intensity)

def exclude_solvent_peaks(sample_mz, sample_intensity, solvent_mz, tolerance=SOLVENT_EXCLUSION_TOLERANCE):
    if len(solvent_mz) == 0:
        return sample_mz, sample_intensity

    mask = np.ones(len(sample_mz), dtype=bool)

    for mz_solv in solvent_mz:
        matches = np.where(np.abs(sample_mz - mz_solv) <= tolerance)[0]
        mask[matches] = False

    filtered_mz = sample_mz[mask]
    filtered_intensity = sample_intensity[mask]

    return filtered_mz, filtered_intensity

def plot_ms1_spectra(main_data, overlap_data_list, title=None, intype='r',
                     outdir=None, clean_threshold=0.0, labelmin_threshold=0.0,
                     main_filename=None, ident_amostra='', lote='', responsavel=''):
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.3])
    ax = fig.add_subplot(gs[0])
    ax.axhline(y=0, color='black', linewidth=1)
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.4)

    def process_and_plot(mz_array, intensity_array, label, color=None):
        if intype.lower() == 'r':
            intensity_array = (intensity_array / np.max(intensity_array)) * 100

        mz_array, intensity_array = prune_close_peaks(mz_array, intensity_array)
        mask = intensity_array >= clean_threshold
        mz_array = mz_array[mask]
        intensity_array = intensity_array[mask]

        markerline, stemlines, baseline = ax.stem(
            mz_array, intensity_array,
            linefmt='-', markerfmt=' ', basefmt=' ', label=label)
        plt.setp(stemlines, 'linewidth', 1)
        if color:
            plt.setp(stemlines, 'color', color)

        for i, (mz, intensity) in enumerate(zip(mz_array, intensity_array)):
            if intensity >= labelmin_threshold:
                too_close = any(abs(mz - other_mz) < 5 for j, other_mz in enumerate(mz_array) if j != i)
                if intensity > 50 and not too_close:
                    ax.text(mz, intensity + 2, f'{mz:.2f}', ha='center', va='bottom', fontsize=8)
                else:
                    dx = 8 if i % 2 == 0 else -20
                    dy = 10 + (i % 3) * 5
                    ax.annotate(f'{mz:.2f}',
                                xy=(mz, intensity),
                                xytext=(mz + dx, intensity + dy),
                                textcoords='data',
                                fontsize=8,
                                ha='left' if dx > 0 else 'right',
                                va='bottom',
                                arrowprops=dict(arrowstyle='-', lw=0.8, color='black'))

        return mz_array, intensity_array

    main_label = os.path.splitext(os.path.basename(main_filename))[0] if main_filename else 'Input mzML'
    sample_mz, sample_intensity = process_and_plot(main_data[0], main_data[1], main_label)

    colors = plt.cm.tab10.colors
    for i, (mz_array, intensity_array, label) in enumerate(overlap_data_list):
        color = colors[(i+1) % len(colors)]
        process_and_plot(mz_array, intensity_array, label, color=color)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)
    ax.set_xlabel('m/z', fontsize=12)
    ylabel = 'Relative Intensity (%)' if intype.lower() == 'r' else 'Absolute Intensity'
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title or 'Average MS1 Spectrum', fontsize=14)
    ax.legend()

    top_indices = np.argsort(sample_intensity)[-5:][::-1]
    top_mz = sample_mz[top_indices]
    top_int = sample_intensity[top_indices]
    table_data = [[f"{mz:.4f}", f"{inten:.2f}"] for mz, inten in zip(top_mz, top_int)]

    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    table = ax_table.table(cellText=table_data, colLabels=["m/z", "Intensity"],
                           loc='center', cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    ax_footer = fig.add_subplot(gs[2])
    ax_footer.axis('off')
    footer_text = f"Ident. Amostra: {ident_amostra}    |    Lote: {lote}    |    Responsável: {responsavel}"
    ax_footer.text(0.5, 0.5, footer_text, ha='center', va='center', fontsize=10)

    plt.tight_layout()
    if outdir:
        os.makedirs(os.path.dirname(outdir) or '.', exist_ok=True)
        ext = os.path.splitext(outdir)[1].lower()
        if ext not in ['.png', '.svg', '.pdf']:
            raise ValueError("Output file extension must be .png, .svg, or .pdf")
        plt.savefig(outdir, dpi=300, bbox_inches='tight')
        print(f"Plot saved as {outdir}")
    else:
        plt.show()

class MS1PlotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MS1 Spectrum Plotter")
        self.geometry("600x600")
        self.configure(padx=10, pady=10)

        self.mzml_file = tk.StringVar()
        self.solvent_file = tk.StringVar()
        self.overlap_paths = []

        self.intype = tk.StringVar(value='r')
        self.clean = tk.DoubleVar(value=0.0)
        self.labelmin = tk.DoubleVar(value=0.0)
        self.plot_title = tk.StringVar()
        self.outfile = tk.StringVar()

        self.ident_amostra = tk.StringVar()
        self.lote = tk.StringVar()
        self.responsavel = tk.StringVar()

        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_widgets(self):
        ttk.Label(self, text="Arquivo de amostra (.mzML):").pack(anchor="w")
        ttk.Entry(self, textvariable=self.mzml_file, width=80).pack()
        ttk.Button(self, text="Selecionar", command=self.select_mzml).pack()

        ttk.Label(self, text="Arquivo de solvente (.mzML, opcional):").pack(anchor="w", pady=(10, 0))
        ttk.Entry(self, textvariable=self.solvent_file, width=80).pack()
        ttk.Button(self, text="Selecionar", command=self.select_solvent).pack()

        ttk.Label(self, text="Arquivos de sobreposição (.mzML):").pack(anchor="w", pady=(10, 0))
        ttk.Button(self, text="Adicionar sobreposição", command=self.add_overlap).pack()
        self.overlap_list = tk.Listbox(self, height=4)
        self.overlap_list.pack(fill="both", padx=5, pady=(0, 10))

        ttk.Label(self, text="Título do gráfico:").pack(anchor="w")
        ttk.Entry(self, textvariable=self.plot_title, width=60).pack()

        ttk.Label(self, text="Tipo de intensidade:").pack(anchor="w")
        ttk.Combobox(self, values=["r", "a"], textvariable=self.intype, state="readonly").pack()

        ttk.Label(self, text="Limite para desenhar picos (clean):").pack(anchor="w")
        ttk.Entry(self, textvariable=self.clean).pack()

        ttk.Label(self, text="Limite para rotular picos (labelmin):").pack(anchor="w")
        ttk.Entry(self, textvariable=self.labelmin).pack()

        ttk.Label(self, text="Ident. Amostra:").pack(anchor="w")
        ttk.Entry(self, textvariable=self.ident_amostra).pack()

        ttk.Label(self, text="Lote:").pack(anchor="w")
        ttk.Entry(self, textvariable=self.lote).pack()

        ttk.Label(self, text="Responsável:").pack(anchor="w")
        ttk.Entry(self, textvariable=self.responsavel).pack()

        ttk.Label(self, text="Salvar gráfico como:").pack(anchor="w")
        ttk.Entry(self, textvariable=self.outfile, width=60).pack()
        ttk.Button(self, text="Escolher destino", command=self.save_as).pack()

        ttk.Button(self, text="Gerar Gráfico", command=self.run_analysis).pack(pady=10)

    def select_mzml(self):
        path = filedialog.askopenfilename(filetypes=[("mzML files", "*.mzML")])
        if path:
            self.mzml_file.set(path)

    def select_solvent(self):
        path = filedialog.askopenfilename(filetypes=[("mzML files", "*.mzML")])
        if path:
            self.solvent_file.set(path)

    def add_overlap(self):
        paths = filedialog.askopenfilenames(filetypes=[("mzML files", "*.mzML")])
        for p in paths:
            if p not in self.overlap_paths:
                self.overlap_paths.append(p)
                self.overlap_list.insert(tk.END, os.path.basename(p))

    def save_as(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=[("Image files", "*.png;*.svg;*.pdf")])
        if path:
            self.outfile.set(path)

    def run_analysis(self):
        if not self.mzml_file.get():
            messagebox.showerror("Erro", "Arquivo de amostra não selecionado.")
            return

        try:
            mz_array, intensity_array = get_average_ms1_spectrum(self.mzml_file.get())

            if self.solvent_file.get():
                solv_mz, solv_int = get_average_ms1_spectrum(self.solvent_file.get())
                mz_array, intensity_array = exclude_solvent_peaks(mz_array, intensity_array, solv_mz)

            main_data = (mz_array, intensity_array)

            overlap_data_list = []
            for file in self.overlap_paths:
                mz_o, int_o = get_average_ms1_spectrum(file)
                label = os.path.splitext(os.path.basename(file))[0]
                overlap_data_list.append((mz_o, int_o, label))

            plot_ms1_spectra(
                main_data,
                overlap_data_list,
                title=self.plot_title.get(),
                intype=self.intype.get(),
                outdir=self.outfile.get() if self.outfile.get() else None,
                clean_threshold=self.clean.get(),
                labelmin_threshold=self.labelmin.get(),
                main_filename=self.mzml_file.get(),
                ident_amostra=self.ident_amostra.get(),
                lote=self.lote.get(),
                responsavel=self.responsavel.get()
            )

        except Exception as e:
            messagebox.showerror("Erro ao processar", str(e))

    def on_close(self):
        self.destroy()
        os._exit(0)

if __name__ == "__main__":
    app = MS1PlotGUI()
    app.mainloop()
