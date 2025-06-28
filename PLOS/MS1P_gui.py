
import sys
import os
import glob
import numpy as np
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from pyteomics import mzml
import threading

DEFAULT_MZ_TOLERANCE = 0.5
SOLVENT_EXCLUSION_TOLERANCE = 0.2

def get_average_ms1_spectrum(mzml_file, mz_tolerance=DEFAULT_MZ_TOLERANCE):
    all_mz, all_intensities = [], []
    with mzml.read(mzml_file) as reader:
        ms1_scans = [s for s in reader if s['ms level'] == 1]
        if not ms1_scans:
            raise ValueError(f"No MS1 scans found in {mzml_file}")
        for scan in ms1_scans:
            all_mz.extend(scan['m/z array'])
            all_intensities.extend(scan['intensity array'])
    all_mz, all_intensities = np.array(all_mz), np.array(all_intensities)
    bins = np.arange(min(all_mz), max(all_mz)+mz_tolerance, mz_tolerance)
    digitized = np.digitize(all_mz, bins)
    avg_mz, avg_intensity = [], []
    for i in range(1, len(bins)):
        bin_idx = np.where(digitized == i)[0]
        if len(bin_idx) > 0:
            avg_mz.append(np.mean(all_mz[bin_idx]))
            avg_intensity.append(np.mean(all_intensities[bin_idx]))
    return np.array(avg_mz), np.array(avg_intensity)

def exclude_solvent_peaks(sample_mz, sample_intensity, solvent_mz, tolerance=SOLVENT_EXCLUSION_TOLERANCE):
    mask = np.ones(len(sample_mz), dtype=bool)
    for mz_solv in solvent_mz:
        matches = np.where(np.abs(sample_mz - mz_solv) <= tolerance)[0]
        mask[matches] = False
    return sample_mz[mask], sample_intensity[mask]

def prune_close_peaks(mz_array, intensity_array, tolerance=DEFAULT_MZ_TOLERANCE):
    if len(mz_array) == 0:
        return mz_array, intensity_array
    sort_idx = np.argsort(mz_array)
    mz_sorted, int_sorted = mz_array[sort_idx], intensity_array[sort_idx]
    filtered_mz, filtered_int = [], []
    current_cluster_mz, current_cluster_int = [mz_sorted[0]], [int_sorted[0]]
    for mz, inten in zip(mz_sorted[1:], int_sorted[1:]):
        if mz - current_cluster_mz[-1] <= tolerance:
            current_cluster_mz.append(mz)
            current_cluster_int.append(inten)
        else:
            max_idx = np.argmax(current_cluster_int)
            filtered_mz.append(current_cluster_mz[max_idx])
            filtered_int.append(current_cluster_int[max_idx])
            current_cluster_mz, current_cluster_int = [mz], [inten]
    max_idx = np.argmax(current_cluster_int)
    filtered_mz.append(current_cluster_mz[max_idx])
    filtered_int.append(current_cluster_int[max_idx])
    return np.array(filtered_mz), np.array(filtered_int)

class MS1PGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.canvas = None

    def init_ui(self):
        self.setWindowTitle("MS1 Plot GUI")
        layout = QtWidgets.QVBoxLayout()

        self.input_entry = QtWidgets.QLineEdit()
        input_btn = QtWidgets.QPushButton("Browse Input mzML")
        input_btn.clicked.connect(lambda: self.browse_file(self.input_entry))
        layout.addWidget(QtWidgets.QLabel("Input mzML:"))
        layout.addWidget(self.input_entry)
        layout.addWidget(input_btn)

        self.solvent_entry = QtWidgets.QLineEdit()
        solvent_btn = QtWidgets.QPushButton("Browse Solvent mzML")
        solvent_btn.clicked.connect(lambda: self.browse_file(self.solvent_entry))
        layout.addWidget(QtWidgets.QLabel("Solvent mzML (optional):"))
        layout.addWidget(self.solvent_entry)
        layout.addWidget(solvent_btn)

        self.overlap_entry = QtWidgets.QLineEdit()
        overlap_file_btn = QtWidgets.QPushButton("Browse Overlap File")
        overlap_file_btn.clicked.connect(lambda: self.browse_file(self.overlap_entry))
        overlap_folder_btn = QtWidgets.QPushButton("Browse Overlap Folder")
        overlap_folder_btn.clicked.connect(lambda: self.browse_folder(self.overlap_entry))
        layout.addWidget(QtWidgets.QLabel("Overlap mzML or folder (optional):"))
        layout.addWidget(self.overlap_entry)
        layout.addWidget(overlap_file_btn)
        layout.addWidget(overlap_folder_btn)

        self.intype_box = QtWidgets.QComboBox()
        self.intype_box.addItems(['r', 'a'])
        layout.addWidget(QtWidgets.QLabel("Intensity type (r=relative, a=absolute):"))
        layout.addWidget(self.intype_box)

        self.clean_entry = QtWidgets.QLineEdit("0.0")
        self.labelmin_entry = QtWidgets.QLineEdit("0.0")
        layout.addWidget(QtWidgets.QLabel("Clean threshold:"))
        layout.addWidget(self.clean_entry)
        layout.addWidget(QtWidgets.QLabel("Labelmin threshold:"))
        layout.addWidget(self.labelmin_entry)

        gen_btn = QtWidgets.QPushButton("Generate")
        gen_btn.clicked.connect(self.generate_plot)
        save_btn = QtWidgets.QPushButton("Save")
        save_btn.clicked.connect(self.save_plot)
        layout.addWidget(gen_btn)
        layout.addWidget(save_btn)

        self.progress = QtWidgets.QProgressBar()
        layout.addWidget(self.progress)

        self.plot_widget = QtWidgets.QWidget()
        layout.addWidget(self.plot_widget)

        self.setLayout(layout)
        self.show()

    def browse_file(self, entry):
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select mzML file", "", "mzML files (*.mzML)")
        if filepath:
            entry.setText(filepath)

    def browse_folder(self, entry):
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder")
        if folderpath:
            entry.setText(folderpath)

    def generate_plot(self):
        self.progress.setValue(0)
        thread = threading.Thread(target=self.process_data_thread)
        thread.start()

    def process_data_thread(self):
        try:
            input_path = self.input_entry.text()
            if not input_path:
                QtWidgets.QMessageBox.warning(self, "Error", "Input mzML is required.")
                return
            mz_array, intensity_array = get_average_ms1_spectrum(input_path)
            self.progress.setValue(30)

            solvent_path = self.solvent_entry.text()
            if solvent_path:
                solvent_mz, _ = get_average_ms1_spectrum(solvent_path)
                mz_array, intensity_array = exclude_solvent_peaks(mz_array, intensity_array, solvent_mz)
            self.progress.setValue(50)

            overlap_path = self.overlap_entry.text()
            overlap_data = []
            if overlap_path:
                files = [overlap_path] if os.path.isfile(overlap_path) else glob.glob(os.path.join(overlap_path, '*.mzML'))
                for file in files:
                    mz_o, int_o = get_average_ms1_spectrum(file)
                    overlap_data.append((mz_o, int_o, os.path.splitext(os.path.basename(file))[0]))
            self.progress.setValue(70)

            QtCore.QMetaObject.invokeMethod(self, "create_plot", QtCore.Qt.QueuedConnection,
                                            QtCore.Q_ARG(object, mz_array),
                                            QtCore.Q_ARG(object, intensity_array),
                                            QtCore.Q_ARG(object, overlap_data))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            self.progress.setValue(0)

    @QtCore.pyqtSlot(object, object, object)
    def create_plot(self, mz_array, intensity_array, overlap_data):
        fig, ax = plt.subplots(figsize=(10, 6))

        intype = self.intype_box.currentText()
        if intype == 'r':
            intensity_array = (intensity_array / np.max(intensity_array)) * 100
        mz_array, intensity_array = prune_close_peaks(mz_array, intensity_array)

        clean = float(self.clean_entry.text())
        labelmin = float(self.labelmin_entry.text())

        mask = intensity_array >= clean
        mz_array, intensity_array = mz_array[mask], intensity_array[mask]

        ax.stem(mz_array, intensity_array, label='Input mzML')

        for mz, inten in zip(mz_array, intensity_array):
            if inten >= labelmin:
                ax.text(mz, inten, f'{mz:.2f}', rotation=90, va='bottom', ha='center', fontsize=7)

        for (mz_o, int_o, label) in overlap_data:
            if intype == 'r':
                int_o = (int_o / np.max(int_o)) * 100
            mz_o, int_o = prune_close_peaks(mz_o, int_o)
            ax.stem(mz_o, int_o, label=label)

        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(True)

        if self.canvas:
            self.plot_layout.removeWidget(self.canvas)
            self.canvas.deleteLater()
        self.canvas = FigureCanvas(fig)
        self.plot_layout = QtWidgets.QVBoxLayout(self.plot_widget)
        self.plot_layout.addWidget(self.canvas)
        self.progress.setValue(100)

    def save_plot(self):
        if not self.canvas:
            QtWidgets.QMessageBox.warning(self, "Error", "No plot to save. Generate it first.")
            return
        filepath, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Plot", "", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if filepath:
            self.canvas.figure.savefig(filepath, dpi=300, bbox_inches='tight')
            QtWidgets.QMessageBox.information(self, "Saved", f"Plot saved as {filepath}")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    gui = MS1PGUI()
    sys.exit(app.exec_())
