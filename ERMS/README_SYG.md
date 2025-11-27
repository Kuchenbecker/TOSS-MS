
# 📘 SYF2 — Survival Yield Fitter & Breakdown Curve Extractor  
### *HCD → CE → CE_COM breakdown curves + automatic best-model fitting from mzML files*

---

## 🔍 What this tool does

`SYF2.py` reads a folder of **.mzML** files containing HCD-based fragmentation spectra, extracts intensities for selected fragment ions, and builds **breakdown curves** using:

- **HCD**
- **CE (lab-frame collision energy)**
- **CE_COM (center-of-mass energy)** ← *recommended for correct physics*

It can also:

| Capability | Description |
|---|---|
| Multi-ion extraction | Select multiple fragment m/z values at once |
| Relative or absolute signal | Choose %SUM(base peak) or raw intensities |
| CE_COM transformation | Requires known precursor m/z |
| Auto-model fitting | Tries 6 models → selects best by AIC |
| CSV export | Saves breakdown data for reuse |
| Combined + individual graphs | Single plot of all ions + separate windows per ion |

---

## 📂 Basic Usage

```bash
python SYF2.py <folder> <ions>
```

### Example:

```bash
python SYF2.py data_folder 59,135.102,117.092,77.04,45.01
```

→ Reads all `*.mzML` → Extracts intensities → Plots **combined breakdown curve vs HCD**

---

## ⚙ Command-Line Options

### Energy System

| Flag | Meaning |
|---|---|
| `--CE` | Convert HCD → collision energy (eV) using calibration `CE = 0.1742·HCD + 3.8701` |
| `--COM` | Convert CE → **CE_COM** (requires `--CE` + `--precursor-mz`) |
| `--precursor-mz <m/z>` | Required for CE_COM. Ensures energy axis is physically correct. |

**CE_COM formula used**

\[
CE_{\text{COM}} = CE \times \frac{28.0134}{M_{\text{precursor}} + 28.0134}
\]

---

### Plot Control

| Flag | Function |
|---|---|
| `--s` | Show individual plots *per ion* (in addition to the combined plot) |
| `--fit` | Fit each ion curve to **all models** and choose the best (requires `--s`) |
| `--tog a,b,c` | Combined plot only shows these ions (still extracts all ions) |

---

### Data Mode & Output

| Flag | Meaning |
|---|---|
| `--r` | Plot intensities as % relative values instead of absolute |
| `--csv` | Save results to `SYF_output/*.csv` |

---

### Peak Integration Parameters

| Flag | Meaning |
|---|---|
| `--mtol <Da>` | m/z extraction tolerance window (absolute Da) |
| `--agg {sum,mean,max,gauss}` | Peak integration approach (default: `sum`) |
| `--gauss-sigma <fraction>` | Gaussian σ width = `<fraction> × mtol` (only used if `--agg gauss`) |

---

## 📈 Built-in Fitting Models

| Model | Purpose |
|---|---|
| 4PL | Sigmoidal survival-type decay |
| Exponential | First‑order dissociation decay |
| WeibullSurv | Flexible curve for survival → dissociation transitions |
| Gompertz | Asymmetric activation curve |
| GaussPeak | Peak‑shaped fragment yield |
| LogNormalPeak | Skewed peak shape for asymmetric dissociation regimes |

Best model = **lowest AIC**, tie → highest R².

---

## 🔧 Recommended Workflows

#### Breakdown curve (multi-ion, CE_COM, relative %)

```bash
python SYF2.py data 59,135.102,117.092,77.04,45.01     --CE --COM --precursor-mz 194.123 --r
```

#### Fit one fragment ion with all models

```bash
python SYF2.py data 59     --CE --COM --precursor-mz 194.123     --s --fit
```

#### Extract many ions but show only a few in the combined graph

```bash
python SYF2.py data 59,135.102,117.092,77.04,45.01     --CE --COM --precursor-mz 194.123     --tog 59,135.102
```

---

## Output Files

```
SYF_output/
├─ ion_intensities_by_cecom.csv
├─ ion_intensities_by_ce.csv
└─ ion_intensities_by_hcd.csv
```

---

## Cite if used in research

```
SYF2 Mass Spectrometry Breakdown/Fitting Toolkit
Author: <Your Name>
GitHub Repository: <repo link>
```
