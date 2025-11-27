
# GRAPHGEN (Multi‑Dataset Version)

`GRAPHGEN (copy).py` is a flexible scientific plotting utility designed to **visualize numerical data from CSV files**, optionally fit mathematical models, and produce publication‑grade graphs.  
Unlike the original GRAPHGEN, this version supports **multiple CSV inputs**, allowing direct comparison between datasets (different experiments, replicates, conditions, energies, etc.).

---

## 🔍 What This Tool Does

- Loads one or more `.csv` files (overlay mode)  
- Plots `x vs y` as scatter or connected lines  
- Optionally performs mathematical model fitting (`--fit`)  
- Displays equations and R² values in terminal for each dataset  
- Saves or displays high‑quality figures automatically  

---

## 📂 CSV Format Requirements

Your file(s) must contain at least two numeric columns. By default:

| Column | Meaning |
|-------|---------|
| Column 1 | **X axis values** |
| Column 2 | **Y axis values** |

You may change which columns are used (see parameters below).

---

## 🛠 Usage

### 📌 Single CSV Mode

```bash
python "GRAPHGEN (copy).py" data.csv
```

### 📌 Multiple CSV Overlay Mode

```bash
python "GRAPHGEN (copy).py" --multi file1.csv,file2.csv,file3.csv
```

Each dataset is plotted with a unique color.

---

## 🔧 Main Options

| Argument | Description |
|---------|-------------|
| `--multi a.csv,b.csv` | Loads & overlays multiple datasets |
| `--xaxis <n>` | Selects column for X (default = 0) |
| `--yaxis <n>` | Selects column for Y (default = 1) |
| `--fit <model>` | Fits curve (linear, exp, 4pl, gompertz, weibull...) |
| `--connect` | Connects scatter points with a line |
| `--show` | Displays y‑values as labels above points |
| `--output <file.png/svg>` | Saves the graph instead of showing on screen |

Example fitting a curve:

```bash
python "GRAPHGEN (copy).py" results.csv --fit 4pl --output CE_fit.svg
```

Example comparing datasets:

```bash
python "GRAPHGEN (copy).py" --multi CE10.csv,CE20.csv,CE40.csv --fit exp --connect
```

Each dataset will be fit **independently**, and equations will appear in terminal output.

---

## 🧠 Notes on Multi‑Dataset Behavior

- All loaded CSVs share the same X/Y axis selections  
- When using multiple files, equations **do not appear inside the plot** (to keep visuals clean)  
- With `--fit` each dataset gets:  
  ✔ its own curve  
  ✔ its own parameters  
  ✔ its own R² score  

---

## 📌 Ideal Applications

- Dose‑response modelling  
- Breakdown curves / ERMS profiling  
- Spectral intensity comparisons  
- Calibration curve evaluation  
- Any research requiring **numeric correlations + model fitting**  

---

## 🌟 Summary

| Feature | GRAPHGEN | GRAPHGEN (copy) |
|---|---|
| Single CSV plotting | ✔ | ✔ |
| Multiple inputs | ❌ | **✔** |
| Fits per dataset | 1 | **Many** |
| Overlaid visualization | ❌ | **✔** |

---

### Ready to Use.
Upload your CSV, call the script, generate models, visualize science. 🚀
