# SMPL-Anthropometry

Measure the SMPL/SMPLX body models and visualize the measurements and landmarks.

<p align="center">
  <img src="https://github.com/DavidBoja/SMPL-Anthropometry/blob/master/assets/measurement_visualization.png" width="950">
</p>

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run quick test
./scripts/quickstart.sh

# 3. View existing results
python3 -m src.visualization.view_smpl_3d \
    --params outputs/output_from_txt_fixed/smpl_params.npz
```

**📖 [5-Minute Quick Start Guide](QUICK_START.md)**

---

## 📁 Project Structure

```
SMPL-Anthropometry/
├── src/                    # Source code
│   ├── core/              # Core measurement modules
│   ├── fitting/           # SMPL fitting algorithms
│   └── visualization/     # 3D visualization tools
├── tools/                 # Utility scripts
├── scripts/               # Convenience shell scripts
├── examples/              # Example code
├── docs/                  # Documentation
├── data/                  # SMPL model files
└── outputs/               # Runtime outputs
```

**📋 [Detailed Structure](docs/PROJECT_STRUCTURE.md)**

---

## 📖 Documentation

### Getting Started
- **[QUICK_START.md](QUICK_START.md)** - 5-minute quick start
- **[docs/INSTALL.md](docs/INSTALL.md)** - Installation guide
- **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Usage guide

### Technical Documentation
- **[docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)** - Project structure
- **[docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Project status
- **[docs/TXT_FITTING_GUIDE.md](docs/TXT_FITTING_GUIDE.md)** - TXT fitting guide
- **[docs/README_TOOLS.md](docs/README_TOOLS.md)** - Tools usage

### Project Restructuring
- **[docs/FINAL_SUMMARY.md](docs/FINAL_SUMMARY.md)** - Final summary
- **[docs/RESTRUCTURE_REPORT.md](docs/RESTRUCTURE_REPORT.md)** - Complete report
- **[docs/GIT_COMMIT_SUMMARY.md](docs/GIT_COMMIT_SUMMARY.md)** - Git commits

---

## 🔧 Installation

### Method 1: Using pip
```bash
pip install -r requirements.txt
```

### Method 2: Using Docker
```bash
cd docker
sh build.sh
sh docker_run.sh /path/to/SMPL-Anthropometry
```

### Download SMPL Models
Place SMPL model files in:
- `data/smpl/` - SMPL_{GENDER}.pkl files
- `data/smplx/` - SMPLX_{GENDER}.pkl files

**📥 [Download Guide](docs/DOWNLOAD_SMPL.md)**

---

## 🏃 Usage

### Basic Measurement
```python
from src.core.measure import MeasureBody
from src.core.measurement_definitions import STANDARD_LABELS

# Create measurer
measurer = MeasureBody('smpl')
measurer.from_body_model(gender='NEUTRAL', shape=betas)

# Measure
measurer.measure(measurer.all_possible_measurements)
measurer.label_measurements(STANDARD_LABELS)

# Get results
measurements = measurer.measurements
```

### Fit SMPL from TXT File
```bash
python3 -m src.fitting.fit_smpl_from_txt_fixed \
    --input your_data.txt \
    --output outputs/result \
    --visualize
```

### 3D Visualization
```bash
python3 -m src.visualization.view_smpl_3d \
    --params outputs/result/smpl_params.npz \
    --save_html outputs/body_3d.html
```

---

## 🛠️ Utility Scripts

Located in `scripts/` directory:

```bash
./scripts/quick_view.sh        # Quick 3D visualization
./scripts/batch_export.sh      # Batch export to HTML
./scripts/compare_results.sh   # Compare measurements
./scripts/quick_commands.sh    # Shortcut commands
```

**📘 [Scripts Documentation](scripts/README.md)**

---

## 📊 Measurements

16 standard body measurements (in cm):
- Height, Chest, Waist, Hip circumferences
- Shoulder breadth, Arm lengths
- Leg lengths, Head circumference
- And more...

**📏 [Full Measurement List](docs/USAGE_GUIDE.md)**

---

## 🎨 Features

- ✅ **Modular Architecture** - Clean src/core, src/fitting, src/visualization structure
- ✅ **Multiple Input Sources** - Point cloud, keypoints, TXT files
- ✅ **3D Interactive Visualization** - Browser-based with Plotly
- ✅ **Batch Processing** - Process multiple results at once
- ✅ **Offline HTML Export** - Share visualizations easily
- ✅ **Comprehensive Documentation** - 16+ documentation files
- ✅ **Utility Scripts** - 6 convenience shell scripts
- ✅ **Package Installation** - `pip install -e .`

---

## 🔬 Advanced Usage

### Command-Line Tools (after pip install)
```bash
smpl-measure        # Measure default model
smpl-fit-txt        # Fit from TXT file
smpl-view-3d        # 3D viewer
smpl-check          # Check model files
```

### Python API
```python
from src.core.measure import MeasureBody
from src.fitting.fit_smpl_from_data import SMPLFitterFromData
from src.visualization.visualize import Visualizer
```

---

## 📈 Project Status

- **Version:** v1.0.0-restructured
- **Status:** ✅ Production Ready
- **Python:** 3.7+
- **License:** MIT

**📊 [Project Status Report](docs/PROJECT_STATUS.md)**

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🗞️ Citation

```bibtex
@misc{SMPL-Anthropometry,
  author = {Bojani\'{c}, D.},
  title = {SMPL-Anthropometry},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DavidBoja/SMPL-Anthropometry}},
}
```

---

## 🔗 Links

- **Documentation:** [docs/](docs/)
- **Examples:** [examples/](examples/)
- **Tools:** [tools/](tools/)
- **Scripts:** [scripts/](scripts/)

---

⭐ **If you find this project useful, please give it a star!** ⭐
