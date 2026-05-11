# Temporal4DFlowNet - Deep learning for temporal super-resolution 4D Flow MRI
A residual CNN for temporal super-resolution of 4D Flow MRI data.
This repository extends [4DFlowNet](https://github.com/EdwardFerdian/4DFlowNet) ([Link to paper](https://www.frontiersin.org/articles/10.3389/fphy.2020.00138/full)) from spatial to **temporal** super-resolution.

📄 **Paper**: [Deep learning for temporal super-resolution 4D Flow MRI](https://pubmed.ncbi.nlm.nih.gov/41880246/)

---

<!-- ## Method Overview
![Adapted 4DFlowNet Architecture](examples/architecture.png)
*Overview of the Temporal4DFlowNet architecture.*

---

## Example Results
![Example result](https://github.com/PiaaCaa/Temporal4DFlowNet/blob/main/examples/Test_M4_Animate_u__HR_SR_LR_Test.gif?raw=true) -->

---

## Requirements
- Python 3.8.10
- TensorFlow 2.9.1 
- See `requirements.txt` for full dependencies

---

## Installation
```bash
git clone https://github.com/PiaaCaa/Temporal4DFlowNet.git
cd Temporal4DFlowNet
pip install -r requirements.txt
```

---

## Data

### Example data
Example high-resolution 4D Flow MRI data can be obtained from the original [4DFlowNet repository](https://github.com/EdwardFerdian/4DFlowNet).
This data needs to be processed through the pipeline first (see **Usage** below) to generate the low-resolution input data before training or prediction.

### Data format
All data should be in HDF5 format with the following structure:

| Field | Description | Shape |
|-------|-------------|-------|
| `u`, `v`, `w` | Velocity components | `(t, x, y, z)` |
| `mask` | Binary mask | `(t, x, y, z)` or `(x, y, z)` |
| `u_max`, `v_max`, `w_max` | VENC values | scalar |
| `mag_u`, `mag_v`, `mag_w` | Magnitude images (optional) | `(t, x, y, z)` |

### Preparing your own data
If you want to use your own 4D Flow MRI data, prepare it in the HDF5 format above before running `prepare_patches.py`. The low-resolution input can either be:
- **Already downsampled** in time — set `step_t: 1` in the patch config
- **Downsampled on the fly** from a full-resolution file — set `step_t: 2`

---

## Configuration
All scripts are configured via YAML config files. Templates are provided in `config/`:
```
config/
├── train.example.yaml            # Training configuration
├── prepare_patches.example.yaml  # Patch generation configuration
└── predict.example.yaml          # Prediction configuration
```

Copy and edit the relevant template before running:
```bash
cp config/train.example.yaml config/train.yaml
# Edit config/train.yaml with your local paths and parameters
```
Config files are excluded from version control — only the example templates are committed.

---

## Usage

### 1. Prepare low-resolution data
Generate a low-resolution HDF5 file from a high-resolution dataset:
```bash
python prepare_data/prepare_temporal_lowres_dataset.py
```
This creates a temporally downsampled version of the high-resolution data to use as LR input.

### 2. Generate training patches
```bash
python prepare_patches.py --config config/prepare_patches.yaml
```
This generates a CSV file with patch indices used during training.

Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `spatial_patch_size` | Spatial patch size | `16` |
| `temporal_patch_size` | Temporal patch size | `16` |
| `n_patch` | Patches per slice | `10` |
| `minimum_coverage` | Minimum fluid coverage per patch | `0.2` |
| `n_patches_augmented_from_original_patch` | Augmented versions per patch (max 6) | `4` |
| `save_nonaugmented_patch` | Also save the original patch | `true` |

### 3. Train
```bash
python train.py --config config/train.yaml
```

Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `patch_size` | Patch size `[t, x, y]` | `[16, 16, 16]` |
| `res_increase` | Temporal upsampling factor | `2` |
| `batch_size` | Training batch size | `32` |
| `initial_learning_rate` | Initial learning rate | `1e-4` |
| `epochs` | Number of training epochs | `100` |
| `n_low_resblock` | Residual blocks in LR space | `8` |
| `n_hi_resblock` | Residual blocks in HR space | `4` |
| `upsampling_block` | Upsampling method (`linear`, `nearest_neighbor`, `conv3d_transpose`) | `linear` |
| `loss_type` | Base loss function (`mse`, `mae`, `huber`) | `mse` |
| `use_directional_loss` | Add physics-informed directional loss term | `true` |
| `preload_data` | Load all data into RAM at init (faster, requires more memory) | `false` |

Model weights, training logs, source backup, and config are saved automatically to `models/`.

### 4. Predict
```bash
python predict.py --config config/predict.yaml
```


Set `downsample_input_first: true` for paired in-vivo data where you want to downsample first and compare directly to the acquired resolution.

---

## Contact
For questions or issues, contact pia.callmer@ki.se or open a GitHub issue.
