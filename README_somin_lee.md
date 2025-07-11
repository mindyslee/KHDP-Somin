# KHDPDatathon
# Multimodal Heart-Failure Risk Project

A machine learning pipeline for the KHDP Datathon+Hackathon that fuses 12-lead ECG waveforms and patient comorbidity data to predict heart-failure risk.

**Purpose:**
- Demonstrate a complete end-to-end workflow on the KHDP platform  
- Extract and label ECG records with heart-failure outcomes  
- Build, train, and evaluate a multimodal deep-learning model under resource constraints  

**Datasets Used:**
- **SNUH ECG Registry**: 12-lead ECG traces in XML format (`datasets/ECG-Registry/1.0.0/1.MAIN`)  
- **SNUH CDM `condition_occurrence_v`**: to generate binary heart-failure labels  
---

This repository contains three Jupyter notebooks:

## 1. `hf_outcome.ipynb`

**Purpose:**
- Connect to SNUH OMOP CDM (`cdm_public.condition_occurrence_v`) and extract a binary heart-failure label per patient.  
- Save output to `data/hf_outcome.csv` with columns:
  - `person_id`  
  - `hf_outcome` (0/1)  

**Key Steps:**
1. Configure `psycopg2` database connection  
2. Run parameterized SQL to flag heart-failure concept IDs  
3. Load results into `hf_df` (pandas DataFrame)  
4. Write `hf_df` to `data/hf_outcome.csv`  
5. (Optional) Print value counts of `hf_outcome`  

### 2. `make_labels.ipynb`

**Purpose:**
- Combine ECG metadata with heart-failure labels to produce `data/labels_with_hf.csv`.  
- Metadata is stored as per-year CSVs (`*.csv.csv`) in  
  `datasets/ECG-Registry/1.0.0/ECG_Metafile`.  

**Key Steps:**
1. Glob and concatenate all registry CSVs into `meta_df`  
2. Build a base `data/labels.csv` with:
   - `person_id` (`nid`)  
   - `ecg_file` (basename of `nstri_path`)  
3. Save `data/labels.csv`  
4. Read `data/hf_outcome.csv`, merge on `person_id`  
5. Fill missing `hf_outcome` with 0, save as `data/labels_with_hf.csv`  
6. Print head and `hf_outcome` counts  

### 3. `multimodal_hf_risk.ipynb`

**Purpose:**
- Train a multimodal model combining:
  - ECG waveforms (`datasets/ECG-Registry/1.0.0/1.MAIN`)  
  - Comorbidity features (`data/tabular.csv`)  
- Predict `hf_outcome` using a 1D-CNN + MLP architecture.  

**Key Steps:**
1. Load `data/labels_with_hf.csv`, split unique patients into train/val  
2. Precompute mean/std for comorbidity features, build `comorb_map`  
3. Define PyTorch `Dataset`s:
   - `ECGDataset` (loads ECG files)  
   - `TabularDataset` (on-the-fly normalized features)  
   - `CombinedDataset` (yields `(signal, tabular, label)`)  
4. Use `WeightedRandomSampler` to balance classes  
5. Define `ECGEncoder`, `TabularEncoder`, and `CombinedModel`  
6. Train for 20 epochs, log per-batch and per-epoch losses  
7. Save checkpoint as `hf_risk_model.pth`  
