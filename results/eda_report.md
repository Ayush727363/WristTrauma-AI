# GRAZPEDWRI-DX Healthcare EDA Report

## 1. Patient & Demographics
- **Total Patients**: 6091
- **Total Images**: 20327
- **Avg Images per Patient**: 3.34
- **Mean Age**: 10.92 years (Min: 0.2, Max: 19.0)
- **Sex**: {'M': 12040, 'F': 8285, 'O': 2}

## 2. Clinical Distribution
- **Laterality**: {'L': 11135, 'R': 9192}
- **Projection**: {'Lateral': 10148, 'PA/AP': 10086, 'Other': 93}

## 3. Pathology & Findings
- **Fracture Prevalence**: 66.7% of images contain a fracture.
- **Top 5 Annotated Findings**:
  - text: 20274
  - fracture: 13550
  - periostealreaction: 2235
  - metal: 707
  - pronatorsign: 566

## 4. Missing Values Analysis
- **initial_exam**: 9466 missing (46.6%)
- **ao_classification**: 6169 missing (30.3%)
- **cast**: 14551 missing (71.6%)
- **diagnosis_uncertain**: 19790 missing (97.4%)
- **osteopenia**: 17854 missing (87.8%)
- **fracture_visible**: 6777 missing (33.3%)
- **metal**: 19619 missing (96.5%)
