import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import itertools

def main():
    base_dir = Path(r"d:\Study\Projects\Fracture-Detection_HDAsubject\WristTrauma-AI")
    results_dir = base_dir / "results"
    fig_dir = results_dir / "figures"
    metadata_path = base_dir / "data" / "dataset_metadata_cleaned.csv"
    labels_dir = base_dir / "data" / "raw" / "yolov5" / "labels"
    
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(metadata_path)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    
    # 2. Extract Finding Info from Annotations
    print("Parsing annotations...")
    class_names = {
        0: 'boneanomaly', 1: 'bonelesion', 2: 'foreignbody', 
        3: 'fracture', 4: 'metal', 5: 'periostealreaction', 
        6: 'pronatorsign', 7: 'softtissue', 8: 'text'
    }
    
    img_findings = {}
    for idx, row in df.iterrows():
        stem = row['filestem']
        lbl_path = labels_dir / f"{stem}.txt"
        findings = set()
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        findings.add(class_names[int(parts[0])])
        img_findings[stem] = list(findings)
        
    df['findings'] = df['filestem'].map(img_findings)
    df['has_fracture'] = df['findings'].apply(lambda x: 'fracture' in x)
    
    # 3. Setup Plotting
    sns.set_theme(style="whitegrid")
    
    # --- A. Demographics ---
    # Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='age', bins=20, kde=True, color='skyblue')
    plt.title('Age Distribution of Patients')
    plt.xlabel('Age (Years)')
    plt.ylabel('Count')
    plt.savefig(fig_dir / 'age_distribution.png')
    plt.close()
    
    # Sex Distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='gender', palette='Set2', order=df['gender'].value_counts().index)
    plt.title('Sex Distribution')
    plt.savefig(fig_dir / 'sex_distribution.png')
    plt.close()
    
    # --- B. Clinical Info ---
    # Laterality
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='laterality', palette='pastel', order=df['laterality'].value_counts().index)
    plt.title('Left vs Right Wrist')
    plt.savefig(fig_dir / 'laterality_distribution.png')
    plt.close()
    
    # Projection
    plt.figure(figsize=(8, 6))
    proj_map = {1: 'PA/AP', 2: 'Lateral', 3: 'Other'}
    df['projection_label'] = df['projection'].map(proj_map)
    sns.countplot(data=df, x='projection_label', palette='Set3', order=['PA/AP', 'Lateral', 'Other'])
    plt.title('Projection Distribution')
    plt.savefig(fig_dir / 'projection_distribution.png')
    plt.close()
    
    # --- C. Findings & Pathology ---
    # All findings counts
    all_findings = list(itertools.chain(*df['findings']))
    finding_counts = pd.Series(all_findings).value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=finding_counts.values, y=finding_counts.index, palette='viridis')
    plt.title('Distribution of Annotated Findings')
    plt.xlabel('Count')
    plt.savefig(fig_dir / 'findings_distribution.png')
    plt.close()
    
    # Fracture Prevalence
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='has_fracture', palette='Set1')
    plt.title('Fracture Prevalence (Presence in Image)')
    plt.savefig(fig_dir / 'fracture_prevalence.png')
    plt.close()
    
    # AO Classification (Top 15)
    plt.figure(figsize=(12, 8))
    ao_counts = df['ao_classification'].value_counts().head(15)
    sns.barplot(x=ao_counts.values, y=ao_counts.index, palette='magma')
    plt.title('Top 15 AO Fracture Classifications')
    plt.xlabel('Count')
    plt.savefig(fig_dir / 'ao_classification_top15.png')
    plt.close()
    
    # --- D. Relationships ---
    # Age vs Fracture
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='age', hue='has_fracture', fill=True, common_norm=False, palette='Set1')
    plt.title('Age Distribution by Fracture Presence')
    plt.savefig(fig_dir / 'age_vs_fracture.png')
    plt.close()
    
    # Sex vs Fracture
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='gender', hue='has_fracture', palette='Set1')
    plt.title('Sex vs Fracture Prevalence')
    plt.savefig(fig_dir / 'sex_vs_fracture.png')
    plt.close()

    # Combinations of findings
    df['findings_tuple'] = df['findings'].apply(lambda x: tuple(sorted(x)))
    combo_counts = df['findings_tuple'].value_counts().head(10)
    
    plt.figure(figsize=(12, 8))
    combo_labels = [', '.join(x) if x else 'None' for x in combo_counts.index]
    sns.barplot(x=combo_counts.values, y=combo_labels, palette='cubehelix')
    plt.title('Top 10 Most Common Finding Combinations')
    plt.xlabel('Count')
    plt.savefig(fig_dir / 'finding_combinations.png')
    plt.close()
    
    # --- E. Missing Values & Patient Stats ---
    missing_data = df.isnull().sum()
    
    patient_stats = {
        'total_images': len(df),
        'unique_patients': df['patient_id'].nunique(),
        'avg_images_per_patient': len(df) / df['patient_id'].nunique(),
        'missing_values': missing_data.to_dict()
    }
    
    # Generate Markdown Report
    report_path = results_dir / "eda_report.md"
    with open(report_path, 'w') as f:
        f.write("# GRAZPEDWRI-DX Healthcare EDA Report\n\n")
        
        f.write("## 1. Patient & Demographics\n")
        f.write(f"- **Total Patients**: {patient_stats['unique_patients']}\n")
        f.write(f"- **Total Images**: {patient_stats['total_images']}\n")
        f.write(f"- **Avg Images per Patient**: {patient_stats['avg_images_per_patient']:.2f}\n")
        f.write(f"- **Mean Age**: {df['age'].mean():.2f} years (Min: {df['age'].min()}, Max: {df['age'].max()})\n")
        f.write(f"- **Sex**: {df['gender'].value_counts().to_dict()}\n\n")
        
        f.write("## 2. Clinical Distribution\n")
        f.write(f"- **Laterality**: {df['laterality'].value_counts().to_dict()}\n")
        f.write(f"- **Projection**: {df['projection_label'].value_counts().to_dict()}\n\n")
        
        f.write("## 3. Pathology & Findings\n")
        f.write(f"- **Fracture Prevalence**: {df['has_fracture'].mean()*100:.1f}% of images contain a fracture.\n")
        f.write("- **Top 5 Annotated Findings**:\n")
        for finding, count in finding_counts.head(5).items():
            f.write(f"  - {finding}: {count}\n")
            
        f.write("\n## 4. Missing Values Analysis\n")
        for col, count in missing_data.items():
            if count > 0:
                f.write(f"- **{col}**: {count} missing ({count/len(df)*100:.1f}%)\n")
                
    print("EDA Generation Complete. Figures saved to results/figures/ and report saved to results/eda_report.md")

if __name__ == "__main__":
    main()
