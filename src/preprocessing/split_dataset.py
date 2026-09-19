import os
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    # Define paths
    base_dir = Path(r"d:\Study\Projects\Fracture-Detection_HDAsubject\WristTrauma-AI")
    raw_images_dir = base_dir / "data" / "raw" / "images"
    raw_labels_dir = base_dir / "data" / "raw" / "yolov5" / "labels"
    processed_dir = base_dir / "data" / "processed"
    metadata_path = base_dir / "data" / "dataset_metadata_cleaned.csv"

    # Load metadata
    print("Loading metadata...")
    df = pd.read_csv(metadata_path)
    
    # Ensure processed directories exist
    splits = ['train', 'val', 'test']
    for split in splits:
        (processed_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        
    print("Copying files to processed structure...")
    missing_images = 0
    missing_labels = 0
    
    # Iterate through rows and copy
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Copying Dataset"):
        filestem = row['filestem']
        split = row['split']
        
        # Source paths
        src_image = raw_images_dir / f"{filestem}.png"
        src_label = raw_labels_dir / f"{filestem}.txt"
        
        # Destination paths
        dst_image = processed_dir / "images" / split / f"{filestem}.png"
        dst_label = processed_dir / "labels" / split / f"{filestem}.txt"
        
        # Copy image
        if src_image.exists():
            shutil.copy2(src_image, dst_image)
        else:
            missing_images += 1
            
        # Copy label
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            missing_labels += 1

    print("\nCopy Operation Complete.")
    if missing_images > 0:
        print(f"Warning: {missing_images} images were missing in raw directory.")
    if missing_labels > 0:
        print(f"Warning: {missing_labels} labels were missing in raw directory.")
        
    # Print summary
    print("\nProcessed Directory Summary:")
    for split in splits:
        img_count = len(list((processed_dir / "images" / split).glob("*.png")))
        lbl_count = len(list((processed_dir / "labels" / split).glob("*.txt")))
        print(f"{split.upper()} - Images: {img_count}, Labels: {lbl_count}")

if __name__ == "__main__":
    main()
