# scripts/download_data.py
"""
Dataset download helper script.
Run: python scripts/download_data.py
"""

import os
import zipfile
from pathlib import Path

def download_instructions():
    """Print download instructions"""
    
    print("=" * 60)
    print("📥 HOG/LBP DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n📁 Dataset will be downloaded to:", data_dir.absolute())
    
    print("\n🔗 Download from Google Drive:")
    print("   1. Visit: https://drive.google.com/drive/folders/1cXLRw66bPsqLzqptwTqUfXlwM-bBUb7I?usp=sharing")
    print("   2. Download the 'hog_lbp_Dataset' folder")
    print("   3. Extract it to:", data_dir.absolute())
    
    print("\n📂 Expected structure after download:")
    print(f"   {data_dir}/")
    print("   ├── train/")
    print("   │   ├── city/")
    print("   │   ├── face/")
    print("   │   ├── green/")
    print("   │   ├── office/")
    print("   │   └── sea/")
    print("   └── test/")
    print("       ├── city/")
    print("       ├── face/")
    print("       ├── green/")
    print("       ├── office/")
    print("       └── sea/")
    
    print("\n✅ After downloading, run:")
    print("   jupyter notebook notebooks/main_analysis.ipynb")
    print("=" * 60)

if __name__ == "__main__":
    download_instructions()
