# eda.py
import pandas as pd
import os

print("="*80)
print("FILE STRUCTURE ANALYSIS")
print("="*80)

files = {
    'train': r'data\manzoni_dev_tokens.csv',
    'dev': r'data\manzoni_train_tokens.csv',
    'ood': r'data\OOD_test.csv'
}

for name, filepath in files.items():
    print(f"\n{name.upper()}: {filepath}")
    print("-" * 80)
    
    if not os.path.exists(filepath):
        print(f"  File not found!")
        continue
    
    # Try multiple reading methods
    try:
        df = pd.read_csv(filepath, nrows=10)
        print(f"  Standard read successful")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"  Shape (first 10 rows): {df.shape}")
        print(f"  Dtypes:\n{df.dtypes}")
        print(f"\n  First 5 rows:")
        print(df.head())
    except Exception as e:
        print(f"  Standard read failed: {e}")
        
        # Try reading first few lines manually
        print(f"\n  Raw first 10 lines:")
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"    Line {i}: {repr(line.strip())}")
    
    # Check full file stats
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"\n  Total lines: {len(lines)}")
        
        # Check for inconsistent column counts
        if lines:
            header_cols = len(lines[0].strip().split(','))
            print(f"  Header columns: {header_cols}")
            
            inconsistent = []
            for i, line in enumerate(lines[1:], 1):
                cols = len(line.strip().split(','))
                if cols != header_cols:
                    inconsistent.append((i, cols))
                if len(inconsistent) >= 5:
                    break
            
            if inconsistent:
                print(f"  Found {len(inconsistent)} inconsistent rows:")
                for line_num, col_count in inconsistent[:5]:
                    print(f"    Line {line_num}: {col_count} columns")
    except Exception as e:
        print(f"  Error checking file stats: {e}")

print("\n" + "="*80)
print("LABEL DISTRIBUTION")
print("="*80)

for name, filepath in files.items():
    if not os.path.exists(filepath):
        continue
    
    try:
        df = pd.read_csv(filepath)
        if 'label' in df.columns:
            print(f"\n{name.upper()}:")
            print(df['label'].value_counts().to_dict())
            print(f"  Class balance: {df['label'].value_counts(normalize=True).to_dict()}")
    except:
        pass