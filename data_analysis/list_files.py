import sys
from pathlib import Path

directory_path = r"C:\Users\scbry\OneDrive - HAN\Research\TKI Onsite monitoring and Removal of pharmaceutical emissions\03 experiments\250604 WFBR experiments\perpendicular_spectrometer"
pattern = '*.csv'

path = Path(directory_path)
filenames = [f.name for f in path.glob(pattern) if not 'summary' in f.name]

print("Files matching pattern:", pattern)
for filename in filenames:
    print(filename)
