import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\d+)', s)]

files = os.listdir('matches_visualization')
files = [f for f in files if f.endswith('.jpg')]

# Natural Sort (Windows Explorer)
files_nat = sorted(files, key=natural_sort_key)
print("--- Natural Sort (First 15) ---")
for i, f in enumerate(files_nat[:15]):
    inliers = f.split('_')[1]
    print(f"{i+1}: Inliers={inliers} | {f}")

print("\n")
