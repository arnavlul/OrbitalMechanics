import csv
import os

# --- CONFIGURATION ---
# List your downloaded JPL text files here
INPUT_FILES = [
    "./test_files/venus_test.txt",
]

# The final output file for your C++ code
OUTPUT_FILE = "./test_files/venus_test.csv"

def clean_jpl_file(input_path):
    cleaned_rows = []
    
    try:
        with open(input_path, 'r') as f:
            lines = f.readlines()

        # 1. Find the Start ($$SOE) and End ($$EOE) of the data
        start_idx = -1
        end_idx = -1
        
        for i, line in enumerate(lines):
            if "$$SOE" in line:
                start_idx = i + 1 # Start reading after this line
            if "$$EOE" in line:
                end_idx = i
                break
        
        if start_idx == -1 or end_idx == -1:
            print(f"Skipping {input_path}: Could not find $$SOE or $$EOE markers.")
            return []

        # 2. Process the data lines
        # JPL Format is usually: JD, Date, X, Y, Z, Vx, Vy, Vz, ...
        for line in lines[start_idx:end_idx]:
            # Remove whitespace and split by comma
            parts = [p.strip() for p in line.split(',')]
            
            # Filter out empty strings just in case
            parts = [p for p in parts if p]
            
            if len(parts) >= 8:
                # KEEP: Index 0 (Julian Date)
                # DROP: Index 1 (The human readable date string)
                # KEEP: Index 2 onwards (X, Y, Z, Vx, Vy, Vz...)
                
                # Create new row: [JD] + [X, Y, Z, Vx, Vy, Vz]
                # We specifically take parts[2:8] to get just position/velocity 
                # (ignoring any extra trailing columns like Light Time)
                new_row = [parts[0]] + parts[2:8]
                cleaned_rows.append(new_row)
                
        print(f"Processed {input_path}: Extracted {len(cleaned_rows)} rows.")
        return cleaned_rows

    except FileNotFoundError:
        print(f"Error: File {input_path} not found.")
        return []

def main():
    all_data = []
    
    # Process each file and collect data
    for fname in INPUT_FILES:
        file_data = clean_jpl_file(fname)
        all_data.extend(file_data)
        
    # Write everything to one big CSV
    if all_data:
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            # The C++ code expects NO HEADER, just numbers
            writer.writerows(all_data)
        print(f"\nSUCCESS! Saved {len(all_data)} total rows to '{OUTPUT_FILE}'.")
    else:
        print("\nNo data found. Check your filenames.")

if __name__ == "__main__":
    main()