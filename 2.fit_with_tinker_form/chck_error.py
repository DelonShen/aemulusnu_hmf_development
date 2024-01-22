import os
import glob

# Set the directory where your logs are stored
log_directory = 'logs/'

# Use glob to match all .err files with the pattern
error_files = glob.glob(os.path.join(log_directory, '2024-01-17*.err'))

# Loop through each file and check for the word 'Traceback'
for file_path in error_files:
    print(file_path)
    with open(file_path, 'r') as file:
        for line in file:
            if 'Traceback' in line or 'CANCELLED' in line:
                print(f"Python error found in {file_path}:")
                print(line)
                # Optionally, print the entire traceback by continuing to print lines until the end of the traceback
                for error_line in file:
                    print(error_line.strip())
                    if error_line.strip() == '':
                        break
                print("-" * 40)  # Separator for readability
                break  # Stop after the first traceback to avoid duplicate prints if there's more than one traceback
