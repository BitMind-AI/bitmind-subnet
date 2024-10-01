import os
import subprocess

def run_all_py_scripts(directory):
    # List all files in the directory
    for filename in os.listdir(directory):
        # Check if the file ends with .py and is not this script itself
        if filename.endswith('.py') and filename != os.path.basename(__file__):
            # Full path of the python file
            filepath = os.path.join(directory, filename)
            print(f"Running {filename}...")
            
            # Run the script using subprocess
            subprocess.run(['python', filepath])

if __name__ == "__main__":
    # Run all python files in the current directory
    run_all_py_scripts(os.getcwd())