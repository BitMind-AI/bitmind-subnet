import subprocess

output_file = "git_diff.txt"

try:
    result = subprocess.run(["git", "diff"], capture_output=True, text=True, check=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)
    print(f"Git diff written to {output_file}")
except subprocess.CalledProcessError as e:
    print(f"Error running git diff: {e}") 