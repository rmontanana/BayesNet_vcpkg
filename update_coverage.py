import subprocess
import os
import sys

readme_file = "README.md"
print("Updating coverage...")
# Generate badge line
output = subprocess.check_output(
    "lcov --summary " + sys.argv[1] + "/coverage.info|cut -d' ' -f4 |head -2|"
    "tail -1",
    shell=True,
)
percentage = output.decode("utf-8").strip().replace(".", ",")
coverage_line = (
    f"![Static Badge](https://img.shields.io/badge/Coverage-{percentage}25-green)"
)
# Update README.md
with open(readme_file, "r") as f:
    lines = f.readlines()
with open(readme_file, "w") as f:
    for line in lines:
        if "Coverage" in line:
            f.write(coverage_line + "\n")
        else:
            f.write(line)
