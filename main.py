import os
import subprocess
import sys

MODE = os.getenv("RUN_MODE", "local")

print(f"Running in {MODE} mode")

PYTHON = sys.executable  # current python

if MODE == "databricks":
    subprocess.run([PYTHON, "src/pyspark.py"], check=True)
else:
    subprocess.run([PYTHON, "src/pyspark.py"], check=True)