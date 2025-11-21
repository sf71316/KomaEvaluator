import sys
import os

print("--- Python Interpreter ---")
print(f"sys.executable: {sys.executable}")
print("-" * 20)

print("--- Python Path (sys.path) ---")
for i, path in enumerate(sys.path):
    print(f"{i}: {path}")
print("-" * 20)

print("--- Checking for torch ---")
try:
    import torch
    print(f"Successfully imported torch. Version: {torch.__version__}")
    print(f"Torch location: {torch.__file__}")
except ImportError as e:
    print(f"Failed to import torch. Error: {e}")
print("-" * 20)

print("--- Checking for sklearn ---")
try:
    import sklearn
    print(f"Successfully imported sklearn. Version: {sklearn.__version__}")
    print(f"Sklearn location: {sklearn.__file__}")
except ImportError as e:
    print(f"Failed to import sklearn. Error: {e}")
print("-" * 20)

print("--- VENV Environment Variables ---")
print(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV')}")
print("-" * 20)
