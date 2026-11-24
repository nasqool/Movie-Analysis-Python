# check_imports.py
# Tries to load each library one by one and reports which ones work.

import importlib

for name in ["numpy", "scipy", "matplotlib", "sklearn", "sqlite3", "streamlit", "pandas"]:
    try:
        importlib.import_module(name)
        print(name, "-> OK")
    except Exception as error:
        print(name, "-> FAILED:", str(error)[:100])
        