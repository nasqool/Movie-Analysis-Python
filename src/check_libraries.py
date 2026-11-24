# check_libraries.py
# Confirms that every library we need is installed, and shows its version.

import sqlite3

import joblib
import matplotlib
import numpy
import pandas
import seaborn
import sklearn
import streamlit

print("pandas       :", pandas.__version__)
print("numpy        :", numpy.__version__)
print("matplotlib   :", matplotlib.__version__)
print("seaborn      :", seaborn.__version__)
print("scikit-learn :", sklearn.__version__)
print("streamlit    :", streamlit.__version__)
print("joblib       :", joblib.__version__)
print("SQLite engine:", sqlite3.sqlite_version)