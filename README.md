# wine_quality_RFclassification
Here's a small project to predict wine quality from a chemical characteristics.

# Data
The dataset can be found on https://archive.ics.uci.edu/ml/datasets/wine+quality.

# Methodology

I'm using Random Forest Classification.
Random Forest is an ensemble of Decision Trees, which perform classification based on rules at each "leaf". 
Since Decision Trees are deterministic, using an ensemble permits a probablistic approach, so we can determine how good that prediction was.
It also permits us to compute how 'important' each of the wine chemical features was to the quality.

# Python Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.model_selection import cross_validate
