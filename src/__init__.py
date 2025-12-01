# These imports go in EVERY .py file that needs them:

# In data_loader.py
import os
import cv2
import numpy as np
from tqdm import tqdm

# In feature_extractors.py  
import numpy as np
import cv2
from skimage import feature
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# In classifiers.py
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# In visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np