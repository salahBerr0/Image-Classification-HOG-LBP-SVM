class Config:
    """Configuration settings for the entire project"""
    # Update this path to your local dataset location
    DATASET_PATH = 'data/'  # Changed from Google Drive path
    
    # HOG parameters
    HOG_ORIENTATIONS = 9
    HOG_PIXELS_PER_CELL = (8, 8)
    HOG_CELLS_PER_BLOCK = (2, 2)
    
    # LBP parameters  
    LBP_RADIUS = 3
    LBP_N_POINTS = 24
    LBP_METHOD = 'uniform'
    
    # SVM parameters
    SVM_C = 1.0
    SVM_KERNEL = 'rbf'  # Options: 'rbf', 'sigmoid'
    SVM_GAMMA = 'scale'
    
    # Training parameters
    K_FOLDS = 5
    RANDOM_STATE = 42
    IMG_SIZE = (128, 128)