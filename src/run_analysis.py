#!/usr/bin/env python3
"""
Main script to run the complete image classification analysis
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config import Config
from data_loader import DataLoader
from feature_extractors import FeatureExtractor
from classifiers import ModelTrainer
from visualizer import Visualizer
import joblib

def main():
    print("="*70)
    print("IMAGE CLASSIFICATION: HOG vs LBP with SVM")
    print("="*70)
    
    # Initialize components
    config = Config()
    data_loader = DataLoader(config)
    feature_extractor = FeatureExtractor(config)
    
    # Load data
    print("\n1. Loading dataset...")
    X_train, X_test, y_train, y_test = data_loader.load_dataset()
    
    # Extract features
    print("\n2. Extracting features...")
    X_train_hog, hog_train_vis = feature_extractor.extract_hog_features(X_train)
    X_test_hog, hog_test_vis = feature_extractor.extract_hog_features(X_test)
    
    X_train_lbp, lbp_train_vis = feature_extractor.extract_lbp_features(X_train)
    X_test_lbp, lbp_test_vis = feature_extractor.extract_lbp_features(X_test)
    
    # Initialize trainer and visualizer
    trainer = ModelTrainer(config, data_loader)
    visualizer = Visualizer(data_loader)
    
    # Visualize features
    print("\n3. Visualizing features...")
    visualizer.visualize_features(X_train, hog_train_vis, lbp_train_vis)
    
    # Train HOG model
    print("\n4. Training HOG + SVM model...")
    hog_results = trainer.train_svm_model(X_train_hog, X_test_hog, y_train, y_test, "HOG")
    visualizer.plot_confusion_matrix(hog_results['confusion_matrix'], "HOG + SVM")
    joblib.dump(hog_results['model'], 'results/models/hog_svm_model.pkl')
    
    # Train LBP model
    print("\n5. Training LBP + SVM model...")
    lbp_results = trainer.train_svm_model(X_train_lbp, X_test_lbp, y_train, y_test, "LBP")
    visualizer.plot_confusion_matrix(lbp_results['confusion_matrix'], "LBP + SVM")
    joblib.dump(lbp_results['model'], 'results/models/lbp_svm_model.pkl')
    
    # Compare results
    print("\n6. Comparing results...")
    df_comparison = visualizer.compare_results(hog_results, lbp_results)
    
    # Save results
    results = {
        'hog_results': hog_results,
        'lbp_results': lbp_results,
        'comparison': df_comparison.to_dict()
    }
    joblib.dump(results, 'results/classification_results.pkl')
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"   Models saved to: results/models/")
    print(f"   Results saved to: results/classification_results.pkl")
    print("="*70)

if __name__ == "__main__":
    main()