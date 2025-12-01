class ModelTrainer:
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
    
    def train_svm_model(self, X_train, X_test, y_train, y_test, feature_name):
        """Train and evaluate SVM model"""
        print(f"\n{'='*60}")
        print(f"🚀 Training SVM with {feature_name} features")
        print(f"{'='*60}")
        
        # Create SVM pipeline with standardization
        svm_pipeline = make_pipeline(
            StandardScaler(),
            SVC(
                C=self.config.SVM_C,
                kernel=self.config.SVM_KERNEL,
                gamma=self.config.SVM_GAMMA,
                random_state=self.config.RANDOM_STATE,
                probability=True
            )
        )
        
        # Cross-validation
        print("📊 Performing cross-validation...")
        kfold = KFold(n_splits=self.config.K_FOLDS, shuffle=True, random_state=self.config.RANDOM_STATE)
        cv_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=kfold, scoring='accuracy')
        
        print(f"📈 Cross-validation scores: {cv_scores}")
        print(f"📈 Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model on entire training set
        print("🎯 Training final model...")
        svm_pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = svm_pipeline.predict(X_test)
        
        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=self.data_loader.classes, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n✅ Test Accuracy: {accuracy:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.data_loader.classes))
        
        return {
            'model': svm_pipeline,
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'confusion_matrix': cm,
            'cv_scores': cv_scores,
            'feature_name': feature_name
        }