class Visualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
    
    def plot_sample_images(self, images, labels, num_samples=5):
        """Plot sample images from each class"""
        fig, axes = plt.subplots(len(self.data_loader.classes), num_samples, figsize=(15, 12))
        
        for class_idx, class_name in enumerate(self.data_loader.classes):
            # Get indices of current class
            class_indices = np.where(labels == class_idx)[0]
            
            # Select random samples
            if len(class_indices) > 0:
                sample_indices = np.random.choice(class_indices, num_samples, replace=False)
                
                for i, idx in enumerate(sample_indices):
                    axes[class_idx, i].imshow(images[idx])
                    axes[class_idx, i].set_title(f'{class_name}\nIdx: {idx}')
                    axes[class_idx, i].axis('off')
            else:
                for i in range(num_samples):
                    axes[class_idx, i].axis('off')
                    axes[class_idx, i].set_title(f'{class_name}\nNo images')
        
        plt.tight_layout()
        plt.suptitle('Sample Images from Each Class', fontsize=16, y=1.02)
        plt.show()
    
    def visualize_features(self, original_images, hog_images, lbp_images, num_examples=3):
        """Visualize original images with HOG and LBP features"""
        print("🖼️ Visualizing HOG and LBP Features...")
        fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))
        
        for i in range(num_examples):
            # Original image
            axes[i, 0].imshow(original_images[i])
            axes[i, 0].set_title(f'Original Image\nClass: {self.data_loader.idx_to_class[y_train[i]]}')
            axes[i, 0].axis('off')
            
            # HOG image
            axes[i, 1].imshow(hog_images[i], cmap='gray')
            axes[i, 1].set_title('HOG Features')
            axes[i, 1].axis('off')
            
            # LBP visualization (first channel)
            axes[i, 2].imshow(lbp_images[i][0], cmap='gray')
            axes[i, 2].set_title('LBP Features (Channel 0)')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm, title):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.data_loader.classes, 
                   yticklabels=self.data_loader.classes)
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

    def compare_results(self, hog_results, lbp_results):
        """Compare HOG and LBP performance"""
        print(f"\n{'='*60}")
        print("📊 PERFORMANCE COMPARISON: HOG vs LBP")
        print(f"{'='*60}")
        
        # Create comparison table
        import pandas as pd
        comparison_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Mean', 'CV Std'],
            'HOG+SVM': [
                hog_results['accuracy'],
                hog_results['precision'],
                hog_results['recall'],
                hog_results['f1'],
                hog_results['cv_scores'].mean(),
                hog_results['cv_scores'].std()
            ],
            'LBP+SVM': [
                lbp_results['accuracy'],
                lbp_results['precision'],
                lbp_results['recall'],
                lbp_results['f1'],
                lbp_results['cv_scores'].mean(),
                lbp_results['cv_scores'].std()
            ]
        }
        
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.round(4).to_string(index=False))
        
        # Visual comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        hog_values = [hog_results['accuracy'], hog_results['precision'], 
                     hog_results['recall'], hog_results['f1']]
        lbp_values = [lbp_results['accuracy'], lbp_results['precision'], 
                     lbp_results['recall'], lbp_results['f1']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        
        # Bar chart comparison
        bars1 = ax[0].bar(x - width/2, hog_values, width, label='HOG+SVM', alpha=0.8, color='skyblue')
        bars2 = ax[0].bar(x + width/2, lbp_values, width, label='LBP+SVM', alpha=0.8, color='lightcoral')
        
        ax[0].set_xlabel('Metrics')
        ax[0].set_ylabel('Scores')
        ax[0].set_title('Performance Comparison: HOG+SVM vs LBP+SVM')
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(metrics)
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax[0].annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax[0].annotate(f'{height:.3f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontweight='bold')
        
        # Cross-validation comparison
        cv_data = [hog_results['cv_scores'], lbp_results['cv_scores']]
        ax[1].boxplot(cv_data, labels=['HOG+SVM', 'LBP+SVM'])
        ax[1].set_title('Cross-Validation Accuracy Distribution')
        ax[1].set_ylabel('Accuracy')
        ax[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Determine best method
        if hog_results['accuracy'] > lbp_results['accuracy']:
            best_method = "HOG+SVM"
            improvement = hog_results['accuracy'] - lbp_results['accuracy']
            print(f"\n🎯 BEST PERFORMING METHOD: {best_method}")
            print(f"📈 Improvement over LBP+SVM: {improvement:.4f} ({improvement*100:.2f}%)")
        else:
            best_method = "LBP+SVM"
            improvement = lbp_results['accuracy'] - hog_results['accuracy']
            print(f"\n🎯 BEST PERFORMING METHOD: {best_method}")
            print(f"📈 Improvement over HOG+SVM: {improvement:.4f} ({improvement*100:.2f}%)")
        
        return df_comparison