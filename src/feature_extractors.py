class FeatureExtractor:
    def __init__(self, config):
        self.config = config
    
    def extract_hog_features(self, images):
        """Extract HOG features from images"""
        hog_features = []
        hog_images = []
        
        print("🔍 Extracting HOG features...")
        for img in tqdm(images):
            # Convert to grayscale for HOG
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Extract HOG features and visualization
            features, hog_image = feature.hog(
                gray,
                orientations=self.config.HOG_ORIENTATIONS,
                pixels_per_cell=self.config.HOG_PIXELS_PER_CELL,
                cells_per_block=self.config.HOG_CELLS_PER_BLOCK,
                visualize=True,
                block_norm='L2-Hys'
            )
            
            hog_features.append(features)
            hog_images.append(hog_image)
        
        return np.array(hog_features), hog_images
    
    def extract_lbp_features(self, images):
        """Extract LBP features from RGB images"""
        lbp_features = []
        lbp_images = []
        
        print("🔍 Extracting LBP features...")
        for img in tqdm(images):
            # Extract LBP for each channel and concatenate
            channel_features = []
            channel_lbp_images = []
            
            for channel in range(3):
                channel_img = img[:, :, channel]
                
                # Compute LBP
                lbp = local_binary_pattern(
                    channel_img,
                    self.config.LBP_N_POINTS,
                    self.config.LBP_RADIUS,
                    method=self.config.LBP_METHOD
                )
                
                # Compute histogram
                n_bins = self.config.LBP_N_POINTS + 2
                hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
                
                # Normalize histogram
                hist = hist.astype("float")
                hist /= (hist.sum() + 1e-6)  # Avoid division by zero
                
                channel_features.extend(hist)
                channel_lbp_images.append(lbp)
            
            lbp_features.append(channel_features)
            lbp_images.append(channel_lbp_images)
        
        return np.array(lbp_features), lbp_images