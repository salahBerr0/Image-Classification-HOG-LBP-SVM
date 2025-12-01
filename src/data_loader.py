class DataLoader:
    def __init__(self, config):
        self.config = config
        self.classes = ['city', 'face', 'green', 'office', 'sea']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.classes)}
    
    def load_images_from_folder(self, folder_path):
        """Load images from a specific folder"""
        images = []
        labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(folder_path, class_name)
            if not os.path.exists(class_path):
                print(f"⚠️ Warning: {class_path} does not exist")
                continue
                
            print(f"📁 Loading {class_name} images...")
            class_images = []
            for img_file in tqdm(os.listdir(class_path)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, img_file)
                    try:
                        # Read image
                        img = cv2.imread(img_path)
                        if img is not None:
                            # Resize and convert color format
                            img = cv2.resize(img, self.config.IMG_SIZE)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            class_images.append(img)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
            
            images.extend(class_images)
            labels.extend([self.class_to_idx[class_name]] * len(class_images))
            print(f"✅ Loaded {len(class_images)} {class_name} images")
        
        return np.array(images), np.array(labels)
    
    def load_dataset(self):
        """Load both train and test datasets"""
        train_path = os.path.join(self.config.DATASET_PATH, 'train')
        test_path = os.path.join(self.config.DATASET_PATH, 'test')
        
        print("🚀 Loading training dataset...")
        X_train, y_train = self.load_images_from_folder(train_path)
        
        print("🚀 Loading test dataset...")
        X_test, y_test = self.load_images_from_folder(test_path)
        
        print(f"\n📊 Dataset Summary:")
        print(f"Training set: {X_train.shape[0]} images")
        print(f"Test set: {X_test.shape[0]} images")
        print(f"Image shape: {X_train[0].shape}")
        print(f"Classes: {self.classes}")
        
        return X_train, X_test, y_train, y_test