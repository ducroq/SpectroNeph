import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RobustHierarchicalClassifier:
    """
    Improved hierarchical classifier with proper validation and realistic feature selection
    """
    
    def __init__(self, test_size=0.3, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.level1_classifier = None
        self.level2_classifier = None  
        self.level3_classifier = None
        self.scalers = {}
        self.imputers = {}
        
        # More conservative feature sets based on physical interpretability
        self.level1_features = [
            'Clear', 'F8', 'F6', 'F7',  # Raw intensities (particle scattering)
            'total_sum', 'long_sum',     # Overall signal strength
            'short_fraction', 'long_fraction'  # Spectral balance
        ]
        
        self.level2_features = [
            'spectral_centroid',         # Spectral shift (main agglutination indicator)
            'spectral_width',            # Spectral broadening
            'short_long_sum_ratio',      # Blue-red balance  
            'F3', 'F5',                  # Specific channels
            'center_edges_sum_ratio'     # Spectral shape
        ]
        
        self.level3_features = [
            'spectral_centroid', 'spectral_skewness',  # Spectral shape
            'short_long_sum_ratio', 'long_fraction',   # Color balance
            'F3', 'F5', 'F7',                          # Key channels
            'mid_fraction'                             # Green region
        ]
        
        # Label mappings - Updated for your salt concentration experiment
        # Since you only have salt conditions, Level 1 becomes low vs high salt
        self.particle_labels = {
            'low_salt': ['salt_0_mg_per_mL'],  # 0 mg/mL = low/no salt
            'high_salt': ['salt_3.6_mg_per_mL', 'salt_7.2_mg_per_mL']  # Higher concentrations
        }
        
        self.agglutination_labels = {
            'no_agglutination': ['salt_0_mg_per_mL'],  # 0 mg/mL salt = no agglutination (control)
            'agglutination': ['salt_3.6_mg_per_mL', 'salt_7.2_mg_per_mL']  # Higher salt = agglutination
        }
    
    def prepare_data_robust(self, df):
        """Prepare data with proper validation split ensuring class balance"""
        # Remove 'skip' samples and unknown samples
        df_clean = df[(df['label'] != 'skip') & (df['label'] != 'unknown')].copy()
        
        print(f"After removing skip/unknown: {len(df_clean)} samples")
        print(f"Label distribution after cleaning:")
        print(df_clean['label'].value_counts())
        
        # Get feature columns (exclude ALL metadata including new salt columns)
        metadata_cols = [
            'label', 'filename', 'detailed_label', 'bead_type', 'buffer_type', 
            'condition', 'test_nr', 'timestamp', 'has_salt', 'salt_concentration', 
            'salt_unit'  # Added the new salt columns to exclusion list
        ]
        
        feature_cols = [col for col in df_clean.columns 
                       if col not in metadata_cols 
                       and not col.startswith('Unnamed')]
        
        print(f"Using {len(feature_cols)} features for analysis")
        print(f"Excluded metadata columns: {[col for col in metadata_cols if col in df_clean.columns]}")
        
        # Check if we have numeric features only
        numeric_features = df_clean[feature_cols].select_dtypes(include=[np.number]).columns
        non_numeric_features = df_clean[feature_cols].select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_features) > 0:
            print(f"Warning: Found non-numeric features that will be excluded: {list(non_numeric_features)}")
            feature_cols = list(numeric_features)
            print(f"Using {len(feature_cols)} numeric features only")
        
        X = df_clean[feature_cols]
        y = df_clean['label']
        
        if len(df_clean) == 0:
            raise ValueError("No valid samples after cleaning! Check your label mapping.")
        
        # Ensure all classes appear in both train and test sets
        # Use stratified split as primary method
        try:
            train_idx, test_idx = train_test_split(
                range(len(X)), test_size=self.test_size, 
                stratify=y,
                random_state=self.random_state
            )
            print("Using stratified split to ensure class balance")
        except ValueError as e:
            print(f"Stratified split failed ({e}), using simple random split")
            train_idx, test_idx = train_test_split(
                range(len(X)), test_size=self.test_size, 
                random_state=self.random_state
            )
        
        # Create train/test splits
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Verify we have all classes in training set
        train_classes = set(y_train.unique())
        test_classes = set(y_test.unique())
        all_classes = set(y.unique())
        
        print(f"All classes: {sorted(all_classes)}")
        print(f"Training classes: {sorted(train_classes)}")
        print(f"Test classes: {sorted(test_classes)}")
        
        # Handle missing values and scaling
        self.imputers['main'] = SimpleImputer(strategy='median')
        self.scalers['main'] = StandardScaler()
        
        X_train_processed = self.scalers['main'].fit_transform(
            self.imputers['main'].fit_transform(X_train)
        )
        X_test_processed = self.scalers['main'].transform(
            self.imputers['main'].transform(X_test)
        )
        
        # Convert back to DataFrame to maintain column names
        X_train_processed = pd.DataFrame(X_train_processed, columns=feature_cols, index=train_idx)
        X_test_processed = pd.DataFrame(X_test_processed, columns=feature_cols, index=test_idx)
        
        X_train_processed['label'] = y_train.values
        X_test_processed['label'] = y_test.values
        
        return X_train_processed, X_test_processed
    
    def create_level1_labels(self, labels):
        """Convert to low_salt vs high_salt"""
        level1_labels = []
        for label in labels:
            if label in self.particle_labels['low_salt']:
                level1_labels.append('low_salt')
            elif label in self.particle_labels['high_salt']:
                level1_labels.append('high_salt')
            else:
                level1_labels.append('unknown')
        return np.array(level1_labels)
    
    def create_level2_labels(self, labels):
        """Convert to agglutination labels"""
        level2_labels = []
        for label in labels:
            if label in self.agglutination_labels['no_agglutination']:
                level2_labels.append('no_agglutination')
            elif label in self.agglutination_labels['agglutination']:
                level2_labels.append('agglutination')
            else:
                level2_labels.append('unknown')
        return np.array(level2_labels)
    
    def train_level1(self, train_df, algorithm='rf'):
        """Train Level 1 with proper cross-validation and moderate regularization"""
        print("=== Training Level 1: Low vs High Salt Detection ===")
        
        # Prepare labels and features
        y_level1 = self.create_level1_labels(train_df['label'])
        valid_mask = y_level1 != 'unknown'
        
        print(f"Level 1 total samples: {len(y_level1)}")
        print(f"Level 1 valid samples: {valid_mask.sum()}")
        
        # Ensure we have all required features
        available_features = [f for f in self.level1_features if f in train_df.columns]
        if len(available_features) != len(self.level1_features):
            missing = set(self.level1_features) - set(available_features)
            print(f"Warning: Missing features for Level 1: {missing}")
        
        # Always set the features used, even if training fails
        self.level1_features_used = available_features
        
        if valid_mask.sum() == 0:
            print("ERROR: No valid samples for Level 1 training!")
            return None, None
        
        X = train_df[valid_mask][available_features].values
        y = y_level1[valid_mask]
        
        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        print(f"Level 1 class distribution: {dict(zip(unique, counts))}")
        
        if len(unique) < 2:
            print("ERROR: Need at least 2 classes for Level 1 training!")
            print("Creating a dummy classifier that predicts the majority class...")
            # Create a dummy classifier for single-class scenarios
            from sklearn.dummy import DummyClassifier
            self.level1_classifier = DummyClassifier(strategy='most_frequent')
            self.level1_classifier.fit(X, y)
            return X, y
        
        # Choose algorithm with moderate regularization
        if algorithm == 'rf':
            self.level1_classifier = RandomForestClassifier(
                n_estimators=10, 
                max_depth=12,
                min_samples_split=15,
                min_samples_leaf=8,
                max_features='sqrt',
                random_state=self.random_state
            )
        elif algorithm == 'lr':
            self.level1_classifier = LogisticRegression(
                random_state=self.random_state, max_iter=1000, C=0.5
            )
        
        # Train with cross-validation only if we have enough samples
        if len(X) > 10 and len(unique) >= 2:
            cv = StratifiedKFold(n_splits=min(5, len(X)//10), shuffle=True, 
                                random_state=self.random_state)
            cv_scores = cross_val_score(self.level1_classifier, X, y, cv=cv, scoring='balanced_accuracy')
            print(f"Level 1 CV Balanced Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model
        self.level1_classifier.fit(X, y)
        
        return X, y
    
    def train_level2(self, train_df, algorithm='rf'):
        """Train Level 2 on all samples for agglutination detection"""
        print("\n=== Training Level 2: Agglutination Detection ===")
        
        # Use all samples for Level 2 (not just particles, since all your samples have particles)
        all_samples = train_df
        
        if len(all_samples) == 0:
            print("No samples found for Level 2 training!")
            self.level2_classifier = None
            return None, None
        
        # Prepare labels
        y_level2 = self.create_level2_labels(all_samples['label'])
        valid_mask = y_level2 != 'unknown'
        
        available_features = [f for f in self.level2_features if f in train_df.columns]
        if len(available_features) != len(self.level2_features):
            missing = set(self.level2_features) - set(available_features)
            print(f"Warning: Missing features for Level 2: {missing}")
        
        # Always set the features used
        self.level2_features_used = available_features
        
        X = all_samples[valid_mask][available_features].values
        y = y_level2[valid_mask]
        
        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        print(f"Level 2 class distribution: {dict(zip(unique, counts))}")
        
        if len(unique) < 2:
            print("Insufficient classes for Level 2 training!")
            from sklearn.dummy import DummyClassifier
            self.level2_classifier = DummyClassifier(strategy='most_frequent')
            self.level2_classifier.fit(X, y)
            return X, y
        
        # HEAVY REGULARIZATION to prevent overfitting
        if algorithm == 'rf':
            self.level2_classifier = RandomForestClassifier(
                n_estimators=50,
                max_depth=4,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif algorithm == 'lr':
            self.level2_classifier = LogisticRegression(
                random_state=self.random_state, 
                max_iter=1000, 
                C=0.01,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced'
            )
        
        # Cross-validation with small dataset handling
        if len(X) > 10:
            cv = StratifiedKFold(n_splits=min(3, len(X)//5), shuffle=True, 
                                random_state=self.random_state)
            cv_scores = cross_val_score(self.level2_classifier, X, y, cv=cv, scoring='balanced_accuracy')
            print(f"Level 2 CV Balanced Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model
        self.level2_classifier.fit(X, y)
        
        return X, y
    
    def train_level3(self, train_df, algorithm='rf'):
        """Train Level 3 multi-class classifier with regularization"""
        print("\n=== Training Level 3: Multi-class Classification ===")
        
        available_features = [f for f in self.level3_features if f in train_df.columns]
        if len(available_features) != len(self.level3_features):
            missing = set(self.level3_features) - set(available_features)
            print(f"Warning: Missing features for Level 3: {missing}")
        
        # Always set the features used
        self.level3_features_used = available_features
        
        X = train_df[available_features].values
        y = train_df['label'].values
        
        # Check class distribution
        unique, counts = np.unique(y, return_counts=True)
        print(f"Level 3 class distribution:")
        for label, count in zip(unique, counts):
            print(f"  {label}: {count}")
        
        # Choose algorithm with appropriate regularization
        if algorithm == 'rf':
            self.level3_classifier = RandomForestClassifier(
                n_estimators=100, 
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                random_state=self.random_state,
                class_weight='balanced_subsample'
            )
        elif algorithm == 'lr':
            self.level3_classifier = LogisticRegression(
                random_state=self.random_state, max_iter=1000, C=0.3,
                class_weight='balanced',
                multi_class='ovr'
            )
        
        # Cross-validation
        if len(X) > 10:
            cv = StratifiedKFold(n_splits=5, shuffle=True, 
                                random_state=self.random_state)
            cv_scores = cross_val_score(self.level3_classifier, X, y, cv=cv, scoring='balanced_accuracy')
            print(f"Level 3 CV Balanced Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Train final model
        self.level3_classifier.fit(X, y)
        
        return X, y
    
    def predict_hierarchical(self, X_sample):
        """Hierarchical prediction with proper feature handling"""
        results = {}
        
        # Level 1: Particle detection
        X1 = X_sample[self.level1_features_used].values.reshape(1, -1)
        level1_pred = self.level1_classifier.predict(X1)[0]
        level1_proba = self.level1_classifier.predict_proba(X1)[0]
        level1_conf = max(level1_proba)
        
        results['level1'] = {
            'prediction': level1_pred,
            'confidence': level1_conf,
            'probabilities': level1_proba
        }
        
        # Level 2: Agglutination detection (for all samples now)
        if self.level2_classifier is not None:
            X2 = X_sample[self.level2_features_used].values.reshape(1, -1)
            level2_pred = self.level2_classifier.predict(X2)[0]
            level2_proba = self.level2_classifier.predict_proba(X2)[0]
            level2_conf = max(level2_proba)
            
            results['level2'] = {
                'prediction': level2_pred,
                'confidence': level2_conf,
                'probabilities': level2_proba
            }
        else:
            results['level2'] = {
                'prediction': 'not_applicable',
                'confidence': None,
                'probabilities': None
            }
        
        # Level 3: Multi-class classification
        X3 = X_sample[self.level3_features_used].values.reshape(1, -1)
        level3_pred = self.level3_classifier.predict(X3)[0]
        level3_proba = self.level3_classifier.predict_proba(X3)[0]
        level3_conf = max(level3_proba)
        
        results['level3'] = {
            'prediction': level3_pred,
            'confidence': level3_conf,
            'probabilities': level3_proba
        }
        
        return results
    
    def evaluate_on_test_set(self, test_df):
        """Evaluate on held-out test set"""
        print("\n" + "="*50)
        print("EVALUATION ON TEST SET")
        print("="*50)
        
        results = []
        for idx, row in test_df.iterrows():
            prediction = self.predict_hierarchical(row)
            results.append({
                'true_label': row['label'],
                'level1_pred': prediction['level1']['prediction'],
                'level1_conf': prediction['level1']['confidence'],
                'level2_pred': prediction['level2']['prediction'],
                'level2_conf': prediction['level2']['confidence'],
                'level3_pred': prediction['level3']['prediction'],
                'level3_conf': prediction['level3']['confidence']
            })
        
        results_df = pd.DataFrame(results)
        
        # Calculate balanced accuracies
        true_level1 = self.create_level1_labels(results_df['true_label'])
        level1_bal_acc = balanced_accuracy_score(true_level1, results_df['level1_pred'])
        
        # Level 2 accuracy (all samples now)
        level2_bal_acc = None
        if self.level2_classifier is not None:
            true_level2 = self.create_level2_labels(results_df['true_label'])
            valid_level2 = results_df['level2_pred'] != 'not_applicable'
            if valid_level2.sum() > 0:
                level2_bal_acc = balanced_accuracy_score(
                    true_level2[valid_level2], 
                    results_df[valid_level2]['level2_pred']
                )
        
        # Level 3 accuracy
        level3_bal_acc = balanced_accuracy_score(results_df['true_label'], results_df['level3_pred'])
        
        print(f"Level 1 Balanced Accuracy: {level1_bal_acc:.3f}")
        if level2_bal_acc is not None:
            print(f"Level 2 Balanced Accuracy: {level2_bal_acc:.3f}")
        else:
            print("Level 2: Not applicable (no particle samples or insufficient classes)")
        print(f"Level 3 Balanced Accuracy: {level3_bal_acc:.3f}")
        
        return results_df

def main():
    """Main training pipeline with robust validation"""
    
    # Load data - CORRECTED FILENAME
    df = pd.read_csv('side_scatter_analysis_features.csv')
    print(f"Loaded {len(df)} samples from side_scatter_analysis_features.csv")
    
    # Debug: Check what we loaded
    print("\nAvailable columns:")
    print([col for col in df.columns if col in ['label', 'filename', 'bead_type', 'buffer_type', 'condition']])
    
    print(f"\nLabel distribution:")
    print(df['label'].value_counts())
    
    # Initialize classifier
    hc = RobustHierarchicalClassifier(test_size=0.3, random_state=42)
    
    # Robust data preparation with proper train/test split
    try:
        train_df, test_df = hc.prepare_data_robust(df)
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return None, None, None, None
    
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train label distribution:")
    print(train_df['label'].value_counts().to_string())
    print(f"\nTest label distribution:")
    print(test_df['label'].value_counts().to_string())
    
    # Train all levels
    print("\n" + "="*50)
    print("TRAINING ROBUST HIERARCHICAL CLASSIFIER")
    print("="*50)
    
    algorithm = 'rf'  # Can change to 'lr' for logistic regression
    
    try:
        hc.train_level1(train_df, algorithm=algorithm)
        hc.train_level2(train_df, algorithm=algorithm)
        hc.train_level3(train_df, algorithm=algorithm)
    except Exception as e:
        print(f"Error in training: {e}")
        return hc, None, train_df, test_df
    
    # Evaluate on test set
    try:
        results = hc.evaluate_on_test_set(test_df)
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return hc, None, train_df, test_df
    
    # Save results
    results.to_csv('robust_hierarchical_results.csv', index=False)
    print(f"\nResults saved to 'robust_hierarchical_results.csv'")
    
    # Show first few results for verification
    print("\nFirst 5 prediction results:")
    print(results[['true_label', 'level1_pred', 'level1_conf', 'level3_pred', 'level3_conf']].head().to_string())
    
    return hc, results, train_df, test_df

if __name__ == "__main__":
    # Run the main function with better error handling
    try:
        classifier, results, train_df, test_df = main()
        
        if results is not None:
            print("\n" + "="*50)
            print("🎯 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*50)
            print("📁 Files created:")
            print("  - robust_hierarchical_results.csv (prediction results)")
            print("\n✅ Basic hierarchical classification completed!")
        else:
            print("\n❌ Analysis failed - check error messages above")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        
        # Debug: Try to load the CSV and check its structure
        try:
            print("\nDebugging CSV structure...")
            debug_df = pd.read_csv('side_scatter_analysis_features.csv')
            print(f"✓ CSV loads successfully: {debug_df.shape}")
            print(f"✓ Sample labels: {debug_df['label'].unique()}")
            print(f"✓ Label counts: {debug_df['label'].value_counts()}")
            print(f"✓ Key columns present: {[col for col in debug_df.columns if col in ['F1', 'F2', 'F3', 'Clear', 'total_sum']]}")
        except Exception as debug_e:
            print(f"❌ Cannot load CSV: {debug_e}")