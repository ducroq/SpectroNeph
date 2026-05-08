import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.inspection import permutation_importance
import joblib
import warnings
warnings.filterwarnings('ignore')

class HierarchicalClassifierInterpreter:
    """
    Fixed interpretability analysis for hierarchical classifiers
    Handles different feature sets for each level correctly
    """
    
    def __init__(self, models_dict, feature_names_dict, test_data, test_labels):
        """
        Initialize with trained models and test data
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary containing trained models {'level1': model1, 'level2': model2, 'level3': model3}
        feature_names_dict : dict
            Dictionary with feature names for each level {'level1': [...], 'level2': [...], 'level3': [...]}
        test_data : pandas.DataFrame
            Test feature data (full feature set)
        test_labels : dict
            Dictionary with test labels for each level
        """
        self.models = models_dict
        self.feature_names_dict = feature_names_dict
        self.test_data = test_data
        self.test_labels = test_labels
        self.insights = {}
        
    def analyze_feature_importance(self):
        """Analyze which features each level uses most"""
        print("=" * 80)
        print("🔍 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        for level_name, model in self.models.items():
            if model is None:
                print(f"\n⚠️ {level_name.upper()} model is None, skipping...")
                continue
                
            print(f"\n📊 {level_name.upper()} - Most Important Features:")
            print("-" * 50)
            
            # Get the correct feature names for this level
            level_features = self.feature_names_dict.get(level_name, [])
            if len(level_features) == 0:
                print(f"  ⚠️ No feature names found for {level_name}")
                continue
            
            # Get feature importance from RandomForest
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            else:
                print(f"  ⚠️ Model doesn't have feature_importances_ attribute")
                continue
            
            # Ensure feature names and importance arrays match
            if len(level_features) != len(importance):
                print(f"  ⚠️ Feature name count ({len(level_features)}) doesn't match importance count ({len(importance)})")
                continue
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': level_features,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            # Store for later use
            self.insights[f'{level_name}_importance'] = importance_df
            
            # Show top features (limit to available features)
            max_features = min(10, len(importance_df))
            top_features = importance_df.head(max_features)
            
            for idx, row in top_features.iterrows():
                print(f"  {row['feature']:<30} {row['importance']:.4f}")
            
            # Calculate feature importance statistics
            total_top = top_features['importance'].sum()
            print(f"\n💡 Top {max_features} features explain {total_top:.1%} of the decision")
            
            # Find feature groups (assuming wavelength-based naming)
            wavelength_groups = self._group_features_by_wavelength(importance_df)
            if wavelength_groups:
                print(f"🌊 Most important wavelength regions:")
                for region, importance_sum in wavelength_groups.items():
                    print(f"  {region}: {importance_sum:.3f}")
    
    def analyze_decision_paths(self, sample_indices=None):
        """Show how specific samples flow through the hierarchy"""
        print("\n" + "=" * 80)
        print("🌳 DECISION PATH ANALYSIS")
        print("=" * 80)
        
        if sample_indices is None:
            # Analyze a few interesting samples
            max_samples = min(5, len(self.test_data))
            sample_indices = list(range(0, len(self.test_data), max(1, len(self.test_data)//max_samples)))[:max_samples]
        
        for idx in sample_indices:
            if idx >= len(self.test_data):
                continue
                
            print(f"\n🔬 Sample #{idx} Decision Flow:")
            print("-" * 40)
            
            # Get sample data for each level
            sample_data = {}
            for level_name in self.models.keys():
                if level_name in self.feature_names_dict:
                    level_features = self.feature_names_dict[level_name]
                    if all(feat in self.test_data.columns for feat in level_features):
                        sample_data[level_name] = self.test_data[level_features].iloc[idx:idx+1].values
                    else:
                        missing_features = [f for f in level_features if f not in self.test_data.columns]
                        print(f"  ⚠️ Missing features for {level_name}: {missing_features}")
                        continue
            
            true_labels = {k: v[idx] if hasattr(v, '__len__') and len(v) > idx else v 
                          for k, v in self.test_labels.items()}
            
            # Track predictions for hierarchy
            l1_pred = None
            
            # Level 1: Particle Detection
            if 'level1' in self.models and self.models['level1'] is not None and 'level1' in sample_data:
                l1_pred = self.models['level1'].predict(sample_data['level1'])[0]
                l1_proba = self.models['level1'].predict_proba(sample_data['level1'])[0]
                l1_conf = max(l1_proba)
                
                print(f"Level 1 (Particle Detection):")
                print(f"  True: {true_labels.get('level1', 'Unknown')}")
                print(f"  Predicted: {l1_pred} (confidence: {l1_conf:.3f})")
                
                # Show probabilities
                classes = self.models['level1'].classes_
                prob_dict = {cls: prob for cls, prob in zip(classes, l1_proba)}
                print(f"  Probabilities: {prob_dict}")
                
                # Show which features contributed most to this decision
                self._explain_single_prediction('level1', sample_data['level1'], l1_pred)
            
            # Level 2: Agglutination (if particle detected)
            if ('level2' in self.models and self.models['level2'] is not None and 
                'level2' in sample_data and l1_pred == 'particle'):
                
                l2_pred = self.models['level2'].predict(sample_data['level2'])[0]
                l2_proba = self.models['level2'].predict_proba(sample_data['level2'])[0]
                l2_conf = max(l2_proba)
                
                print(f"\nLevel 2 (Agglutination Detection):")
                print(f"  True: {true_labels.get('level2', 'Unknown')}")
                print(f"  Predicted: {l2_pred} (confidence: {l2_conf:.3f})")
                
                classes = self.models['level2'].classes_
                prob_dict = {cls: prob for cls, prob in zip(classes, l2_proba)}
                print(f"  Probabilities: {prob_dict}")
                
                self._explain_single_prediction('level2', sample_data['level2'], l2_pred)
            elif l1_pred != 'particle':
                print(f"\nLevel 2: Skipped (no particle detected)")
            
            # Level 3: Final Classification
            if 'level3' in self.models and self.models['level3'] is not None and 'level3' in sample_data:
                l3_pred = self.models['level3'].predict(sample_data['level3'])[0]
                l3_proba = self.models['level3'].predict_proba(sample_data['level3'])[0]
                l3_conf = max(l3_proba)
                
                print(f"\nLevel 3 (Final Classification):")
                print(f"  True: {true_labels.get('level3', 'Unknown')}")
                print(f"  Predicted: {l3_pred} (confidence: {l3_conf:.3f})")
                print(f"  Top 3 predictions:")
                
                # Show top 3 predictions
                classes = self.models['level3'].classes_
                top_3_idx = np.argsort(l3_proba)[-3:][::-1]
                for i, class_idx in enumerate(top_3_idx):
                    print(f"    {i+1}. {classes[class_idx]}: {l3_proba[class_idx]:.3f}")
            
            print("-" * 40)
    
    def analyze_model_confidence(self):
        """Analyze confidence patterns across levels"""
        print("\n" + "=" * 80)
        print("🎯 CONFIDENCE ANALYSIS")
        print("=" * 80)
        
        confidence_data = {}
        
        for level_name, model in self.models.items():
            if model is None:
                print(f"\n⚠️ {level_name.upper()} model is None, skipping...")
                continue
                
            # Get the correct features for this level
            level_features = self.feature_names_dict.get(level_name, [])
            if not level_features or not all(feat in self.test_data.columns for feat in level_features):
                print(f"\n⚠️ Missing features for {level_name}, skipping confidence analysis...")
                continue
            
            # Get test data for this level
            level_test_data = self.test_data[level_features].values
            
            # Get predictions and confidence scores
            probabilities = model.predict_proba(level_test_data)
            confidences = np.max(probabilities, axis=1)
            predictions = model.predict(level_test_data)
            
            confidence_data[level_name] = {
                'confidences': confidences,
                'predictions': predictions,
                'probabilities': probabilities
            }
            
            print(f"\n📊 {level_name.upper()} Confidence Statistics:")
            print(f"  Mean confidence: {np.mean(confidences):.3f}")
            print(f"  Std confidence:  {np.std(confidences):.3f}")
            print(f"  Min confidence:  {np.min(confidences):.3f}")
            print(f"  Max confidence:  {np.max(confidences):.3f}")
            
            # Low confidence samples (potential errors)
            low_conf_mask = confidences < 0.8
            if np.any(low_conf_mask):
                low_conf_count = np.sum(low_conf_mask)
                print(f"  ⚠ {low_conf_count} samples with confidence < 0.8")
                
                # Show some low confidence samples
                low_conf_indices = np.where(low_conf_mask)[0][:5]
                print(f"  Low confidence samples (showing first 5):")
                for idx in low_conf_indices:
                    print(f"    Sample {idx}: {predictions[idx]} (conf: {confidences[idx]:.3f})")
        
        self.insights['confidence_data'] = confidence_data
        return confidence_data
    
    def analyze_error_patterns(self):
        """Analyze what kinds of errors each level makes"""
        print("\n" + "=" * 80)
        print("❌ ERROR PATTERN ANALYSIS")
        print("=" * 80)
        
        for level_name, model in self.models.items():
            if model is None or level_name not in self.test_labels:
                print(f"\n⚠️ Skipping {level_name} - model is None or no test labels")
                continue
            
            # Get the correct features for this level
            level_features = self.feature_names_dict.get(level_name, [])
            if not level_features or not all(feat in self.test_data.columns for feat in level_features):
                print(f"\n⚠️ Missing features for {level_name}, skipping error analysis...")
                continue
            
            # Get test data and predictions for this level
            level_test_data = self.test_data[level_features].values
            predictions = model.predict(level_test_data)
            true_labels = self.test_labels[level_name]
            
            # Ensure arrays are same length
            min_length = min(len(predictions), len(true_labels))
            predictions = predictions[:min_length]
            true_labels = true_labels[:min_length]
            
            # Find misclassified samples
            errors = predictions != true_labels
            error_count = np.sum(errors)
            
            print(f"\n🔍 {level_name.upper()} Error Analysis:")
            print("-" * 40)
            print(f"Total errors: {error_count}/{len(predictions)} ({error_count/len(predictions):.1%})")
            
            if error_count > 0:
                error_indices = np.where(errors)[0]
                
                # Analyze error patterns
                error_df = pd.DataFrame({
                    'true_label': true_labels[error_indices],
                    'predicted_label': predictions[error_indices],
                    'sample_index': error_indices
                })
                
                # Most common error types
                error_types = error_df.groupby(['true_label', 'predicted_label']).size().sort_values(ascending=False)
                print("Most common error types:")
                for (true_lab, pred_lab), count in error_types.head(5).items():
                    print(f"  {true_lab} → {pred_lab}: {count} errors")
                
                # Analyze confidence of errors
                error_confidences = np.max(model.predict_proba(level_test_data[error_indices]), axis=1)
                print(f"Error confidence statistics:")
                print(f"  Mean confidence of errors: {np.mean(error_confidences):.3f}")
                print(f"  Errors with high confidence (>0.9): {np.sum(error_confidences > 0.9)}")
    
    def visualize_feature_importance(self):
        """Create visualizations of feature importance"""
        print("\n" + "=" * 80)
        print("📊 CREATING FEATURE IMPORTANCE VISUALIZATIONS")
        print("=" * 80)
        
        # Count valid models for plotting
        valid_models = [(name, model) for name, model in self.models.items() 
                       if model is not None and f'{name}_importance' in self.insights]
        
        if not valid_models:
            print("⚠️ No valid models with importance data for visualization")
            return
        
        fig, axes = plt.subplots(len(valid_models), 1, figsize=(12, 4*len(valid_models)))
        if len(valid_models) == 1:
            axes = [axes]
        
        for idx, (level_name, model) in enumerate(valid_models):
            importance_df = self.insights.get(f'{level_name}_importance')
            if importance_df is not None and len(importance_df) > 0:
                max_features = min(15, len(importance_df))
                top_features = importance_df.head(max_features)
                
                axes[idx].barh(range(len(top_features)), top_features['importance'])
                axes[idx].set_yticks(range(len(top_features)))
                axes[idx].set_yticklabels(top_features['feature'])
                axes[idx].set_xlabel('Feature Importance')
                axes[idx].set_title(f'{level_name.upper()} - Top {max_features} Most Important Features')
                axes[idx].grid(True, alpha=0.3)
                
                # Invert y-axis so most important is at top
                axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('hierarchical_feature_importance.png', dpi=300, bbox_inches='tight')
        print("💾 Saved feature importance plot as 'hierarchical_feature_importance.png'")
        plt.show()
    
    def generate_interpretability_report(self):
        """Generate a comprehensive interpretability report"""
        print("\n" + "=" * 80)
        print("📋 GENERATING COMPREHENSIVE INTERPRETABILITY REPORT")
        print("=" * 80)
        
        # Run all analyses
        self.analyze_feature_importance()
        confidence_data = self.analyze_model_confidence()
        self.analyze_decision_paths()
        self.analyze_error_patterns()
        
        # Create visualizations
        self.visualize_feature_importance()
        
        # Summary insights
        print("\n" + "=" * 80)
        print("🎯 KEY INSIGHTS SUMMARY")
        print("=" * 80)
        
        for level_name in self.models.keys():
            if self.models[level_name] is None:
                continue
                
            importance_df = self.insights.get(f'{level_name}_importance')
            if importance_df is not None and len(importance_df) > 0:
                top_feature = importance_df.iloc[0]
                conf_data = confidence_data.get(level_name, {})
                confidences = conf_data.get('confidences', [0])
                mean_conf = np.mean(confidences) if len(confidences) > 0 else 0
                
                print(f"\n🔍 {level_name.upper()}:")
                print(f"  Most important feature: {top_feature['feature']} ({top_feature['importance']:.3f})")
                print(f"  Average confidence: {mean_conf:.3f}")
                print(f"  Decision strategy: {'High confidence' if mean_conf > 0.9 else 'Moderate confidence' if mean_conf > 0.7 else 'Low confidence'}")
        
        return self.insights
    
    def _explain_single_prediction(self, level_name, sample, prediction):
        """Explain a single prediction using feature contributions"""
        importance_df = self.insights.get(f'{level_name}_importance')
        level_features = self.feature_names_dict.get(level_name, [])
        
        if importance_df is not None and len(level_features) > 0:
            # Get top 5 features and their values for this sample
            max_features = min(5, len(importance_df))
            top_features = importance_df.head(max_features)
            print(f"  Key contributing features:")
            
            for idx, row in top_features.iterrows():
                feature_name = row['feature']
                if feature_name in level_features:
                    feature_idx = level_features.index(feature_name)
                    if feature_idx < sample.shape[1]:
                        feature_value = sample[0, feature_idx]
                        importance = row['importance']
                        print(f"    {feature_name}: {feature_value:.3f} (importance: {importance:.3f})")
    
    def _group_features_by_wavelength(self, importance_df):
        """Group features by wavelength ranges (if applicable)"""
        wavelength_groups = {}
        
        for _, row in importance_df.iterrows():
            feature = row['feature']
            importance = row['importance']
            
            # Group based on common spectroscopic feature patterns
            feature_lower = feature.lower()
            
            if any(term in feature_lower for term in ['f1', 'f2', 'f3', 'violet', 'blue', '400', '450']):
                group = "UV-Blue (< 500nm)"
            elif any(term in feature_lower for term in ['f4', 'f5', 'green', '500', '550']):
                group = "Green (500-600nm)"
            elif any(term in feature_lower for term in ['f6', 'f7', 'f8', 'red', '600', '650', '700']):
                group = "Red (600-700nm)"
            elif any(term in feature_lower for term in ['nir', 'infrared', '750', '800']):
                group = "NIR (> 700nm)"
            elif any(term in feature_lower for term in ['clear', 'total', 'sum']):
                group = "Broadband/Total"
            elif any(term in feature_lower for term in ['ratio', 'centroid', 'width', 'skew']):
                group = "Spectral Shape"
            else:
                group = "Other"
            
            if group not in wavelength_groups:
                wavelength_groups[group] = 0
            wavelength_groups[group] += importance
        
        # Sort by importance
        return dict(sorted(wavelength_groups.items(), key=lambda x: x[1], reverse=True))


# Example usage function
def analyze_hierarchical_classifier_interpretability(models_path, test_data_path, feature_names):
    """
    Main function to run interpretability analysis
    
    Parameters:
    -----------
    models_path : str
        Path to saved models (pickle files)
    test_data_path : str
        Path to test data CSV
    feature_names : list
        List of feature column names
    """
    
    # Load models (assuming they're saved as pickle files)
    models = {}
    try:
        models['level1'] = joblib.load(f'{models_path}/level1_model.pkl')
        models['level2'] = joblib.load(f'{models_path}/level2_model.pkl') 
        models['level3'] = joblib.load(f'{models_path}/level3_model.pkl')
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return
    
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Separate features and labels
    X_test = test_df[feature_names].values
    
    # Create test labels dictionary (customize based on your data structure)
    test_labels = {
        'level1': test_df['level1_true'].values if 'level1_true' in test_df.columns else None,
        'level2': test_df['level2_true'].values if 'level2_true' in test_df.columns else None,
        'level3': test_df['level3_true'].values if 'level3_true' in test_df.columns else None,
    }
    
    # Create interpreter and run analysis
    interpreter = HierarchicalClassifierInterpreter(
        models_dict=models,
        feature_names=feature_names,
        test_data=X_test,
        test_labels=test_labels
    )
    
    # Generate comprehensive report
    insights = interpreter.generate_interpretability_report()
    
    return interpreter, insights


if __name__ == "__main__":
    # Example feature names (replace with your actual feature names)
    example_features = [
        'intensity_400nm', 'intensity_450nm', 'intensity_500nm', 'intensity_550nm',
        'intensity_600nm', 'intensity_650nm', 'intensity_700nm', 'intensity_750nm',
        'peak_width_main', 'peak_asymmetry', 'baseline_level', 'total_intensity',
        'scattering_forward', 'scattering_side', 'scattering_back', 'noise_level'
    ]
    
    print("🔍 Hierarchical Classifier Interpretability Analyzer")
    print("This tool reveals HOW your classifier makes decisions!")
    print("\nTo use:")
    print("1. Save your trained models as pickle files")
    print("2. Prepare test data with true labels") 
    print("3. Call analyze_hierarchical_classifier_interpretability()")
    print("\nExample:")
    print("interpreter, insights = analyze_hierarchical_classifier_interpretability(")
    print("    models_path='./models',")
    print("    test_data_path='test_results.csv',") 
    print("    feature_names=your_feature_list")
    print(")")