import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class HierarchicalDecisionVisualizer:
    """
    Create decision boundary visualizations for hierarchical classifier
    Shows exactly how decisions are made based on most important features
    """
    
    def __init__(self, classifier, train_df, test_df, insights=None):
        self.classifier = classifier
        self.train_df = train_df
        self.test_df = test_df
        self.insights = insights
        self.decision_plots = {}
        
    def plot_feature_decision_boundaries(self, level_name, n_features=3, save_plots=True):
        """
        Create 2D decision boundary plots using top N features
        
        Parameters:
        -----------
        level_name : str
            Which level to visualize ('level1', 'level2', 'level3')
        n_features : int
            Number of top features to use (2, 3, or 4)
        save_plots : bool
            Whether to save plots as files
        """
        
        print(f"\n🎨 Creating decision boundary plots for {level_name.upper()}")
        print("=" * 60)
        
        # Get the model and features for this level
        model = getattr(self.classifier, f'{level_name}_classifier')
        if model is None:
            print(f"❌ No model found for {level_name}")
            return
        
        # Get feature names used by this level
        if hasattr(self.classifier, f'{level_name}_features_used'):
            features_used = getattr(self.classifier, f'{level_name}_features_used')
        else:
            features_used = getattr(self.classifier, f'{level_name}_features')
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = list(zip(features_used, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in feature_importance[:n_features]]
            
            print(f"📊 Top {n_features} features for {level_name}:")
            for i, (feat, imp) in enumerate(feature_importance[:n_features]):
                print(f"  {i+1}. {feat}: {imp:.4f}")
        else:
            print(f"❌ Model doesn't have feature importance")
            return
        
        # Prepare data for this level
        if level_name == 'level1':
            # Use all training data
            plot_data = self.train_df.copy()
            y_true = self.classifier.create_level1_labels(plot_data['label'])
            
        elif level_name == 'level2':
            # Use only particle samples
            particle_samples = self.train_df[
                self.train_df['label'].isin(self.classifier.particle_labels['particle'])
            ].copy()
            if len(particle_samples) == 0:
                print(f"❌ No particle samples for {level_name}")
                return
            plot_data = particle_samples
            y_true = self.classifier.create_level2_labels(plot_data['label'])
            
        elif level_name == 'level3':
            # Use all training data
            plot_data = self.train_df.copy()
            y_true = plot_data['label'].values
        
        # Filter out unknown labels
        valid_mask = y_true != 'unknown'
        plot_data = plot_data[valid_mask]
        y_true = y_true[valid_mask]
        
        if len(plot_data) == 0:
            print(f"❌ No valid data for {level_name}")
            return
        
        print(f"📈 Using {len(plot_data)} samples for visualization")
        
        # Create all possible 2D combinations of top features
        feature_pairs = list(combinations(top_features, 2))
        
        # Calculate grid layout
        n_plots = len(feature_pairs)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create the plot
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Color mapping for different classes
        unique_classes = np.unique(y_true)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
        class_colors = dict(zip(unique_classes, colors))
        
        for idx, (feat1, feat2) in enumerate(feature_pairs):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Check if features exist in data
            if feat1 not in plot_data.columns or feat2 not in plot_data.columns:
                ax.text(0.5, 0.5, f'Features\n{feat1}\n{feat2}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Extract feature data
            X_plot = plot_data[[feat1, feat2]].values
            
            # Create decision boundary
            h = 0.02  # Step size in the mesh
            x_min, x_max = X_plot[:, 0].min() - 0.1, X_plot[:, 0].max() + 0.1
            y_min, y_max = X_plot[:, 1].min() - 0.1, X_plot[:, 1].max() + 0.1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                               np.arange(y_min, y_max, h))
            
            # For decision boundary, we need to create a full feature vector
            # We'll use median values for missing features
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            full_feature_vectors = np.zeros((len(mesh_points), len(features_used)))
            
            # Set the two features we're plotting
            feat1_idx = features_used.index(feat1)
            feat2_idx = features_used.index(feat2)
            full_feature_vectors[:, feat1_idx] = mesh_points[:, 0]
            full_feature_vectors[:, feat2_idx] = mesh_points[:, 1]
            
            # Fill other features with median values from training data
            for i, feat in enumerate(features_used):
                if i != feat1_idx and i != feat2_idx:
                    if feat in plot_data.columns:
                        full_feature_vectors[:, i] = plot_data[feat].median()
                    else:
                        full_feature_vectors[:, i] = 0  # Fallback
            
            # Predict on mesh
            try:
                Z = model.predict(full_feature_vectors)
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                ax.contourf(xx, yy, Z, alpha=0.3, levels=len(unique_classes)-1)
                
            except Exception as e:
                print(f"⚠️ Could not create decision boundary for {feat1} vs {feat2}: {e}")
            
            # Plot data points
            for class_name in unique_classes:
                mask = y_true == class_name
                if np.any(mask):
                    ax.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                             c=[class_colors[class_name]], 
                             label=class_name, 
                             alpha=0.7, 
                             s=50,
                             edgecolors='black',
                             linewidth=0.5)
            
            ax.set_xlabel(f'{feat1}')
            ax.set_ylabel(f'{feat2}')
            ax.set_title(f'{level_name.upper()}: {feat1} vs {feat2}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        for idx in range(n_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                fig.delaxes(axes[row, col])
            else:
                fig.delaxes(axes[col])
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'{level_name}_decision_boundaries.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved decision boundary plot as '{filename}'")
        
        plt.show()
        
        # Store for later use
        self.decision_plots[level_name] = {
            'top_features': top_features,
            'feature_pairs': feature_pairs,
            'class_colors': class_colors,
            'data_used': len(plot_data)
        }
        
        return fig
    
    def plot_feature_distributions(self, level_name, n_features=4, save_plots=True):
        """
        Create violin/box plots showing feature distributions by class
        """
        print(f"\n📊 Creating feature distribution plots for {level_name.upper()}")
        print("=" * 60)
        
        # Get the model and features for this level
        model = getattr(self.classifier, f'{level_name}_classifier')
        if model is None:
            print(f"❌ No model found for {level_name}")
            return
        
        # Get feature names and importance
        if hasattr(self.classifier, f'{level_name}_features_used'):
            features_used = getattr(self.classifier, f'{level_name}_features_used')
        else:
            features_used = getattr(self.classifier, f'{level_name}_features')
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance = list(zip(features_used, importance))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            top_features = [f[0] for f in feature_importance[:n_features]]
        else:
            print(f"❌ Model doesn't have feature importance")
            return
        
        # Prepare data
        if level_name == 'level1':
            plot_data = self.train_df.copy()
            y_true = self.classifier.create_level1_labels(plot_data['label'])
        elif level_name == 'level2':
            particle_samples = self.train_df[
                self.train_df['label'].isin(self.classifier.particle_labels['particle'])
            ].copy()
            plot_data = particle_samples
            y_true = self.classifier.create_level2_labels(plot_data['label'])
        elif level_name == 'level3':
            plot_data = self.train_df.copy()
            y_true = plot_data['label'].values
        
        # Filter valid data
        valid_mask = y_true != 'unknown'
        plot_data = plot_data[valid_mask]
        y_true = y_true[valid_mask]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(top_features):
            if idx >= 4:  # Limit to 4 plots
                break
                
            ax = axes[idx]
            
            if feature not in plot_data.columns:
                ax.text(0.5, 0.5, f'Feature\n{feature}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Create violin plot
            plot_df = pd.DataFrame({
                'feature_value': plot_data[feature],
                'class': y_true
            })
            
            sns.violinplot(data=plot_df, x='class', y='feature_value', ax=ax, inner='box')
            ax.set_title(f'{feature}\n(Importance: {dict(feature_importance).get(feature, 0):.3f})')
            ax.set_xlabel('Class')
            ax.set_ylabel('Feature Value')
            ax.grid(True, alpha=0.3)
            
            # Add sample counts
            class_counts = plot_df.groupby('class').size()
            for i, (class_name, count) in enumerate(class_counts.items()):
                ax.text(i, ax.get_ylim()[1], f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # Remove empty subplots
        for idx in range(len(top_features), 4):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'{level_name.upper()} - Feature Distributions by Class', fontsize=16)
        plt.tight_layout()
        
        if save_plots:
            filename = f'{level_name}_feature_distributions.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved feature distribution plot as '{filename}'")
        
        plt.show()
        return fig
    
    def create_decision_explanation_summary(self, sample_index=0):
        """
        Create a comprehensive decision explanation for a specific sample
        """
        print(f"\n🔍 DECISION EXPLANATION FOR SAMPLE #{sample_index}")
        print("=" * 60)
        
        if sample_index >= len(self.test_df):
            print(f"❌ Sample index {sample_index} out of range (max: {len(self.test_df)-1})")
            return
        
        sample_row = self.test_df.iloc[sample_index]
        true_label = sample_row['label']
        
        print(f"📋 Sample Information:")
        print(f"   True Label: {true_label}")
        print(f"   Sample Index: {sample_index}")
        
        # Create figure with subplots for each level
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        level_names = ['level1', 'level2', 'level3']
        level_titles = ['Particle Detection', 'Agglutination Detection', 'Final Classification']
        
        for idx, (level_name, title) in enumerate(zip(level_names, level_titles)):
            ax = axes[idx]
            
            model = getattr(self.classifier, f'{level_name}_classifier')
            if model is None:
                ax.text(0.5, 0.5, f'{level_name}\nNo Model', ha='center', va='center')
                continue
            
            # Get features for this level
            if hasattr(self.classifier, f'{level_name}_features_used'):
                features_used = getattr(self.classifier, f'{level_name}_features_used')
            else:
                features_used = getattr(self.classifier, f'{level_name}_features')
            
            # Get sample data for this level
            sample_features = sample_row[features_used].values.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(sample_features)[0]
            probabilities = model.predict_proba(sample_features)[0]
            classes = model.classes_
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                top_features_idx = np.argsort(importance)[-5:][::-1]  # Top 5
                
                # Plot feature importance for this sample
                top_features = [features_used[i] for i in top_features_idx]
                top_importance = importance[top_features_idx]
                top_values = [sample_features[0, i] for i in top_features_idx]
                
                bars = ax.barh(range(len(top_features)), top_importance, 
                              color='lightblue', alpha=0.7)
                
                # Add feature values as text
                for i, (feat, imp, val) in enumerate(zip(top_features, top_importance, top_values)):
                    ax.text(imp + 0.01, i, f'{feat}\nVal: {val:.3f}', 
                           va='center', fontsize=8)
                
                ax.set_yticks(range(len(top_features)))
                ax.set_yticklabels(top_features)
                ax.set_xlabel('Feature Importance')
                ax.set_title(f'{title}\nPredicted: {prediction}\nConfidence: {max(probabilities):.3f}')
                ax.grid(True, alpha=0.3)
                
                # Add probability bars at the bottom
                ax2 = ax.twinx()
                prob_bars = ax2.bar(range(len(classes)), probabilities, 
                                   alpha=0.5, color='orange', width=0.3)
                ax2.set_ylabel('Probability')
                ax2.set_ylim(0, 1)
                
                # Add class labels
                for i, (cls, prob) in enumerate(zip(classes, probabilities)):
                    ax2.text(i, prob + 0.02, f'{cls}\n{prob:.3f}', 
                            ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'sample_{sample_index}_decision_explanation.png', dpi=300, bbox_inches='tight')
        print(f"💾 Saved explanation as 'sample_{sample_index}_decision_explanation.png'")
        plt.show()
        
        return fig
    
    def generate_all_visualizations(self, n_features=3):
        """
        Generate all visualizations for all levels
        """
        print("\n🎨 GENERATING ALL DECISION VISUALIZATIONS")
        print("=" * 80)
        
        levels_to_plot = []
        if self.classifier.level1_classifier is not None:
            levels_to_plot.append('level1')
        if self.classifier.level2_classifier is not None:
            levels_to_plot.append('level2')
        if self.classifier.level3_classifier is not None:
            levels_to_plot.append('level3')
        
        for level in levels_to_plot:
            self.plot_feature_decision_boundaries(level, n_features=n_features)
            self.plot_feature_distributions(level, n_features=n_features)
        
        # Create explanation for a few interesting samples
        interesting_samples = [0, len(self.test_df)//4, len(self.test_df)//2, 3*len(self.test_df)//4]
        for sample_idx in interesting_samples:
            if sample_idx < len(self.test_df):
                self.create_decision_explanation_summary(sample_idx)
        
        print("\n✅ All visualizations completed!")
        print("📁 Files created:")
        print("   - *_decision_boundaries.png (decision boundary plots)")
        print("   - *_feature_distributions.png (feature distribution plots)")
        print("   - sample_*_decision_explanation.png (individual sample explanations)")


# Usage function to add to your main code
def create_decision_visualizations(classifier, train_df, test_df, insights=None):
    """
    Create comprehensive decision visualization plots
    
    Parameters:
    -----------
    classifier : RobustHierarchicalClassifier
        Your trained classifier
    train_df : pandas.DataFrame
        Training data
    test_df : pandas.DataFrame  
        Test data
    insights : dict
        Optional insights from interpretability analysis
    """
    
    # First, fix the agglutination labels in your classifier
    print("🔧 Updating agglutination labels...")
    classifier.agglutination_labels = {
        'no_agglutination': ['beads_in_tween_buffer', 'beads_in_running_buffer'],
        'agglutination': ['beads_in_tween_buffer_with_salt', 'beads_in_running_buffer_with_CRP']
    }
    print("✅ Updated: Salt-induced clustering now correctly classified as agglutination!")
    
    # Create visualizer
    visualizer = HierarchicalDecisionVisualizer(classifier, train_df, test_df, insights)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations(n_features=4)
    
    return visualizer


# Add this to your main code after training:
if __name__ == "__main__":
    # classifier, results, train_df, test_df, interpreter, insights = main_with_interpretability()
    
    # Then add this:
    print("\n" + "="*80)
    print("🎨 CREATING DECISION VISUALIZATIONS FOR COLLEAGUE PRESENTATION")
    print("="*80)
    
    visualizer = create_decision_visualizations(classifier, train_df, test_df, insights)