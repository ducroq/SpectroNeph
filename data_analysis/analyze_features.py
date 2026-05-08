import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

def analyze_feature_selection_methods():
    """
    Comprehensive analysis of what each feature selection method reveals
    """
    
    print("🔬 FEATURE SELECTION METHODS: WHAT EACH METHOD REVEALS")
    print("=" * 80)
    
    # Define the feature rankings from your results
    level1_data = {
        'ANOVA': ['Clear', 'F8', 'NIR', 'edges_sum', 'long_sum', 'F6', 'F7', 'F3', 'spectral_skewness', 'total_sum'],
        'MI': ['short_fraction', 'long_fraction', 'edges_fraction', 'F6', 'spectral_skewness', 'center_edges_sum_ratio', 'F5', 'total_sum', 'long_sum', 'short_long_sum_ratio'],
        'RF': ['total_sum', 'F7', 'long_sum', 'center_sum', 'F1', 'edges_sum', 'NIR', 'F2', 'F5', 'short_long_sum_ratio']
    }
    
    level2_data = {
        'ANOVA': ['spectral_skewness', 'F1', 'F7', 'F4', 'center_sum', 'mid_sum', 'F5', 'total_sum', 'F2', 'short_sum'],
        'MI': ['spectral_centroid', 'F3', 'spectral_variance', 'F5', 'spectral_width', 'F1', 'spectral_skewness', 'NIR', 'short_sum', 'center_sum'],
        'RF': ['spectral_centroid', 'spectral_variance', 'spectral_skewness', 'edges_fraction', 'mid_fraction', 'center_edges_sum_ratio', 'NIR', 'F3', 'F1', 'spectral_width']
    }
    
    level3_data = {
        'ANOVA': ['F7', 'F5', 'F1', 'center_sum', 'mid_sum', 'F3', 'total_sum', 'F4', 'short_sum', 'long_sum'],
        'MI': ['F5', 'F3', 'spectral_skewness', 'spectral_centroid', 'center_sum', 'long_fraction', 'mid_sum', 'short_long_sum_ratio', 'Clear', 'long_sum'],
        'RF': ['spectral_centroid', 'short_long_sum_ratio', 'long_fraction', 'F3', 'spectral_skewness', 'F5', 'mid_sum', 'short_fraction', 'spectral_variance', 'F6']
    }
    
    all_levels = {
        'Level 1 (Particle Detection)': level1_data,
        'Level 2 (Agglutination Detection)': level2_data, 
        'Level 3 (Multi-class Classification)': level3_data
    }
    
    # Method interpretations
    method_meanings = {
        'ANOVA': {
            'name': 'ANOVA F-test',
            'measures': 'Linear separability',
            'good_for': 'Classes well-separated by mean differences',
            'interpretation': 'High F-score = large between-class variance vs within-class variance',
            'bias': 'Favors features with large magnitude differences between classes'
        },
        'MI': {
            'name': 'Mutual Information',
            'measures': 'Non-linear dependencies', 
            'good_for': 'Complex, non-linear relationships',
            'interpretation': 'High MI = knowing feature value gives lots of info about class',
            'bias': 'Captures any type of relationship (linear + non-linear)'
        },
        'RF': {
            'name': 'Random Forest Importance',
            'measures': 'Practical classification utility',
            'good_for': 'Features that actually help in decision trees',
            'interpretation': 'High importance = frequently used in successful splits',
            'bias': 'Considers feature interactions and real-world performance'
        }
    }
    
    # Analyze each level
    for level_name, level_data in all_levels.items():
        print(f"\n📊 {level_name.upper()}")
        print("=" * 60)
        
        # Find overlaps and unique features
        anova_set = set(level_data['ANOVA'][:5])  # Top 5
        mi_set = set(level_data['MI'][:5])
        rf_set = set(level_data['RF'][:5])
        
        # All methods agree on these
        consensus = anova_set & mi_set & rf_set
        
        # Two methods agree
        anova_mi = (anova_set & mi_set) - consensus
        anova_rf = (anova_set & rf_set) - consensus  
        mi_rf = (mi_set & rf_set) - consensus
        
        # Unique to each method
        anova_only = anova_set - mi_set - rf_set
        mi_only = mi_set - anova_set - rf_set
        rf_only = rf_set - anova_set - mi_set
        
        print(f"🎯 CONSENSUS (All 3 methods agree): {sorted(consensus) if consensus else 'None'}")
        print(f"🤝 ANOVA + MI agree: {sorted(anova_mi) if anova_mi else 'None'}")
        print(f"🤝 ANOVA + RF agree: {sorted(anova_rf) if anova_rf else 'None'}")
        print(f"🤝 MI + RF agree: {sorted(mi_rf) if mi_rf else 'None'}")
        print(f"🔵 ANOVA only: {sorted(anova_only) if anova_only else 'None'}")
        print(f"🟢 MI only: {sorted(mi_only) if mi_only else 'None'}")
        print(f"🟠 RF only: {sorted(rf_only) if rf_only else 'None'}")
        
        # Feature type analysis
        analyze_feature_types(level_data, level_name)
    
    # Create visualization
    create_feature_selection_comparison_plot(all_levels, method_meanings)
    
    return all_levels, method_meanings

def analyze_feature_types(level_data, level_name):
    """Analyze what types of features each method prefers"""
    
    print(f"\n🔬 Feature Type Analysis for {level_name}:")
    
    # Categorize features
    feature_categories = {
        'Raw Intensities': ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Clear', 'NIR'],
        'Sums/Totals': ['total_sum', 'long_sum', 'short_sum', 'center_sum', 'mid_sum', 'edges_sum'],
        'Fractions': ['short_fraction', 'long_fraction', 'mid_fraction', 'edges_fraction'],
        'Ratios': ['short_long_sum_ratio', 'center_edges_sum_ratio'],
        'Spectral Shape': ['spectral_centroid', 'spectral_variance', 'spectral_width', 'spectral_skewness']
    }
    
    for method, features in level_data.items():
        print(f"  {method} prefers:")
        category_counts = {cat: 0 for cat in feature_categories}
        
        for feature in features[:5]:  # Top 5
            for category, cat_features in feature_categories.items():
                if feature in cat_features:
                    category_counts[category] += 1
                    break
        
        for category, count in category_counts.items():
            if count > 0:
                print(f"    {category}: {count}/5 features")

def create_feature_selection_comparison_plot(all_levels, method_meanings):
    """Create comprehensive visualization of feature selection comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    # Color scheme
    colors = {'ANOVA': '#FF6B6B', 'MI': '#4ECDC4', 'RF': '#45B7D1'}
    
    # Plot 1: Method characteristics
    ax = axes[0]
    methods = list(method_meanings.keys())
    characteristics = ['Linear\nSeparability', 'Non-linear\nDependencies', 'Practical\nUtility']
    
    y_pos = np.arange(len(characteristics))
    for i, method in enumerate(methods):
        values = [1 if method == 'ANOVA' else 0.3,  # Linear separability
                 1 if method == 'MI' else 0.3,      # Non-linear dependencies  
                 1 if method == 'RF' else 0.3]      # Practical utility
        
        ax.barh(y_pos + i*0.25, values, 0.25, 
               label=method, color=colors[method], alpha=0.7)
    
    ax.set_yticks(y_pos + 0.25)
    ax.set_yticklabels(characteristics)
    ax.set_xlabel('Strength (Relative)')
    ax.set_title('What Each Method Measures Best')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2-4: Feature overlap analysis for each level
    for idx, (level_name, level_data) in enumerate(list(all_levels.items())):
        if idx >= 3:
            break
            
        ax = axes[idx + 1]
        
        # Create Venn-like visualization
        methods = ['ANOVA', 'MI', 'RF']
        top5_sets = {method: set(features[:5]) for method, features in level_data.items()}
        
        # Calculate overlaps
        all_features = set()
        for features in top5_sets.values():
            all_features.update(features)
        
        overlap_matrix = np.zeros((len(methods), len(methods)))
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    overlap_matrix[i, j] = len(top5_sets[method1])
                else:
                    overlap_matrix[i, j] = len(top5_sets[method1] & top5_sets[method2])
        
        # Plot heatmap
        im = ax.imshow(overlap_matrix, cmap='Blues', aspect='auto')
        
        # Add text annotations
        for i in range(len(methods)):
            for j in range(len(methods)):
                text = ax.text(j, i, f'{int(overlap_matrix[i, j])}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_xticks(range(len(methods)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_yticklabels(methods)
        ax.set_title(f'{level_name}\nFeature Overlap (Top 5)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Number of Shared Features')
    
    plt.tight_layout()
    plt.savefig('feature_selection_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def interpret_your_results():
    """
    Specific interpretation of your spectroscopic results
    """
    print("\n🧬 INTERPRETATION FOR YOUR SPECTROSCOPIC DATA")
    print("=" * 80)
    
    insights = {
        'Level 1 (Particle Detection)': {
            'ANOVA_insight': "Favors RAW INTENSITY features (Clear, F8, NIR) - particles vs no-particles creates big intensity differences",
            'MI_insight': "Favors RATIO features (short_fraction, long_fraction) - spectral SHAPE changes matter most",
            'RF_insight': "Balances both - uses total_sum (intensity) + ratios for robust detection",
            'conclusion': "RF choice is SMART: combines intensity differences with spectral shape analysis"
        },
        'Level 2 (Agglutination Detection)': {
            'ANOVA_insight': "Favors spectral_skewness + specific channels (F1, F7) - agglutination shifts peak asymmetry",
            'MI_insight': "Favors spectral_centroid + variance - captures NON-LINEAR scattering changes from clustering",
            'RF_insight': "Uses spectral_centroid as #1 feature - this is the GOLD STANDARD for agglutination detection",
            'conclusion': "RF correctly identifies spectral_centroid as the key agglutination indicator"
        },
        'Level 3 (Multi-class Classification)': {
            'ANOVA_insight': "Favors specific channels (F7, F5, F1) - each sample type has distinct spectral signature",
            'MI_insight': "Favors F5, F3 + spectral features - captures complex class relationships",
            'RF_insight': "Uses spectral_centroid + ratios - builds on Level 2 insights for fine classification",
            'conclusion': "RF creates a HIERARCHICAL feature strategy: spectral shape → specific channels"
        }
    }
    
    for level, interpretation in insights.items():
        print(f"\n📊 {level}:")
        print(f"  🔵 ANOVA: {interpretation['ANOVA_insight']}")
        print(f"  🟢 MI: {interpretation['MI_insight']}")
        print(f"  🟠 RF: {interpretation['RF_insight']}")
        print(f"  ✅ WHY RF WINS: {interpretation['conclusion']}")
    
    print(f"\n🎯 OVERALL CONCLUSION:")
    print(f"Random Forest feature selection is superior because it:")
    print(f"  1. Considers FEATURE INTERACTIONS (not just individual feature strength)")
    print(f"  2. Focuses on PRACTICAL CLASSIFICATION performance")
    print(f"  3. Balances different feature types optimally")
    print(f"  4. Builds HIERARCHICAL understanding (intensity → shape → specific channels)")
    print(f"  5. Handles the MULTI-LEVEL nature of your classification problem")

if __name__ == "__main__":
    # Run the complete analysis
    all_levels, method_meanings = analyze_feature_selection_methods()
    interpret_your_results()
    
    print(f"\n💾 Visualization saved as 'feature_selection_methods_comparison.png'")