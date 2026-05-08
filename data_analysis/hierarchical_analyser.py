import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

def analyze_hierarchical_results(results_df):
    """
    Comprehensive analysis of hierarchical classification results with clean visualizations
    """
    
    plt.style.use('default')
    fig = plt.figure(figsize=(24, 16))
    
    # Create label mappings
    particle_labels = {
        'blanco': 'no_particle', 'tween_buffer': 'no_particle', 
        'tween_buffer_with_salt': 'no_particle',
        'beads_in_tween_buffer': 'particle', 'beads_in_tween_buffer_with_salt': 'particle',
        'beads_in_running_buffer': 'particle', 'beads_in_running_buffer_with_CRP': 'particle'
    }
    
    agglutination_labels = {
        'beads_in_tween_buffer': 'no_agglutination',
        'beads_in_tween_buffer_with_salt': 'no_agglutination', 
        'beads_in_running_buffer': 'no_agglutination',
        'beads_in_running_buffer_with_CRP': 'agglutination'
    }
    
    # 1. Level 1 Analysis - Particle Detection
    plt.subplot(3, 4, 1)
    true_level1 = [particle_labels.get(label, 'unknown') for label in results_df['true_label']]
    cm1 = confusion_matrix(true_level1, results_df['level1_pred'])
    
    # Clean confusion matrix plot
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['no_particle', 'particle'],
                yticklabels=['no_particle', 'particle'],
                cbar_kws={'shrink': 0.8})
    plt.title('Level 1: Particle Detection\nConfusion Matrix', fontsize=12, pad=10)
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.tick_params(axis='both', labelsize=9)
    
    # Calculate and display BALANCED accuracy
    level1_bal_acc = balanced_accuracy_score(true_level1, results_df['level1_pred'])
    plt.figtext(0.08, 0.82, f'Balanced Acc: {level1_bal_acc:.1%}', fontsize=10, weight='bold')
    
    # 2. Level 1 Confidence Distribution
    plt.subplot(3, 4, 2)
    correct_level1 = [t == p for t, p in zip(true_level1, results_df['level1_pred'])]
    correct_conf = results_df[correct_level1]['level1_conf']
    incorrect_conf = results_df[[not c for c in correct_level1]]['level1_conf']
    
    plt.hist(correct_conf, alpha=0.7, label=f'Correct ({len(correct_conf)})', bins=15, color='green', density=True)
    if len(incorrect_conf) > 0:
        plt.hist(incorrect_conf, alpha=0.7, label=f'Incorrect ({len(incorrect_conf)})', bins=15, color='red', density=True)
    plt.xlabel('Confidence Score', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.title('Level 1: Confidence Distribution', fontsize=12, pad=10)
    plt.legend(fontsize=9)
    plt.tick_params(axis='both', labelsize=9)
    
    # 3. Level 2 Analysis - Agglutination Detection (particles only)
    plt.subplot(3, 4, 5)
    particle_mask = results_df['level1_pred'] == 'particle'
    particle_results = results_df[particle_mask]
    
    if len(particle_results) > 0:
        true_level2 = [agglutination_labels.get(label, 'unknown') for label in particle_results['true_label']]
        pred_level2 = particle_results['level2_pred'].fillna('unknown')
        
        # Filter out unknown labels for confusion matrix
        valid_mask = [(t != 'unknown' and p != 'unknown') for t, p in zip(true_level2, pred_level2)]
        if sum(valid_mask) > 0:
            true_level2_valid = [t for t, v in zip(true_level2, valid_mask) if v]
            pred_level2_valid = [p for p, v in zip(pred_level2, valid_mask) if v]
            
            cm2 = confusion_matrix(true_level2_valid, pred_level2_valid, 
                                 labels=['agglutination', 'no_agglutination'])
            
            sns.heatmap(cm2, annot=True, fmt='d', cmap='Oranges',
                       xticklabels=['Agglutination', 'No Agglutination'],
                       yticklabels=['Agglutination', 'No Agglutination'],
                       cbar_kws={'shrink': 0.8})
            
            # Calculate BALANCED accuracy for Level 2 manually to avoid sklearn edge case
            tp = sum(1 for t, p in zip(true_level2_valid, pred_level2_valid) if t == 'agglutination' and p == 'agglutination')
            fn = sum(1 for t, p in zip(true_level2_valid, pred_level2_valid) if t == 'agglutination' and p == 'no_agglutination')
            tn = sum(1 for t, p in zip(true_level2_valid, pred_level2_valid) if t == 'no_agglutination' and p == 'no_agglutination')
            fp = sum(1 for t, p in zip(true_level2_valid, pred_level2_valid) if t == 'no_agglutination' and p == 'agglutination')
            
            # Manual balanced accuracy calculation
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            level2_bal_acc_manual = (sensitivity + specificity) / 2
            
            plt.figtext(0.33, 0.49, f'Balanced Acc: {level2_bal_acc_manual:.1%}', fontsize=10, weight='bold')
            
            # Add simple accuracy and breakdown for comparison
            level2_simple_acc = np.trace(cm2) / np.sum(cm2)
            plt.figtext(0.33, 0.45, f'Simple Acc: {level2_simple_acc:.1%}', fontsize=9, style='italic')
            plt.figtext(0.33, 0.41, f'Sens: {sensitivity:.1%}, Spec: {specificity:.1%}', fontsize=8, style='italic')
        else:
            plt.text(0.5, 0.5, 'No valid Level 2\npredictions', ha='center', va='center', fontsize=12)
    
    plt.title('Level 2: Agglutination Detection\nConfusion Matrix', fontsize=12, pad=10)
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.tick_params(axis='both', labelsize=9)
    plt.xticks(rotation=15)
    plt.yticks(rotation=0)
    
    # 4. Level 2 Confidence Distribution
    plt.subplot(3, 4, 6)
    if len(particle_results) > 0:
        level2_conf = particle_results['level2_conf'].dropna()
        if len(level2_conf) > 0:
            plt.hist(level2_conf, bins=15, alpha=0.7, color='orange', density=True)
            plt.xlabel('Confidence Score', fontsize=10)
            plt.ylabel('Density', fontsize=10)
            plt.title('Level 2: Confidence Distribution', fontsize=12, pad=10)
            plt.tick_params(axis='both', labelsize=9)
        else:
            plt.text(0.5, 0.5, 'No Level 2\nconfidence data', ha='center', va='center', fontsize=12)
    
    # 5. Level 3 Analysis - Multi-class Classification
    plt.subplot(3, 4, 9)
    
    # Get unique labels and create a cleaner confusion matrix
    unique_labels = sorted(results_df['true_label'].unique())
    cm3 = confusion_matrix(results_df['true_label'], results_df['level3_pred'], labels=unique_labels)
    
    # Create abbreviated labels for readability
    # short_labels = []
    # for label in unique_labels:
    #     if 'running_buffer_with_CRP' in label:
    #         short_labels.append('Run+CRP')
    #     elif 'beads_in_running_buffer' in label:	
    #         short_labels.append('Run-CRP')
    #     elif 'running_buffer' in label:
    #         short_labels.append('Run buf')
    #     elif 'tween_buffer_with_salt' in label:
    #         short_labels.append('Tw+Salt')
    #     elif 'beads_in_tween' in label:
    #         short_labels.append('Tw-Salt')
    #     elif 'tween_buffer' in label and 'salt' not in label:
    #         short_labels.append('Tween')
    #     elif 'blanco' in label:
    #         short_labels.append('Blank')
    #     else:
    #         short_labels.append(label[:8])  # Truncate long labels
    
    sns.heatmap(cm3, annot=True, fmt='d', cmap='Purples',
                xticklabels=unique_labels, yticklabels=unique_labels,
                cbar_kws={'shrink': 0.8})
    plt.title('Level 3: Multi-class Classification\nConfusion Matrix', fontsize=12, pad=10)
    plt.ylabel('True Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    # Calculate BALANCED accuracy for Level 3
    level3_bal_acc = balanced_accuracy_score(results_df['true_label'], results_df['level3_pred'])
    plt.figtext(0.58, 0.16, f'Balanced Acc: {level3_bal_acc:.1%}', fontsize=10, weight='bold')
    
    # Add simple accuracy for comparison
    level3_simple_acc = np.trace(cm3) / np.sum(cm3)
    plt.figtext(0.58, 0.12, f'Simple Acc: {level3_simple_acc:.1%}', fontsize=9, style='italic')
    
    # 6. Level 3 Confidence Distribution
    plt.subplot(3, 4, 10)
    correct_level3 = results_df['true_label'] == results_df['level3_pred']
    correct_conf3 = results_df[correct_level3]['level3_conf']
    incorrect_conf3 = results_df[~correct_level3]['level3_conf']
    
    plt.hist(correct_conf3, alpha=0.7, label=f'Correct ({len(correct_conf3)})', bins=15, color='green', density=True)
    if len(incorrect_conf3) > 0:
        plt.hist(incorrect_conf3, alpha=0.7, label=f'Incorrect ({len(incorrect_conf3)})', bins=15, color='red', density=True)
    plt.xlabel('Confidence Score', fontsize=10)
    plt.ylabel('Density', fontsize=10)
    plt.title('Level 3: Confidence Distribution', fontsize=12, pad=10)
    plt.legend(fontsize=9)
    plt.tick_params(axis='both', labelsize=9)
    
    # 7. Decision Flow Pie Chart
    plt.subplot(3, 4, 3)
    level1_counts = results_df['level1_pred'].value_counts()
    colors = ['lightblue', 'lightcoral']
    plt.pie(level1_counts.values, labels=[f'{l}\n({v})' for l, v in zip(level1_counts.index, level1_counts.values)], 
            autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Level 1: Decision Distribution', fontsize=12, pad=10)
    
    # 8. Error Analysis Bar Chart
    plt.subplot(3, 4, 7)
    error_analysis = []
    
    # Level 1 errors
    level1_errors = sum(t != p for t, p in zip(true_level1, results_df['level1_pred']))
    error_analysis.append(('Level 1', level1_errors, len(results_df)))
    
    # Level 2 errors (particles only)
    if len(particle_results) > 0:
        true_level2 = [agglutination_labels.get(label, 'unknown') for label in particle_results['true_label']]
        pred_level2 = particle_results['level2_pred'].fillna('unknown')
        valid_mask = [(t != 'unknown' and p != 'unknown') for t, p in zip(true_level2, pred_level2)]
        if sum(valid_mask) > 0:
            true_level2_valid = [t for t, v in zip(true_level2, valid_mask) if v]
            pred_level2_valid = [p for p, v in zip(pred_level2, valid_mask) if v]
            level2_errors = sum(t != p for t, p in zip(true_level2_valid, pred_level2_valid))
            error_analysis.append(('Level 2', level2_errors, len(true_level2_valid)))
    
    # Level 3 errors
    level3_errors = sum(results_df['true_label'] != results_df['level3_pred'])
    error_analysis.append(('Level 3', level3_errors, len(results_df)))
    
    # Plot error rates
    levels = [ea[0] for ea in error_analysis]
    error_rates = [ea[1]/ea[2] for ea in error_analysis]
    colors = ['lightblue', 'orange', 'plum']
    
    bars = plt.bar(levels, error_rates, color=colors[:len(levels)])
    plt.ylabel('Error Rate (Balanced)', fontsize=10)
    plt.title('Error Rate by Classification Level\n(Balanced Accuracy)', fontsize=12, pad=10)
    plt.ylim(0, max(error_rates) * 1.2)
    plt.tick_params(axis='both', labelsize=9)
    
    # Add error rate labels on bars
    for bar, rate in zip(bars, error_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(error_rates)*0.02,
                f'{rate:.1%}', ha='center', va='bottom', fontsize=9)
    
    # 9. Sample Distribution
    plt.subplot(3, 4, 11)
    true_label_counts = results_df['true_label'].value_counts()
    # short_true_labels = [short_labels[unique_labels.index(label)] for label in true_label_counts.index]
    true_labels = true_label_counts.index.tolist()
    
    plt.bar(range(len(true_label_counts)), true_label_counts.values, color='lightgreen')
    plt.xlabel('Sample Type', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.title('Test Set Sample Distribution', fontsize=12, pad=10)
    plt.xticks(range(len(true_label_counts)), true_labels, rotation=45, ha='right', fontsize=9)
    plt.tick_params(axis='both', labelsize=9)
    
    # Add count labels on bars
    for i, count in enumerate(true_label_counts.values):
        plt.text(i, count + max(true_label_counts.values)*0.01, str(count), 
                ha='center', va='bottom', fontsize=9)
    
    # 10. Level 2 Detailed Analysis (for particles only)
    plt.subplot(3, 4, 4)
    if len(particle_results) > 0 and 'level2_pred' in particle_results.columns:
        # Show Level 2 prediction distribution for particle samples
        level2_pred_counts = particle_results['level2_pred'].value_counts()
        plt.pie(level2_pred_counts.values, 
               labels=[f'{l}\n({v})' for l, v in zip(level2_pred_counts.index, level2_pred_counts.values)],
               autopct='%1.1f%%', startangle=90, colors=['lightyellow', 'lightcyan'])
        plt.title('Level 2: Agglutination Predictions\n(Particle Samples Only)', fontsize=12, pad=10)
    else:
        plt.text(0.5, 0.5, 'No Level 2\nData Available', ha='center', va='center', fontsize=14)
        plt.title('Level 2: Agglutination Analysis', fontsize=12, pad=10)
    plt.axis('off')
    
    # 11. Confidence Score Comparison
    plt.subplot(3, 4, 8)
    conf_data = []
    conf_labels = []
    
    # Level 1 confidence
    conf_data.append(results_df['level1_conf'].values)
    conf_labels.append('Level 1')
    
    # Level 2 confidence (particles only)
    if len(particle_results) > 0:
        level2_conf_clean = particle_results['level2_conf'].dropna()
        if len(level2_conf_clean) > 0:
            conf_data.append(level2_conf_clean.values)
            conf_labels.append('Level 2')
    
    # Level 3 confidence
    conf_data.append(results_df['level3_conf'].values)
    conf_labels.append('Level 3')
    
    plt.boxplot(conf_data, labels=conf_labels)
    plt.ylabel('Confidence Score', fontsize=10)
    plt.title('Confidence Score Comparison', fontsize=12, pad=10)
    plt.tick_params(axis='both', labelsize=9)
    plt.grid(True, alpha=0.3)
    
    # 12. Performance Summary Table
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    # Create performance summary with BALANCED accuracy
    perf_text = "PERFORMANCE SUMMARY\n" + "="*25 + "\n\n"
    
    # Level 1
    level1_bal_acc = balanced_accuracy_score(true_level1, results_df['level1_pred'])
    perf_text += f"Level 1 (Particle Detection):\n"
    perf_text += f"  Balanced Accuracy: {level1_bal_acc:.1%}\n"
    perf_text += f"  Samples: {len(results_df)}\n\n"
    
    # Level 2
    if len(particle_results) > 0:
        true_level2 = [agglutination_labels.get(label, 'unknown') for label in particle_results['true_label']]
        pred_level2 = particle_results['level2_pred'].fillna('unknown')
        valid_mask = [(t != 'unknown' and p != 'unknown') for t, p in zip(true_level2, pred_level2)]
        if sum(valid_mask) > 0:
            true_level2_valid = [t for t, v in zip(true_level2, valid_mask) if v]
            pred_level2_valid = [p for p, v in zip(pred_level2, valid_mask) if v]
            
            # Manual balanced accuracy calculation to fix the 100% issue
            tp = sum(1 for t, p in zip(true_level2_valid, pred_level2_valid) if t == 'agglutination' and p == 'agglutination')
            fn = sum(1 for t, p in zip(true_level2_valid, pred_level2_valid) if t == 'agglutination' and p == 'no_agglutination')
            tn = sum(1 for t, p in zip(true_level2_valid, pred_level2_valid) if t == 'no_agglutination' and p == 'no_agglutination')
            fp = sum(1 for t, p in zip(true_level2_valid, pred_level2_valid) if t == 'no_agglutination' and p == 'agglutination')
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            level2_bal_acc = (sensitivity + specificity) / 2
            
            perf_text += f"Level 2 (Agglutination Detection):\n"
            perf_text += f"  Balanced Accuracy: {level2_bal_acc:.1%}\n"
            perf_text += f"  Sensitivity: {sensitivity:.1%}\n"
            perf_text += f"  Specificity: {specificity:.1%}\n"
            perf_text += f"  Samples: {len(true_level2_valid)}\n"
            
            # Add bias warning for Level 2
            if level2_bal_acc < 0.75:
                perf_text += f"  ⚠ BIAS: Overpredicts agglutination\n"
            if specificity < 0.3:
                perf_text += f"  ⚠ LOW SPECIFICITY: Too many false positives\n"
            perf_text += "\n"
    
    # Level 3
    level3_bal_acc = balanced_accuracy_score(results_df['true_label'], results_df['level3_pred'])
    perf_text += f"Level 3 (Multi-class):\n"
    perf_text += f"  Balanced Accuracy: {level3_bal_acc:.1%}\n"
    perf_text += f"  Samples: {len(results_df)}\n\n"
    
    # Overall assessment
    if level1_bal_acc > 0.95:
        perf_text += "✓ Excellent particle detection\n"
    if 'level2_bal_acc' in locals():
        if level2_bal_acc > 0.7:
            perf_text += "✓ Good agglutination detection\n"
        else:
            perf_text += "⚠ Agglutination needs improvement\n"
    if level3_bal_acc > 0.9:
        perf_text += "✓ Excellent multi-class performance\n"
    
    plt.text(0.05, 0.95, perf_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout(pad=2.0)
    plt.savefig('hierarchical_analysis_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return error_analysis

def generate_classification_report(results_df):
    """Generate detailed classification reports for each level"""
    
    # Create label mappings
    particle_labels = {
        'blanco': 'no_particle', 'tween_buffer': 'no_particle', 
        'tween_buffer_with_salt': 'no_particle',
        'beads_in_tween_buffer': 'particle', 'beads_in_tween_buffer_with_salt': 'particle',
        'beads_in_running_buffer': 'particle', 'beads_in_running_buffer_with_CRP': 'particle'
    }
    
    agglutination_labels = {
        'beads_in_tween_buffer': 'no_agglutination',
        'beads_in_tween_buffer_with_salt': 'no_agglutination', 
        'beads_in_running_buffer': 'no_agglutination',
        'beads_in_running_buffer_with_CRP': 'agglutination'
    }
    
    print("="*80)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*80)
    
    # Level 1 Report
    print("\nLEVEL 1: PARTICLE DETECTION")
    print("-" * 40)
    true_level1 = [particle_labels.get(label, 'unknown') for label in results_df['true_label']]
    print(classification_report(true_level1, results_df['level1_pred']))
    
    # Level 2 Report
    print("\nLEVEL 2: AGGLUTINATION DETECTION (Particles Only)")
    print("-" * 50)
    particle_mask = results_df['level1_pred'] == 'particle'
    particle_results = results_df[particle_mask]
    
    if len(particle_results) > 0:
        true_level2 = [agglutination_labels.get(label, 'unknown') for label in particle_results['true_label']]
        pred_level2 = particle_results['level2_pred'].fillna('unknown')
        
        # Filter out unknown labels
        valid_mask = [(t != 'unknown' and p != 'unknown') for t, p in zip(true_level2, pred_level2)]
        if sum(valid_mask) > 0:
            true_level2_valid = [t for t, v in zip(true_level2, valid_mask) if v]
            pred_level2_valid = [p for p, v in zip(pred_level2, valid_mask) if v]
            print(classification_report(true_level2_valid, pred_level2_valid))
            
            # Additional Level 2 analysis
            from collections import Counter
            print("\nLevel 2 Prediction Analysis:")
            print(f"True labels distribution: {Counter(true_level2_valid)}")
            print(f"Predicted labels distribution: {Counter(pred_level2_valid)}")
        else:
            print("No valid predictions for Level 2 analysis")
    else:
        print("No particle samples detected for Level 2 analysis")
    
    # Level 3 Report
    print("\nLEVEL 3: MULTI-CLASS CLASSIFICATION")
    print("-" * 40)
    print(classification_report(results_df['true_label'], results_df['level3_pred']))

def analyze_confidence_thresholds(results_df):
    """Analyze how different confidence thresholds affect accuracy"""
    
    # Create label mappings
    particle_labels = {
        'blanco': 'no_particle', 'tween_buffer': 'no_particle', 
        'tween_buffer_with_salt': 'no_particle',
        'beads_in_tween_buffer': 'particle', 'beads_in_tween_buffer_with_salt': 'particle',
        'beads_in_running_buffer': 'particle', 'beads_in_running_buffer_with_CRP': 'particle'
    }
    
    true_level1 = [particle_labels.get(label, 'unknown') for label in results_df['true_label']]
    
    print("\n" + "="*60)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("="*60)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    
    print("\nLevel 1 (Particle Detection):")
    print("Threshold | Accuracy | Coverage | High-Conf Samples")
    print("-" * 50)
    
    for threshold in thresholds:
        high_conf_mask = results_df['level1_conf'] >= threshold
        if high_conf_mask.sum() > 0:
            high_conf_results = results_df[high_conf_mask]
            true_high_conf = [true_level1[i] for i, mask in enumerate(high_conf_mask) if mask]
            accuracy = sum(t == p for t, p in zip(true_high_conf, high_conf_results['level1_pred'])) / len(true_high_conf)
            coverage = high_conf_mask.sum() / len(results_df)
            print(f"   {threshold:.2f}   |   {accuracy:.3f}  |   {coverage:.3f}  |      {high_conf_mask.sum()}")
        else:
            print(f"   {threshold:.2f}   |    N/A   |   0.000  |         0")

def main():
    """Main analysis pipeline"""
    try:
        # Load results 
        results_file = 'side_scatter_analysis_features.csv'
        results_df = pd.read_csv(results_file)
        print(f"Loaded {len(results_df)} test samples from {results_file}")
        
        # Quick sanity check - print first few rows
        print(f"\nFirst 3 rows of data:")
        print(results_df[['true_label', 'level2_pred', 'level2_conf']].head(3).to_string())
        
        # Check Level 2 predictions distribution
        particle_mask = results_df['level1_pred'] == 'particle'
        particle_results = results_df[particle_mask]
        print(f"\nLevel 2 predictions in CSV file:")
        print(particle_results['level2_pred'].value_counts().to_string())
        
        # Generate comprehensive analysis
        print("\nGenerating detailed visualization...")
        error_analysis = analyze_hierarchical_results(results_df)
        
        # Generate detailed reports
        generate_classification_report(results_df)
        
        # Confidence threshold analysis
        analyze_confidence_thresholds(results_df)
        
        # Summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        # Overall accuracies with BALANCED accuracy
        particle_labels = {
            'blanco': 'no_particle', 'tween_buffer': 'no_particle', 
            'tween_buffer_with_salt': 'no_particle',
            'beads_in_tween_buffer': 'particle', 'beads_in_tween_buffer_with_salt': 'particle',
            'beads_in_running_buffer': 'particle', 'beads_in_running_buffer_with_CRP': 'particle'
        }
        
        true_level1 = [particle_labels.get(label, 'unknown') for label in results_df['true_label']]
        level1_bal_acc = balanced_accuracy_score(true_level1, results_df['level1_pred'])
        level3_bal_acc = balanced_accuracy_score(results_df['true_label'], results_df['level3_pred'])
        
        print(f"Level 1 (Particle Detection) Balanced Accuracy: {level1_bal_acc:.3f}")
        print(f"Level 3 (Multi-class) Balanced Accuracy: {level3_bal_acc:.3f}")
        
        # Level 2 balanced accuracy - fix the counting bug with detailed debugging
        particle_mask = results_df['level1_pred'] == 'particle'
        particle_results = results_df[particle_mask]
        if len(particle_results) > 0:
            print(f"\nDEBUG: Total particle samples: {len(particle_results)}")
            
            # Show all Level 2 predictions from CSV
            level2_pred_counts = particle_results['level2_pred'].value_counts()
            print(f"DEBUG: All Level 2 predictions from CSV: {level2_pred_counts.to_dict()}")
            
            # Direct calculation from CSV to avoid filtering bugs
            level2_true_labels = []
            level2_pred_labels = []
            skipped_samples = []
            
            for idx, row in particle_results.iterrows():
                true_label = row['true_label']
                pred_label = row['level2_pred']
                
                if pd.notna(pred_label) and pred_label != 'not_applicable':
                    # Map true labels to agglutination categories
                    if true_label == 'beads_in_running_buffer_with_CRP':
                        true_agg = 'agglutination'
                    elif true_label in ['beads_in_running_buffer', 'beads_in_tween_buffer', 'beads_in_tween_buffer_with_salt', 'blanco']:
                        true_agg = 'no_agglutination'
                    else:
                        skipped_samples.append((true_label, pred_label))
                        continue  # Skip unknown labels
                    
                    level2_true_labels.append(true_agg)
                    level2_pred_labels.append(pred_label)
                else:
                    skipped_samples.append((true_label, pred_label))
            
            print(f"DEBUG: Processed {len(level2_true_labels)} valid Level 2 samples")
            print(f"DEBUG: Skipped {len(skipped_samples)} samples: {skipped_samples}")
            print(f"DEBUG: True distribution: {dict(pd.Series(level2_true_labels).value_counts())}")
            print(f"DEBUG: Pred distribution: {dict(pd.Series(level2_pred_labels).value_counts())}")
            
            # Check for the discrepancy
            csv_no_agg = level2_pred_counts.get('no_agglutination', 0)
            processed_no_agg = pd.Series(level2_pred_labels).value_counts().get('no_agglutination', 0)
            
            if csv_no_agg != processed_no_agg:
                print(f"🚨 DISCREPANCY FOUND!")
                print(f"   CSV shows {csv_no_agg} no_agglutination predictions")
                print(f"   But only {processed_no_agg} were processed")
                print(f"   Missing: {csv_no_agg - processed_no_agg} samples")
                
                # Find the missing sample(s)
                all_no_agg = particle_results[particle_results['level2_pred'] == 'no_agglutination']
                print(f"\nAll no_agglutination samples in CSV:")
                for idx, row in all_no_agg.iterrows():
                    print(f"   {row['true_label']} -> {row['level2_pred']}")
            
            if len(level2_true_labels) > 0:
                tp = sum(1 for t, p in zip(level2_true_labels, level2_pred_labels) if t == 'agglutination' and p == 'agglutination')
                fn = sum(1 for t, p in zip(level2_true_labels, level2_pred_labels) if t == 'agglutination' and p == 'no_agglutination')
                tn = sum(1 for t, p in zip(level2_true_labels, level2_pred_labels) if t == 'no_agglutination' and p == 'no_agglutination')
                fp = sum(1 for t, p in zip(level2_true_labels, level2_pred_labels) if t == 'no_agglutination' and p == 'agglutination')
                
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                level2_bal_acc = (sensitivity + specificity) / 2
                
                print(f"Level 2 (Agglutination Detection) CORRECTED Balanced Accuracy: {level2_bal_acc:.3f}")
                print(f"CORRECTED Level 2 Analysis:")
                print(f"  TP (agglutination correct): {tp}")
                print(f"  TN (no_agglutination correct): {tn}")
                print(f"  FP (false agglutination): {fp}")
                print(f"  FN (missed agglutination): {fn}")
                print(f"  Sensitivity: {sensitivity:.3f}")
                print(f"  Specificity: {specificity:.3f}")
                
                if specificity < 1.0:
                    print(f"✓ REALISTIC: Model has {fp} false positive(s) - this is normal!")
                elif fp == 0 and csv_no_agg > processed_no_agg:
                    print(f"🤔 SUSPICIOUS: Perfect specificity but missing {csv_no_agg - processed_no_agg} samples")
                else:
                    print(f"⚠ Perfect specificity - check for issues")
        
        print(f"\nNote: The correct Level 2 performance from direct CSV analysis.")
        
        # Confidence statistics
        print(f"\nAverage Confidence Scores:")
        print(f"Level 1: {results_df['level1_conf'].mean():.3f}")
        level2_conf_mean = results_df['level2_conf'].dropna().mean()
        if not pd.isna(level2_conf_mean):
            print(f"Level 2: {level2_conf_mean:.3f}")
        print(f"Level 3: {results_df['level3_conf'].mean():.3f}")
        
        # Sample distribution
        print(f"\nSample Distribution:")
        print(results_df['true_label'].value_counts().to_string())
        
        print(f"\nLevel 1 Predictions:")
        print(results_df['level1_pred'].value_counts().to_string())
        
        print(f"\nAnalysis complete! Check 'hierarchical_analysis_detailed.png' for visualizations.")
        
    except FileNotFoundError:
        print("Error: 'robust_hierarchical_results.csv' not found!")
        print("Please run the hierarchical classifier training script first.")
        return None
    
    return results_df

if __name__ == "__main__":
    results = main()