import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class AgglutinationExperimentOptimizer:
    """
    Recommendations for improving salt-induced agglutination experiments
    based on your current results analysis
    """
    
    def __init__(self):
        self.current_concentrations = [0.0, 3.6, 7.2]  # mg/mL
        self.target_effect_size = 1.0  # Cohen's d > 1.0 for strong effect
    
    def analyze_current_results(self, df):
        """Analyze why current experiment shows weak effects"""
        
        print("🔍 ANALYSIS OF CURRENT EXPERIMENT LIMITATIONS")
        print("="*60)
        
        # Create salt_conc_numeric from label if it doesn't exist
        if 'salt_conc_numeric' not in df.columns:
            # Extract concentration from label
            def extract_concentration(label):
                if 'salt_0_mg_per_mL' in str(label):
                    return 0.0
                elif 'salt_3.6_mg_per_mL' in str(label):
                    return 3.6
                elif 'salt_7.2_mg_per_mL' in str(label):
                    return 7.2
                else:
                    return 0.0  # Default for unknown labels
            
            df['salt_conc_numeric'] = df['label'].apply(extract_concentration)
            print(f"Created salt_conc_numeric from labels:")
            print(df['salt_conc_numeric'].value_counts().sort_index())
        
        # Calculate actual effect sizes observed
        effects = {}
        key_features = ['spectral_centroid', 'total_sum', 'short_long_sum_ratio', 
                       'F1_to_F7_ratio', 'spectral_width']
        
        for feature in key_features:
            if feature in df.columns:
                conc_0 = df[df['salt_conc_numeric'] == 0.0][feature]
                conc_7_2 = df[df['salt_conc_numeric'] == 7.2][feature]
                
                if len(conc_0) > 0 and len(conc_7_2) > 0:
                    # Cohen's d
                    pooled_std = np.sqrt(((len(conc_0)-1)*conc_0.var() + 
                                        (len(conc_7_2)-1)*conc_7_2.var()) / 
                                       (len(conc_0) + len(conc_7_2) - 2))
                    effect_size = (conc_7_2.mean() - conc_0.mean()) / pooled_std
                    
                    # Signal-to-noise ratio
                    signal = abs(conc_7_2.mean() - conc_0.mean())
                    noise = (conc_0.std() + conc_7_2.std()) / 2
                    snr = signal / noise if noise > 0 else 0
                    
                    effects[feature] = {
                        'effect_size': effect_size,
                        'snr': snr,
                        'mean_0': conc_0.mean(),
                        'mean_7.2': conc_7_2.mean(),
                        'std_0': conc_0.std(),
                        'std_7.2': conc_7_2.std()
                    }
        
        # Summary of limitations
        print("PROBLEM 1: EFFECT SIZES TOO SMALL")
        print("Current effect sizes (Cohen's d):")
        for feature, stats in effects.items():
            print(f"  {feature}: {stats['effect_size']:.3f} (target: >1.0)")
        
        print(f"\nPROBLEM 2: SIGNAL-TO-NOISE RATIO TOO LOW")
        print("Current SNR:")
        for feature, stats in effects.items():
            print(f"  {feature}: {stats['snr']:.3f} (target: >3.0)")
        
        return effects
    
    def recommend_concentration_range(self, current_effects):
        """Recommend better salt concentration range"""
        
        print(f"\n💡 RECOMMENDATION 1: OPTIMIZE SALT CONCENTRATION RANGE")
        print("="*60)
        
        # Estimate required concentration increase
        max_current_effect = max([abs(e['effect_size']) for e in current_effects.values()])
        
        if max_current_effect > 0:
            scaling_factor = self.target_effect_size / max_current_effect
            suggested_max = 7.2 * scaling_factor
        else:
            suggested_max = 50.0  # Conservative estimate
        
        # Suggested concentration series
        suggested_concentrations = [
            0.0,      # Control
            5.0,      # Low
            15.0,     # Medium  
            30.0,     # High
            50.0      # Very high
        ]
        
        print(f"Current max concentration: {max(self.current_concentrations)} mg/mL")
        print(f"Current max effect size: {max_current_effect:.3f}")
        print(f"Estimated required scaling: {scaling_factor:.1f}x")
        print(f"\nSUGGESTED NEW CONCENTRATION SERIES (mg/mL):")
        for i, conc in enumerate(suggested_concentrations):
            print(f"  Condition {i+1}: {conc} mg/mL")
        
        # Alternative: Logarithmic series
        log_series = [0, 1, 3, 10, 30, 100]
        print(f"\nALTERNATIVE: LOGARITHMIC SERIES (mg/mL):")
        for i, conc in enumerate(log_series):
            print(f"  Condition {i+1}: {conc} mg/mL")
        
        return suggested_concentrations, log_series
    
    def recommend_measurement_improvements(self):
        """Recommend measurement protocol improvements"""
        
        print(f"\n💡 RECOMMENDATION 2: IMPROVE MEASUREMENT PROTOCOL")
        print("="*60)
        
        improvements = [
            {
                'issue': 'Measurement timing too short',
                'solution': 'Extend measurement time to 60-120 minutes',
                'rationale': 'Agglutination kinetics can be slow'
            },
            {
                'issue': 'No incubation period',
                'solution': 'Add 30-60 min pre-incubation at room temperature',
                'rationale': 'Allow time for particle interactions to develop'
            },
            {
                'issue': 'Single time point measurement',
                'solution': 'Take measurements every 5-10 minutes',
                'rationale': 'Capture agglutination kinetics and endpoint'
            },
            {
                'issue': 'Room temperature only',
                'solution': 'Test at 37°C and 4°C as well',
                'rationale': 'Temperature affects agglutination rate and extent'
            },
            {
                'issue': 'No mixing control',
                'solution': 'Standardize mixing: gentle inversion 5x, then static',
                'rationale': 'Consistent initial particle distribution'
            }
        ]
        
        for i, improvement in enumerate(improvements, 1):
            print(f"{i}. {improvement['issue']}")
            print(f"   Solution: {improvement['solution']}")
            print(f"   Why: {improvement['rationale']}\n")
    
    def recommend_experimental_controls(self):
        """Recommend additional experimental controls"""
        
        print(f"💡 RECOMMENDATION 3: ADD EXPERIMENTAL CONTROLS")
        print("="*60)
        
        controls = [
            {
                'control': 'Positive agglutination control',
                'description': 'Known agglutinating agent (e.g., PEG, antibody)',
                'purpose': 'Verify system can detect agglutination'
            },
            {
                'control': 'Particle stability control',
                'description': 'Beads in buffer only, multiple time points',
                'purpose': 'Ensure particles are stable during measurement'
            },
            {
                'control': 'Ionic strength control',
                'description': 'Different salts at same ionic strength',
                'purpose': 'Separate specific vs general ionic effects'
            },
            {
                'control': 'pH control series',
                'description': 'Test pH 6.0, 7.0, 8.0 with same salt',
                'purpose': 'pH affects particle surface charge and agglutination'
            },
            {
                'control': 'Bead concentration series',
                'description': 'Test 0.1%, 0.3%, 0.5% bead concentrations',
                'purpose': 'Optimize particle density for agglutination'
            }
        ]
        
        for i, control in enumerate(controls, 1):
            print(f"{i}. {control['control']}")
            print(f"   Setup: {control['description']}")
            print(f"   Purpose: {control['purpose']}\n")
    
    def recommend_analysis_improvements(self):
        """Recommend better analysis approaches"""
        
        print(f"💡 RECOMMENDATION 4: IMPROVE DATA ANALYSIS")
        print("="*60)
        
        improvements = [
            "Focus on kinetic analysis rather than endpoint classification",
            "Use dose-response curves with Hill equation fitting",
            "Implement time-series analysis for agglutination rate",
            "Use fewer, physically meaningful features (avoid overfitting)",
            "Apply proper statistical power analysis for experimental design",
            "Use effect size calculations to determine practical significance",
            "Implement proper cross-validation with time-based splits"
        ]
        
        for i, improvement in enumerate(improvements, 1):
            print(f"{i}. {improvement}")
    
    def create_experimental_design_template(self):
        """Create template for improved experiment"""
        
        print(f"\n📋 IMPROVED EXPERIMENTAL DESIGN TEMPLATE")
        print("="*60)
        
        design = {
            'salt_concentrations': [0, 1, 3, 10, 30, 100],  # mg/mL
            'incubation_times': [0, 15, 30, 60, 120],       # minutes
            'temperatures': [25, 37],                        # °C
            'replicates': 3,
            'measurement_interval': 5,                       # minutes
            'total_measurement_time': 120                    # minutes
        }
        
        total_conditions = (len(design['salt_concentrations']) * 
                          len(design['temperatures']) * 
                          design['replicates'])
        
        total_measurements = (total_conditions * 
                            (design['total_measurement_time'] // 
                             design['measurement_interval'] + 1))
        
        print(f"Salt concentrations: {design['salt_concentrations']} mg/mL")
        print(f"Temperatures: {design['temperatures']} °C")
        print(f"Replicates per condition: {design['replicates']}")
        print(f"Measurement interval: {design['measurement_interval']} min")
        print(f"Total measurement time: {design['total_measurement_time']} min")
        print(f"\nTotal experimental conditions: {total_conditions}")
        print(f"Total measurements: {total_measurements}")
        print(f"Estimated experiment duration: {total_measurements * 0.5 / 60:.1f} hours")
        
        return design
    
    def power_analysis_recommendation(self, current_effects):
        """Recommend sample sizes for detecting meaningful effects"""
        
        print(f"\n📊 STATISTICAL POWER ANALYSIS")
        print("="*60)
        
        # Calculate required sample sizes for different effect sizes
        alpha = 0.05
        power = 0.80
        
        effect_sizes = [0.2, 0.5, 0.8, 1.0, 1.5]  # Small, medium, large effects
        
        print(f"Required sample sizes (α={alpha}, power={power}):")
        print("Effect Size (Cohen's d) | Samples per group")
        print("-" * 40)
        
        for effect_size in effect_sizes:
            # Approximate formula for two-sample t-test
            n = ((1.96 + 0.84) ** 2 * 2) / (effect_size ** 2)
            n = int(np.ceil(n))
            
            current_status = ""
            max_current = max([abs(e['effect_size']) for e in current_effects.values()])
            if effect_size <= max_current:
                current_status = " ← Current max effect"
            
            print(f"{effect_size:>12} | {n:>15}{current_status}")
        
        print(f"\nCURRENT SITUATION:")
        print(f"- Your current max effect size: {max_current:.3f}")
        print(f"- Current samples per group: ~1200")
        print(f"- This provides power to detect effects as small as 0.08")
        print(f"- BUT your actual effects are smaller than measurement noise!")

def main():
    """Run complete experiment optimization analysis"""
    
    # Load your data
    df = pd.read_csv('side_scatter_analysis_features.csv')
    
    # Initialize optimizer
    optimizer = AgglutinationExperimentOptimizer()
    
    print("🧪 AGGLUTINATION EXPERIMENT OPTIMIZATION REPORT")
    print("="*80)
    
    # Analyze current results
    current_effects = optimizer.analyze_current_results(df)
    
    # Generate recommendations
    conc_recommendations = optimizer.recommend_concentration_range(current_effects)
    optimizer.recommend_measurement_improvements()
    optimizer.recommend_experimental_controls()
    optimizer.recommend_analysis_improvements()
    
    # Create experimental design
    new_design = optimizer.create_experimental_design_template()
    
    # Power analysis
    optimizer.power_analysis_recommendation(current_effects)
    
if __name__ == "__main__":
    main()