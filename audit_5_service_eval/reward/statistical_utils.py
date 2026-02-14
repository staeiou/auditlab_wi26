"""
Statistical Testing Utilities for AI Perturbation Bias Experiments
================================================================

This module provides modular, reusable functions for statistical testing
across different experimental designs. Specify your experiment variables 
and the functions should work across different audits.

Author: Rosey Gutierrez
Date: 2026-01-29
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro, levene, kruskal, f_oneway, alexandergovern
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_sample_sizes(results, variables):
    """
    Print sample size tables for Model and all specified variables.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Data containing 'Model' column and specified variables
    variables : list
        List of categorical variables to summarize (e.g., ['Ethnicity', 'Gender'])
    
    Returns:
    --------
    dict
        Dictionary of DataFrames with sample sizes for each variable
    """
    print("="*80)
    print("SAMPLE SIZE SUMMARY TABLES")
    print("="*80)
    print()
    
    summary_tables = {}
    counter = 1
    
    # Always include Model first
    print(f"{counter}. Sample Sizes by Model:")
    print("-"*40)
    model_counts = pd.DataFrame({
        'Count': results['Model'].value_counts().sort_index(),
        'Percentage': (results['Model'].value_counts(normalize=True) * 100).sort_index()
    })
    model_counts['Percentage'] = model_counts['Percentage'].round(2)
    print(model_counts)
    print()
    summary_tables['Model'] = model_counts
    counter += 1
    
    # Then iterate through all specified variables
    for variable in variables:
        if variable in results.columns:
            print(f"{counter}. Sample Sizes by {variable}:")
            print("-"*40)
            
            var_counts = pd.DataFrame({
                'Count': results[variable].value_counts().sort_index(),
                'Percentage': (results[variable].value_counts(normalize=True) * 100).sort_index()
            })
            var_counts['Percentage'] = var_counts['Percentage'].round(2)
            print(var_counts)
            print()
            
            summary_tables[variable] = var_counts
            counter += 1
        else:
            print(f"{counter}. Sample Sizes by {variable}:")
            print("-"*40)
            print(f"⚠ Variable '{variable}' not found in dataset")
            print()
            counter += 1
    
    print("="*80)
    print(f"Summary: {counter-1} tables generated")
    print("="*80)
    
    return summary_tables

# ============================================================================
# ASSUMPTION TESTING FUNCTIONS
# ============================================================================

def test_normality_by_model(results, outcome_var='Parsed'):
    """
    Test normality for each model using Shapiro-Wilk test.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Data containing 'Model' column and outcome variable
    outcome_var : str
        Name of the outcome variable (default: 'Parsed')
    
    Returns:
    --------
    pd.DataFrame
        Normality test results for each model
    """
    print("="*80)
    print("NORMALITY TESTING (Shapiro-Wilk Test)")
    print("="*80)
    print()
    print("H0: Data is normally distributed")
    print("H1: Data is NOT normally distributed")
    print("Reject H0 if p < 0.05")
    print()
    
    normality_results = []
    
    for model in sorted(results['Model'].unique()):
        model_data = results[results['Model'] == model][outcome_var].dropna()
        
        # Shapiro-Wilk test
        stat, p_value = shapiro(model_data)
        is_normal = p_value > 0.05
        
        normality_results.append({
            'Model': model,
            'Statistic': stat,
            'P-value': p_value,
            'Normal?': 'Yes' if is_normal else 'No'
        })
        
        print(f"{model}:")
        print(f"  Statistic: {stat:.6f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Normal? {normality_results[-1]['Normal?']}")
        print()
    
    normality_df = pd.DataFrame(normality_results)
    print("\nSummary:")
    print(normality_df)
    print()
    
    return normality_df


def test_variance_homogeneity(results, variables, outcome_var='Parsed'):
    """
    Test homogeneity of variance using Levene's test.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Data containing 'Model', outcome variable, and test variables
    variables : list
        List of categorical variables to test (e.g., ['Ethnicity', 'Gender'])
    outcome_var : str
        Name of the outcome variable (default: 'Parsed')
    
    Returns:
    --------
    pd.DataFrame
        Levene's test results for each model-variable combination
    """
    print("="*80)
    print("HOMOGENEITY OF VARIANCE TESTING (Levene's Test)")
    print("="*80)
    print()
    print("H0: All groups have equal variances")
    print("H1: At least one group has different variance")
    print("Reject H0 if p < 0.05")
    print()
    
    levene_results = []
    
    for variable in variables:
        print(f"Testing variance equality across {variable.upper()} groups within each model:")
        print("-"*80)
        
        for model in sorted(results['Model'].unique()):
            model_data = results[results['Model'] == model]
            
            # Check if variable exists
            if variable not in model_data.columns:
                print(f"{model}: Variable '{variable}' not found - skipping")
                continue
            
            # Prepare groups for Levene's test
            groups = []
            for category in model_data[variable].unique():
                group_data = model_data[model_data[variable] == category][outcome_var].dropna()
                if len(group_data) > 0:
                    groups.append(group_data)
            
            if len(groups) > 1:
                stat, p_value = levene(*groups)
                equal_var = p_value > 0.05
                
                levene_results.append({
                    'Model': model,
                    'Variable': variable,
                    'Statistic': stat,
                    'P-value': p_value,
                    'Equal Variance?': 'Yes' if equal_var else 'No'
                })
                
                print(f"{model}:")
                print(f"  Statistic: {stat:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Equal variance? {levene_results[-1]['Equal Variance?']}")
                print()
        
        print()
    
    levene_df = pd.DataFrame(levene_results)
    print("Summary:")
    print(levene_df)
    print()
    
    return levene_df


def summarize_assumptions(normality_df, levene_df):
    """
    Summarize assumption testing results and provide recommendations.
    
    Parameters:
    -----------
    normality_df : pd.DataFrame
        Results from test_normality_by_model()
    levene_df : pd.DataFrame
        Results from test_variance_homogeneity()
    
    Returns:
    --------
    None (prints summary)
    """
    print("="*80)
    print("ASSUMPTION TESTING SUMMARY")
    print("="*80)
    print()
    
    print("NORMALITY:")
    print("-"*80)
    normal_count = (normality_df['Normal?'] == 'Yes').sum()
    print(f"Models meeting normality assumption: {normal_count}/{len(normality_df)}")
    if normal_count < len(normality_df):
        print("\nModels with non-normal distributions:")
        for _, row in normality_df[normality_df['Normal?'] == 'No'].iterrows():
            print(f"  - {row['Model']} (p = {row['P-value']:.6f})")
    else:
        print("✓ All models meet normality assumption")
    
    print("\n\nHOMOGENEITY OF VARIANCE:")
    print("-"*80)
    equal_var_count = (levene_df['Equal Variance?'] == 'Yes').sum()
    print(f"Tests meeting variance assumption: {equal_var_count}/{len(levene_df)}")
    if equal_var_count < len(levene_df):
        print("\nTests with unequal variances:")
        for _, row in levene_df[levene_df['Equal Variance?'] == 'No'].iterrows():
            print(f"  - {row['Model']} ({row['Variable']}): p = {row['P-value']:.6f}")
    else:
        print("✓ All tests meet variance homogeneity assumption")
    
    print("\n\nRECOMMENDATION:")
    print("-"*80)
    if normal_count >= len(normality_df) * 0.5 and equal_var_count >= len(levene_df) * 0.5:
        print("✓ Proceed with parametric ANOVA tests")
        print("  Most assumptions are met or ANOVA is robust to minor violations")
    else:
        print("⚠ Consider non-parametric alternatives for models with severe violations")
        print("  We'll run both parametric and non-parametric tests for comparison")


def verify_sample_sizes(results, variables, outcome_var='Parsed', min_threshold=30):
    """
    Verify sample sizes per group for ANOVA robustness.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Data containing 'Model', outcome variable, and test variables
    variables : list
        List of categorical variables to check (e.g., ['Ethnicity', 'Gender'])
    outcome_var : str
        Name of outcome variable (default: 'Parsed')
    min_threshold : int
        Minimum recommended sample size per group (default: 30)
    
    Returns:
    --------
    pd.DataFrame
        Summary of sample sizes for each model and variable
    """
    print("="*80)
    print("SAMPLE SIZE VERIFICATION")
    print("="*80)
    print()
    print(f"Checking if we have sufficient sample sizes for ANOVA robustness (n > {min_threshold})...")
    print()
    
    sample_size_data = []
    
    for model in sorted(results['Model'].unique()):
        model_data = results[results['Model'] == model].dropna(subset=[outcome_var])
        
        print(f"{model}:")
        print(f"  Total n: {len(model_data)}")
        
        min_sizes = []
        all_sufficient = True
        
        # Check each variable
        for variable in variables:
            if variable in model_data.columns:
                var_counts = model_data[variable].value_counts()
                min_size = var_counts.min()
                min_sizes.append(min_size)
                
                print(f"  Min group size ({variable}): {min_size}")
                
                # Track for summary
                sample_size_data.append({
                    'Model': model,
                    'Variable': variable,
                    'Min_Group_Size': min_size,
                    'Total_Groups': len(var_counts),
                    'Sufficient': 'Yes' if min_size >= min_threshold else 'No'
                })
                
                if min_size < min_threshold:
                    all_sufficient = False
            else:
                print(f"  ⚠ Variable '{variable}' not found in data")
                all_sufficient = False
        
        # Overall assessment for this model
        if all_sufficient and len(min_sizes) > 0:
            print(f"  ✓ Sufficient for robust ANOVA (all groups > {min_threshold})")
        else:
            print(f"  ⚠ Some small groups - be cautious with ANOVA assumptions")
        print()
    
    # Create summary DataFrame
    sample_size_df = pd.DataFrame(sample_size_data)
    
    print("="*80)
    print("SAMPLE SIZE SUMMARY TABLE")
    print("="*80)
    print(sample_size_df)
    print()
    
    # Overall conclusion
    print("="*80)
    print("CONCLUSION:")
    print("="*80)
    
    insufficient_count = (sample_size_df['Sufficient'] == 'No').sum()
    total_checks = len(sample_size_df)
    
    if insufficient_count == 0:
        print("✓ All groups have sufficient sample sizes (n > {})".format(min_threshold))
        print("  ANOVA is robust to normality violations with these sample sizes.")
        print("  We can proceed with parametric tests confidently.")
    elif insufficient_count < total_checks * 0.25:  # Less than 25% insufficient
        print(f"⚠ {insufficient_count}/{total_checks} checks have small groups")
        print(f"  Most groups are sufficient (n > {min_threshold})")
        print("  Proceed with parametric tests but also run non-parametric tests.")
    else:
        print(f"⚠⚠ {insufficient_count}/{total_checks} checks have small groups")
        print(f"  Many groups below recommended size (n < {min_threshold})")
        print("  Prioritize non-parametric tests and use parametric with caution.")
    
    print("="*80)
    
    return sample_size_df


# ============================================================================
# ANOVA FUNCTIONS
# ============================================================================

def run_factorial_anova(results, variables, outcome_var='Parsed', include_interactions=True):
    """
    Run factorial ANOVA for each model with specified variables.
    """
    print("="*80)
    print("FACTORIAL ANOVA BY MODEL")
    print("="*80)
    print()
    
    anova_results = {}
    
    for model in sorted(results['Model'].unique()):
        model_data = results[results['Model'] == model].dropna(subset=[outcome_var])
        
        # CHECK FOR ZERO VARIANCE FIRST
        overall_variance = model_data[outcome_var].var()
        unique_values = model_data[outcome_var].nunique()
        
        if overall_variance == 0 or unique_values == 1:
            print(f"\n{'='*80}")
            print(f"MODEL: {model}")
            print(f"{'='*80}")
            print(f"Sample size: {len(model_data)}")
            print()
            print(f"⚠⚠ ZERO VARIANCE DETECTED ⚠⚠")
            print(f"  All observations have identical scores: {model_data[outcome_var].iloc[0]:.2f}")
            print(f"  No differential treatment across any variables")
            print(f"  ANOVA cannot be computed (or would be meaningless)")
            print(f"  → Excluding this model from bias analysis")
            print()
            
            anova_results[model] = {
                'anova_table': None,
                'model_fit': None,
                'sample_size': len(model_data),
                'zero_variance': True,
                'constant_score': model_data[outcome_var].iloc[0]
            }
            continue
        
        # Build formula (rest of the function continues as before)
        main_effects = [f'C({var})' for var in variables]
        formula_parts = [outcome_var] + ['~'] + [' + '.join(main_effects)]
        
        if include_interactions and len(variables) >= 2:
            interactions = []
            for i in range(len(variables)):
                for j in range(i+1, len(variables)):
                    interactions.append(f'C({variables[i]}):C({variables[j]})')
            formula_parts.append(' + ' + ' + '.join(interactions))
        
        formula = ''.join(formula_parts)
        
        # Check if all variables exist
        missing_vars = [var for var in variables if var not in model_data.columns]
        if missing_vars:
            print(f"\n{model}: Missing variables {missing_vars} - skipping")
            anova_results[model] = None
            continue
        
        print(f"\n{'='*80}")
        print(f"MODEL: {model}")
        print(f"{'='*80}")
        print(f"Sample size: {len(model_data)}")
        print(f"Score variance: {overall_variance:.4f}")
        print()
        
        try:
            model_fit = ols(formula, data=model_data).fit()
            anova_table = anova_lm(model_fit, typ=2)
            
            anova_results[model] = {
                'anova_table': anova_table,
                'model_fit': model_fit,
                'sample_size': len(model_data),
                'zero_variance': False
            }
            
            print("ANOVA Table:")
            print(anova_table)
            print()
            
            # Highlight significant effects
            print("Significant effects (p < 0.05):")
            sig_effects = anova_table[anova_table['PR(>F)'] < 0.05]
            if len(sig_effects) > 0:
                for effect in sig_effects.index:
                    if effect != 'Residual':
                        print(f"  ✓ {effect}: F = {anova_table.loc[effect, 'F']:.4f}, p = {anova_table.loc[effect, 'PR(>F)']:.6f}")
            else:
                print("  (No significant effects)")
        
        except Exception as e:
            print(f"  ⚠ Could not fit model: {e}")
            anova_results[model] = None
    
    print("\n" + "="*80)
    print("ANOVA ANALYSIS COMPLETE")
    print("="*80)
    
    # Summary of zero-variance models
    zero_var_models = [m for m, r in anova_results.items() if r is not None and r.get('zero_variance', False)]
    if len(zero_var_models) > 0:
        print(f"\n⚠ {len(zero_var_models)} model(s) excluded due to zero variance:")
        for model in zero_var_models:
            score = anova_results[model]['constant_score']
            print(f"  - {model}: Always assigns score of {score:.2f}")
        print("\nThese models show no differential treatment and should be excluded from bias analysis.")
    
    return anova_results


def calculate_effect_sizes(anova_results):
    """
    Calculate eta-squared effect sizes from ANOVA results.
    
    Parameters:
    -----------
    anova_results : dict
        Dictionary from run_factorial_anova()
    
    Returns:
    --------
    pd.DataFrame
        Effect sizes for each model and effect
    """
    print("="*80)
    print("EFFECT SIZES (ETA SQUARED)")
    print("="*80)
    print()
    
    effect_sizes_summary = []
    
    for model, result in anova_results.items():
        if result is not None:
            anova_table = result['anova_table']
            
            print(f"\n{model}:")
            print("-"*80)
            
            # Calculate eta squared for each effect
            total_ss = anova_table['sum_sq'].sum()
            
            for effect in anova_table.index:
                if effect != 'Residual':
                    eta_sq = anova_table.loc[effect, 'sum_sq'] / total_ss
                    p_value = anova_table.loc[effect, 'PR(>F)']
                    
                    # Interpret effect size
                    if eta_sq < 0.01:
                        interpretation = 'negligible'
                    elif eta_sq < 0.06:
                        interpretation = 'small'
                    elif eta_sq < 0.14:
                        interpretation = 'medium'
                    else:
                        interpretation = 'large'
                    
                    print(f"  {effect}:")
                    print(f"    eta² = {eta_sq:.4f} ({interpretation})")
                    print(f"    p = {p_value:.6f}")
                    
                    effect_sizes_summary.append({
                        'Model': model,
                        'Effect': effect,
                        'Eta_Squared': eta_sq,
                        'P_value': p_value,
                        'Interpretation': interpretation
                    })
    
    effect_sizes_df = pd.DataFrame(effect_sizes_summary)
    print("\n" + "="*80)
    print("Effect sizes calculated for all models")
    print("="*80)
    
    return effect_sizes_df


# ============================================================================
# POST-HOC TESTING FUNCTIONS
# ============================================================================

def run_posthoc_tests(results, anova_results, variables, outcome_var='Parsed', alpha=0.05):
    """
    Run Tukey HSD post-hoc tests for significant main effects.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Original data
    anova_results : dict
        Dictionary from run_factorial_anova()
    variables : list
        List of categorical variables tested
    outcome_var : str
        Name of outcome variable
    alpha : float
        Significance level (default: 0.05)
    
    Returns:
    --------
    dict
        Dictionary of post-hoc results
    """
    print("="*80)
    print("POST-HOC TESTS: PAIRWISE COMPARISONS (Tukey HSD)")
    print("="*80)
    print()
    
    posthoc_results = {}
    
    for variable in variables:
        print(f"\n{'='*80}")
        print(f"TESTING: {variable.upper()}")
        print(f"{'='*80}")
        
        for model in sorted(results['Model'].unique()):
            if anova_results[model] is not None:
                anova_table = anova_results[model]['anova_table']
                effect_name = f'C({variable})'
                
                # Check if variable exists and is significant
                if effect_name in anova_table.index:
                    p_value = anova_table.loc[effect_name, 'PR(>F)']
                    
                    if p_value < alpha:
                        print(f"\n{model}:")
                        print("-"*80)
                        print(f"{variable} main effect: p = {p_value:.6f} (significant)")
                        print()
                        
                        model_data = results[results['Model'] == model].dropna(subset=[outcome_var])
                        
                        if variable in model_data.columns:
                            # Tukey HSD test
                            tukey = pairwise_tukeyhsd(endog=model_data[outcome_var], 
                                                     groups=model_data[variable], 
                                                     alpha=alpha)
                            
                            print(tukey)
                            print()
                            
                            # Store results
                            if model not in posthoc_results:
                                posthoc_results[model] = {}
                            posthoc_results[model][variable] = tukey
                            
                            # Summarize significant differences
                            tukey_df = pd.DataFrame(data=tukey.summary().data[1:], 
                                                   columns=tukey.summary().data[0])
                            sig_pairs = tukey_df[tukey_df['reject'] == True]
                            
                            if len(sig_pairs) > 0:
                                print(f"Significant pairwise differences ({len(sig_pairs)} pairs):")
                                for _, row in sig_pairs.iterrows():
                                    print(f"  {row['group1']} vs {row['group2']}: "
                                          f"diff = {row['meandiff']:.2f}, p = {row['p-adj']:.6f}")
                            else:
                                print("No significant pairwise differences after correction")
                    else:
                        print(f"\n{model}: {variable} not significant (p = {p_value:.6f}), skipping post-hoc")
    
    print("\n" + "="*80)
    print("POST-HOC TESTING COMPLETE")
    print("="*80)
    
    return posthoc_results


# ============================================================================
# NON-PARAMETRIC TESTING FUNCTIONS
# ============================================================================

def run_kruskal_wallis(results, variables, outcome_var='Parsed'):
    """
    Run Kruskal-Wallis test for each variable across all models.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Data containing 'Model', outcome variable, and test variables
    variables : list
        List of categorical variables to test
    outcome_var : str
        Name of outcome variable (default: 'Parsed')
    
    Returns:
    --------
    pd.DataFrame
        Kruskal-Wallis test results
    """
    print("="*80)
    print("KRUSKAL-WALLIS TEST (Non-parametric)")
    print("="*80)
    print()
    print("This test doesn't assume normality or equal variances")
    print()
    
    kw_results = []
    
    for variable in variables:
        print(f"Testing {variable.upper()} effects:")
        print("-"*80)
        
        for model in sorted(results['Model'].unique()):
            model_data = results[results['Model'] == model].dropna(subset=[outcome_var])
            
            # Check if variable exists
            if variable not in model_data.columns:
                print(f"{model}: Variable '{variable}' not found - skipping")
                print()
                continue
            
            # Check if all values are identical (zero variance)
            all_values = model_data[outcome_var].values
            if len(np.unique(all_values)) == 1:
                print(f"{model}:")
                print(f"  ⚠ All values identical (score = {all_values[0]:.1f}) - cannot compute")
                print(f"  → Model assigns same score to all observations")
                kw_results.append({
                    'Model': model,
                    'Variable': variable,
                    'H-statistic': np.nan,
                    'P-value': np.nan,
                    'Significant': 'Cannot compute - zero variance'
                })
                print()
                continue
            
            try:
                # Prepare groups
                groups = []
                for category in sorted(model_data[variable].unique()):
                    group_data = model_data[model_data[variable] == category][outcome_var]
                    groups.append(group_data)
                
                # Kruskal-Wallis test
                stat, p_value = kruskal(*groups)
                
                kw_results.append({
                    'Model': model,
                    'Variable': variable,
                    'H-statistic': stat,
                    'P-value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
                
                print(f"{model}:")
                print(f"  H-statistic: {stat:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Significant? {kw_results[-1]['Significant']}")
                print()
                
            except ValueError as e:
                print(f"{model}:")
                print(f"  ⚠ Error: {e}")
                kw_results.append({
                    'Model': model,
                    'Variable': variable,
                    'H-statistic': np.nan,
                    'P-value': np.nan,
                    'Significant': 'Cannot compute - error'
                })
                print()
        
        print()
    
    kw_df = pd.DataFrame(kw_results)
    print("="*80)
    print("Kruskal-Wallis Test Summary:")
    print("="*80)
    print(kw_df)
    print()
    print("Note: 'Cannot compute' indicates zero variance or identical values.")
    print("This is itself a finding - complete lack of differentiation.")
    
    return kw_df


def run_welch_anova(results, levene_df, outcome_var='Parsed'):
    """
    Run Welch's ANOVA (or Alexander-Govern test) for models with unequal variances.
    
    Parameters:
    -----------
    results : pd.DataFrame
        Original data
    levene_df : pd.DataFrame
        Results from test_variance_homogeneity()
    outcome_var : str
        Name of outcome variable
    
    Returns:
    --------
    pd.DataFrame
        Welch's ANOVA results
    """
    print("="*80)
    print("WELCH'S ANOVA (Robust to Unequal Variances)")
    print("="*80)
    print()
    print("Running for model-variable combinations that failed Levene's test (p < 0.05)")
    print()
    
    # Dynamically identify which models need Welch's ANOVA
    models_to_test = {}
    
    for _, row in levene_df.iterrows():
        if row['Equal Variance?'] == 'No':
            model = row['Model']
            variable = row['Variable']
            
            if model not in models_to_test:
                models_to_test[model] = []
            models_to_test[model].append(variable)
    
    print(f"Models requiring Welch's ANOVA: {len(models_to_test)}")
    for model, variables in models_to_test.items():
        print(f"  - {model}: {', '.join(variables)}")
    print()
    
    if len(models_to_test) == 0:
        print("✓ No models require Welch's ANOVA - all passed Levene's test")
        return pd.DataFrame()
    
    welch_results = []
    
    for model, variables in models_to_test.items():
        print(f"\n{model}:")
        print("-"*80)
        
        model_data = results[results['Model'] == model].dropna(subset=[outcome_var])
        
        for variable in variables:
            print(f"\n  Testing {variable}:")
            
            if variable not in model_data.columns:
                print(f"    Variable not found - skipping")
                continue
            
            # Prepare groups
            groups = []
            for category in sorted(model_data[variable].unique()):
                group_data = model_data[model_data[variable] == category][outcome_var].values
                if len(group_data) > 0:
                    groups.append(group_data)
            
            # Check for zero variance
            zero_var_groups = []
            for i, group in enumerate(groups):
                var = np.var(group, ddof=1)
                if var == 0 or np.isnan(var):
                    zero_var_groups.append(i)
            
            if len(zero_var_groups) > 0:
                print(f"    ⚠ Zero variance detected - cannot compute")
                welch_results.append({
                    'Model': model,
                    'Variable': variable,
                    'Statistic': np.nan,
                    'P-value': np.nan,
                    'Significant': 'Cannot compute',
                    'Note': 'Zero variance'
                })
                continue
            
            try:
                # Alexander-Govern test (robust to both issues)
                result = alexandergovern(*groups)
                stat = result.statistic
                p_value = result.pvalue
                
                welch_results.append({
                    'Model': model,
                    'Variable': variable,
                    'Statistic': stat,
                    'P-value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No',
                    'Note': ''
                })
                
                print(f"    Statistic: {stat:.4f}")
                print(f"    P-value: {p_value:.6f}")
                print(f"    Result: {'Significant' if p_value < 0.05 else 'Not significant'}")
                
            except Exception as e:
                print(f"    Could not compute: {e}")
                welch_results.append({
                    'Model': model,
                    'Variable': variable,
                    'Statistic': np.nan,
                    'P-value': np.nan,
                    'Significant': 'Error',
                    'Note': str(e)
                })
    
    welch_df = pd.DataFrame(welch_results)
    print("\n\nWelch's ANOVA Summary:")
    print(welch_df)
    
    return welch_df


# ============================================================================
# SUMMARY & REPORTING FUNCTIONS
# ============================================================================

def create_summary_table(anova_results, effect_sizes_df, variables):
    """
    Create comprehensive summary table of all statistical results.
    
    Parameters:
    -----------
    anova_results : dict
        Dictionary from run_factorial_anova()
    effect_sizes_df : pd.DataFrame
        DataFrame from calculate_effect_sizes()
    variables : list
        List of variables tested
    
    Returns:
    --------
    pd.DataFrame
        Summary table
    """
    print("="*80)
    print("COMPREHENSIVE STATISTICAL SUMMARY")
    print("="*80)
    print()

    # Filter out zero-variance models
    valid_models = {k: v for k, v in anova_results.items() 
                    if v is not None and not v.get('zero_variance', False)}
    
    if len(valid_models) < len(anova_results):
        excluded = len(anova_results) - len(valid_models)
        print(f"Note: {excluded} model(s) excluded from analysis due to zero variance")
        print()
    
    summary_data = []
    
    for model in sorted(anova_results.keys()):
        if anova_results[model] is not None:
            anova_table = anova_results[model]['anova_table']
            
            row_data = {'Model': model}
            
            # For each variable, get p-value, effect size, and significance
            for variable in variables:
                effect_name = f'C({variable})'
                
                if effect_name in anova_table.index:
                    p_value = anova_table.loc[effect_name, 'PR(>F)']
                    
                    # Get effect size
                    effect_size_row = effect_sizes_df[
                        (effect_sizes_df['Model'] == model) & 
                        (effect_sizes_df['Effect'] == effect_name)
                    ]
                    
                    if len(effect_size_row) > 0:
                        eta2 = effect_size_row.iloc[0]['Eta_Squared']
                    else:
                        eta2 = 0
                    
                    row_data[f'{variable}_p'] = p_value
                    row_data[f'{variable}_eta2'] = eta2
                    row_data[f'{variable}_Sig'] = '✓' if p_value < 0.05 else '✗'
                else:
                    row_data[f'{variable}_p'] = np.nan
                    row_data[f'{variable}_eta2'] = np.nan
                    row_data[f'{variable}_Sig'] = '—'
            
            summary_data.append(row_data)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Format for display
    pd.options.display.float_format = '{:.4f}'.format
    print(summary_df.to_string(index=False))
    print()
    
    # Count significant effects
    print("\nSUMMARY BY MODEL:")
    print("-"*80)
    for _, row in summary_df.iterrows():
        sig_cols = [col for col in summary_df.columns if col.endswith('_Sig')]
        sig_count = sum([row[col] == '✓' for col in sig_cols])
        print(f"{row['Model']}: {sig_count}/{len(variables)} significant main effects")
        
        if sig_count > 0:
            effects = []
            for variable in variables:
                if row[f'{variable}_Sig'] == '✓':
                    effects.append(f"{variable} (eta²={row[f'{variable}_eta2']:.3f})")
            print(f"  Significant: {', '.join(effects)}")
    
    return summary_df


def rank_models_by_bias(summary_df, variables):
    """
    Rank models by total bias (sum of significant effect sizes).
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        DataFrame from create_summary_table()
    variables : list
        List of variables tested
    
    Returns:
    --------
    pd.DataFrame
        Ranked models
    """
    print("\n" + "="*80)
    print("MODELS RANKED BY TOTAL BIAS")
    print("="*80)
    print()
    
    ranking_data = []
    
    for _, row in summary_df.iterrows():
        total_eta2 = 0
        sig_count = 0
        
        for variable in variables:
            if row[f'{variable}_Sig'] == '✓':
                total_eta2 += row[f'{variable}_eta2']
                sig_count += 1
        
        ranking_data.append({
            'Model': row['Model'],
            'Total_eta2': total_eta2,
            'Sig_Effects': sig_count
        })
    
    ranking_df = pd.DataFrame(ranking_data).sort_values('Total_eta2', ascending=False)
    
    print("Rank | Model                          | Significant Effects | Total eta-squared")
    print("-"*80)
    for i, row in enumerate(ranking_df.itertuples(), 1):
        print(f"{i:2d}   | {row.Model:30s} | {row.Sig_Effects}/{len(variables)}             | {row.Total_eta2:.4f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("-"*80)
    print("Higher total eta-squared = Stronger overall demographic bias")
    print("More significant effects = More types of bias present")
    
    return ranking_df
