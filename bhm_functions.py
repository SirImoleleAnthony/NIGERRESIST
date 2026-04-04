# Bayesian Hierarchical Modeling (BHM) Analysis 

import pandas as pd
import numpy as np
import streamlit as st
import os
from pathlib import Path


# ============================================================================
# 1. NRRS & GENOTYPE-PHENOTYPE DISCORDANCE DETECTION
# ============================================================================

def load_nrrs_analysis(antibiotic_name):
    """Load and interpret NRRS predictions and discordance data for an antibiotic."""
    bhm_path = f"bhm_results/{antibiotic_name}"
    
    try:
        # Load main prediction data
        nrrs_pred = pd.read_csv(f"{bhm_path}/nrrs_predictions.csv")
        nrrs_model = pd.read_csv(f"{bhm_path}/nrrs_model_summary.csv", index_col=0)
        discord_species = pd.read_csv(f"{bhm_path}/discordance_by_species.csv")
        discord_isolates = pd.read_csv(f"{bhm_path}/discordant_isolates.csv")
        
        return {
            'predictions': nrrs_pred,
            'model_summary': nrrs_model,
            'discord_by_species': discord_species,
            'discord_isolates': discord_isolates
        }
    except FileNotFoundError as e:
        return None


def generate_nrrs_observation(antibiotic_name, nrrs_data):

    if nrrs_data is None:
        return f"NRRS data for {antibiotic_name} not available.", {}
    
    nrrs_scores = nrrs_data['predictions']['Nigerian_Resistance_Risk_Score_NRRS']
    discord_by_species = nrrs_data['discord_by_species']
    discord_isolates = nrrs_data['discord_isolates'] if 'discord_isolates' in nrrs_data else None
    
    # Convergence check
    model_summary = nrrs_data['model_summary']
    has_convergence_issue = (model_summary['r_hat'] > 1.01).sum() > 0 if 'r_hat' in model_summary.columns else False
    
    # NRRS Risk Distribution Analysis
    p90_threshold = nrrs_scores.quantile(0.90)
    high_risk = (nrrs_scores >= 0.75).sum()  # Primary threshold: NRRS ≥ 0.75
    top_10_percentile = (nrrs_scores >= p90_threshold).sum()
    
    mean_nrrs = nrrs_scores.mean()
    median_nrrs = nrrs_scores.median()
    std_nrrs = nrrs_scores.std()
    
    # Posterior distribution characteristics
    hdi_lower = nrrs_scores.quantile(0.025)
    hdi_upper = nrrs_scores.quantile(0.975)
    
    # Discordance overview
    total_discord = discord_by_species['Discordant_Count'].sum()
    total_isolates = discord_by_species['Total_Isolates'].sum()
    discord_rate = (total_discord / total_isolates) * 100 if total_isolates > 0 else 0
    
    # Top discordant species
    top_discord_species = discord_by_species.nlargest(3, 'Discordance_Rate_%')
    
    # Flag: Predicted resistant (>90%) but observed susceptible
    high_predicted_discord = 0
    if discord_isolates is not None and 'Predicted_Resistance_Probability' in discord_isolates.columns:
        high_predicted_discord = ((discord_isolates['Predicted_Resistance_Probability'] > 0.90) & 
                                   (discord_isolates['Observed_Phenotype'] == 'Susceptible')).sum()
    
    observation = (
        f"## Nigerian Resistance Risk Score (NRRS) for {antibiotic_name}\n\n"
        f"### Bayesian Health Check\n"
        f"**Model Convergence**: {'⚠️ WARNING - R-hat > 1.01 detected. Results may be unreliable.' if has_convergence_issue else '✓ All parameters converged (R-hat < 1.01).'}\n"
        f"**Posterior Predictive Distribution**: "
        f"Mean NRRS = {mean_nrrs:.3f}, Median = {median_nrrs:.3f} (SD = {std_nrrs:.3f}). "
        f"95% HDI: [{hdi_lower:.3f}, {hdi_upper:.3f}].\n\n"
        
        f"### Risk Stratification\n"
        f"**High-Risk Isolates (NRRS ≥ 0.75):** {high_risk}/{len(nrrs_scores)} isolates ({(high_risk/len(nrrs_scores)*100):.1f}%). "
        f"These isolates require alternative therapeutics or combination therapy.\n"
        f"**Top 10th Percentile (P90):** {top_10_percentile} isolates with NRRS ≥ {p90_threshold:.3f}. "
        f"Cross-reference with species and study hierarchy for localized vs. widespread clusters.\n\n"
        
        f"### Gene Burden Analysis\n"
        f"*(If available: Examine coefficient weight—does one additional gene increase risk linearly or synergistically?)*\n"
        f"Specific gene combinations (e.g., NDM + OXA) may have exponential risk rather than additive.\n\n"
    )
    
    observation += (
        f"## Genotype-Phenotype Discordance Detection\n\n"
        f"**Overall Network Discordance Rate:** {discord_rate:.2f}% ({total_discord}/{total_isolates} isolates).\n"
        f"This includes gene-present-but-susceptible (potential silencing) and gene-absent-but-resistant (cryptic mechanisms).\n"
        f"**Critical Flag**: {high_predicted_discord} isolates show >90% predicted resistance but observed susceptibility. "
        f"**Interpretation**: Potential gene silencing, quality control issues, or regulatory mutations.\n\n"
        
        f"**Species with Highest Discordance:**\n"
    )
    
    for idx, row in top_discord_species.iterrows():
        observation += (
            f"- **{row['Species']}**: {row['Discordant_Count']}/{row['Total_Isolates']} ({row['Discordance_Rate_%']:.1f}%). "
        )
        if row['Discordance_Rate_%'] > 20:
            observation += "**High discordance—investigate lab-specific calibration or biological heterogeneity.** "
        observation += "\n"
    
    observation += (
        f"\n**Nigerian Context**: If discordances cluster within specific studies, this suggests laboratory-specific biases "
        f"(e.g., culture conditions, resistance cassette detection loss) rather than true biological variation. "
        f"Recommend quality assurance audit and cross-validation with reference labs."
    )
    
    
    implications = {
        'policymakers': (
            f"Implement genome-based surveillance with phenotypic confirmation for {antibiotic_name}. "
            f"The {discord_rate:.1f}% discordance rate highlights that molecular markers alone are insufficient; "
            f"culture and susceptibility testing remain essential for treatment decisions in Nigeria."
        ),
        'clinicians': (
            f"For {antibiotic_name}, do not rely solely on rapid molecular tests that report resistance genes. "
            f"Discordant isolates ({discord_rate:.1f}%) may yield misleading results. Always correlate with phenotypic susceptibility and clinical response."
        ),
        'researchers': (
            f"Priority research areas: Investigate silent genes ({top_discord_species.iloc[0]['Species']} has {top_discord_species.iloc[0]['Discordance_Rate_%']:.1f}% discordance), "
            f"evaluate alternative resistance mechanisms, and characterize the {total_discord} discordant isolates at the functional/transcriptomic level."
        ),
        'public': (
            f"Even if bacteria carry resistance genes, they may still respond to {antibiotic_name}. Conversely, bacteria without visible genes may be resistant. "
            f"Laboratory tests must include actual susceptibility testing, not just genetic screening."
        )
    }
    
    return observation, implications


# ============================================================================
# 2. CROSS-SPECIES RESISTANCE TRANSFER SIGNATURES (HGT Networks)
# ============================================================================

def load_gene_sharing_analysis(antibiotic_name):
    # Load gene sharing and HGT data for an antibiotic.
    bhm_path = f"bhm_results/{antibiotic_name}"
    
    try:
        gene_sharing = pd.read_csv(f"{bhm_path}/gene_sharing_summary.csv")
        
        # Load per-gene species probabilities
        gene_data = {}
        for gene in gene_sharing['Gene'].unique():
            try:
                gene_probs = pd.read_csv(f"{bhm_path}/gene_{gene}_species_probabilities.csv")
                gene_data[gene] = gene_probs
            except:
                pass
        
        return {'summary': gene_sharing, 'per_gene': gene_data}
    except FileNotFoundError:
        return None


def generate_hgt_observation(antibiotic_name, gene_data):

    if gene_data is None:
        return f"Gene sharing data for {antibiotic_name} not available.", {}
    
    gene_sharing = gene_data['summary']
    
    # Key HGT Identification: P(multi-species) > 0.80 in 3+ species
    high_priority_hgt = []
    for gene in gene_sharing['Gene'].unique():
        if gene in gene_data['per_gene']:
            gene_probs = gene_data['per_gene'][gene]
            species_with_high_prob = (gene_probs['Median_Probability_Gene_Presence'] > 0.80).sum()
            if species_with_high_prob >= 3:
                high_priority_hgt.append({
                    'Gene': gene,
                    'Species_Count': species_with_high_prob,
                    'Posterior_Prob': gene_sharing[gene_sharing['Gene'] == gene]['Posterior_Probability_Shared_Across_Species'].values[0] if len(gene_sharing[gene_sharing['Gene'] == gene]) > 0 else 0.0
                })
    
    high_priority_hgt = pd.DataFrame(high_priority_hgt).sort_values('Posterior_Prob', ascending=False) if high_priority_hgt else pd.DataFrame()
    
    # Additional inter-species variance assessment (if available)
    has_variance_data = 'Inter_Species_Variance' in gene_sharing.columns
    low_variance_genes = gene_sharing[gene_sharing['Inter_Species_Variance'] < 0.05] if has_variance_data else pd.DataFrame()
    
    # Standard HGT risk categories
    high_sharing = gene_sharing[gene_sharing['Posterior_Probability_Shared_Across_Species'] >= 0.99]
    moderate_sharing = gene_sharing[(gene_sharing['Posterior_Probability_Shared_Across_Species'] >= 0.80) &
                                    (gene_sharing['Posterior_Probability_Shared_Across_Species'] < 0.99)]
    
    observation = (
        f"## Cross-Species Resistance Transfer Signatures for {antibiotic_name}\n\n"
        f"### Bayesian HGT Detection\n"
        f"Horizontal gene transfer (HGT) occurs when resistance genes spread between species via mobile elements (plasmids, transposons). "
        f"**Total genes analyzed**: {len(gene_sharing)}.\n\n"
        
        f"### High-Priority Mobile Elements\n"
        f"**Threshold**: P(multi-species presence) > 0.80 across ≥3 species (strong HGT signature).\n"
    )
    
    if len(high_priority_hgt) > 0:
        observation += f"**Identified {len(high_priority_hgt)} high-priority mobile elements:**\n"
        for idx, row in high_priority_hgt.head(10).iterrows():
            observation += (
                f"- **{row['Gene']}**: Present with high probability (P > 0.80) in {int(row['Species_Count'])} bacterial species. "
                f"Posterior P(shared) = {row['Posterior_Prob']:.4f}. "
                f"**Clinical Action**: Flag mixed infections involving these species—suspect multi-resistant pathogens.\n"
            )
    else:
        observation += "No genes meet the high-priority threshold (P > 0.80 in 3+ species).\n"
    
    observation += (
        f"\n### Very High HGT Risk (P(shared across species) ≥ 0.99)\n"
    )
    
    if len(high_sharing) > 0:
        for idx, row in high_sharing.iterrows():
            species_list = ""
            if row['Gene'] in gene_data['per_gene']:
                gene_probs = gene_data['per_gene'][row['Gene']]
                top_carriers = gene_probs.nlargest(3, 'Median_Probability_Gene_Presence')
                species_list = ", ".join([s for s in top_carriers['Species'].values])
            
            observation += (
                f"- **{row['Gene']}**: {row['Posterior_Probability_Shared_Across_Species']:.4f} probability of presence in ≥2 species "
                f"({int(row['Median_Number_of_Species_With_Gene'])} species median; e.g., {species_list}). "
                f"**Interpretation**: Near-certainty of horizontal transfer; this gene is circulating broadly across the Nigerian microbiome.\n"
            )
    else:
        observation += "None at this threshold.\n"
    
    observation += f"\n### Moderate HGT Risk (P(shared) 0.80–0.99)\n"
    
    if len(moderate_sharing) > 0:
        for idx, row in moderate_sharing.head(5).iterrows():
            observation += (
                f"- **{row['Gene']}**: {row['Posterior_Probability_Shared_Across_Species']:.4f} probability; median {int(row['Median_Number_of_Species_With_Gene'])} species. "
                f"Strong evidence of inter-species transmission; monitor for further spread.\n"
            )
    else:
        observation += "None at this threshold.\n"
    
    # Most widely distributed gene
    most_distributed = None
    if len(gene_sharing) > 0:
        most_distributed = gene_sharing.nlargest(1, 'Median_Number_of_Species_With_Gene').iloc[0]
        observation += (
            f"\n### Most Widely Distributed Gene\n"
            f"**{most_distributed['Gene']}** is found in {int(most_distributed['Median_Number_of_Species_With_Gene'])} bacterial species, "
            f"indicating an established, multi-species resistance network for {antibiotic_name} in Nigeria. "
            f"This suggests either historical spread or ongoing transmission favorable to resistance dissemination.\n"
        )
    
    if has_variance_data and len(low_variance_genes) > 0:
        observation += (
            f"\n### Species-Agnostic Genes (Inter-Species Variance ≈ 0)\n"
            f"Genes with near-zero variance across species are NOT adapting to species-specific niches; instead, they persist uniformly. "
            f"**This is a strong HGT signature.** Identified {len(low_variance_genes)} such genes:\n"
        )
        for idx, row in low_variance_genes.head(5).iterrows():
            observation += f"- **{row['Gene']}**: σ²_species = {row['Inter_Species_Variance']:.4f}\n"
    
    
    # Extract safe values for implications
    top_gene_name = most_distributed['Gene'] if most_distributed is not None else 'identified gene'
    top_gene_species_count = int(most_distributed['Median_Number_of_Species_With_Gene']) if most_distributed is not None else 0
    
    implications = {
        'policymakers': (
            f"The documented HGT networks (with {len(high_sharing)} very-high-risk genes) indicate that resistance to {antibiotic_name} "
            f"is not confined to individual species but represents a pan-population threat. "
            f"Implement coordinated stewardship across all sentinel laboratories to track {top_gene_name} prevalence."
        ),
        'clinicians': (
            f"When {top_gene_name} is identified, consider that multiple bacterial species in a patient may carry resistance. "
            f"For mixed infections, suspect multi-drug-resistant organisms due to HGT. Consider broad-spectrum empirical coverage pending cultures."
        ),
        'researchers': (
            f"Priority investigations: Map the transmission chains for {top_gene_name} across {top_gene_species_count} species, "
            f"identify replicon types facilitating transfer, sequence surveillance of conjugative plasmids, "
            f"and conduct experimental transfer studies to validate inferred HGT networks."
        ),
        'public': (
            f"Resistance genes are being shared between different bacteria species in Nigeria, spreading resistance more rapidly. "
            f"Proper sewage treatment, handwashing, and infection control are critical to slow this spread ecosystem-wide."
        )
    }
    
    return observation, implications


# ============================================================================
# 3. TEMPORAL EVOLUTION OF ANTIBIOTIC RESISTANCE
# ============================================================================

def load_temporal_analysis(antibiotic_name):
    # Load temporal trend data for an antibiotic.
    bhm_path = f"bhm_results/{antibiotic_name}"
    
    try:
        species_slopes = pd.read_csv(f"{bhm_path}/{antibiotic_name}_species_slopes.csv")
        model_summary = pd.read_csv(f"{bhm_path}/{antibiotic_name}_temporal_model_summary.csv", index_col=0)
        
        return {
            'species_slopes': species_slopes,
            'model_summary': model_summary
        }
    except FileNotFoundError:
        return None


def generate_temporal_observation(antibiotic_name, temporal_data):
    
    if temporal_data is None:
        return f"Temporal data for {antibiotic_name} not available.", {}
    
    species_slopes = temporal_data['species_slopes']
    model_summary = temporal_data['model_summary']
    
    # Bayesian Health Checks
    has_convergence_issue = (model_summary['r_hat'] > 1.01).sum() > 0 if 'r_hat' in model_summary.columns else False
    
    # National trend (posterior mean and credible interval)
    national_slope = model_summary.loc['national_slope', 'mean'] if 'national_slope' in model_summary.index else 0.0
    national_slope_hdi_lower = model_summary.loc['national_slope', 'hdi_3%'] if 'national_slope' in model_summary.index else 0.0
    national_slope_hdi_upper = model_summary.loc['national_slope', 'hdi_97%'] if 'national_slope' in model_summary.index else 0.0
    
    # HDI Width Assessment
    hdi_width = abs(national_slope_hdi_upper - national_slope_hdi_lower)
    has_narrow_hdi = hdi_width < 0.10  # Rough threshold for "narrow"
    
    # Calculate P(β_year > 0) from species data as proxy for national significance
    # If most species accelerating, national trend is likely significant
    nat_prob_increase = (species_slopes['Probability_Slope_Positive'] > 0.5).sum() / len(species_slopes)
    
    # Identify accelerating and decelerating species (threshold: P > 0.95 or < 0.05)
    accelerating = species_slopes[species_slopes['Probability_Slope_Positive'] >= 0.95]
    decelerating = species_slopes[species_slopes['Probability_Slope_Positive'] <= 0.05]
    uncertain = species_slopes[(species_slopes['Probability_Slope_Positive'] > 0.05) &
                                (species_slopes['Probability_Slope_Positive'] < 0.95)]
    
    # Species outlier analysis: Is national trend distributed or driven by one species?
    if len(accelerating) > 0:
        outlier_species = accelerating.nlargest(1, 'Median_Slope_Resistance_Change_per_Year').iloc[0]
        outlier_contribution = (outlier_species['Median_Slope_Resistance_Change_per_Year'] / 
                                 (accelerating['Median_Slope_Resistance_Change_per_Year'].sum() / len(accelerating)))
    else:
        outlier_species = None
        outlier_contribution = 0
    
    observation = (
        f"## Temporal Evolution of {antibiotic_name} Resistance in Nigeria\n\n"
        f"### Bayesian Health Checks\n"
        f"**Model Convergence**: {'⚠️ WARNING - R-hat > 1.01 detected. Results may be unreliable.' if has_convergence_issue else '✓ All parameters converged (R-hat < 1.01).'}\n"
        f"**Uncertainty Assessment (95% HDI)**: HDI width = {hdi_width:.4f}. "
        f"{'Narrow interval—robust evidence.' if has_narrow_hdi else 'Wide interval—insufficient longitudinal data or heterogeneous species trends.'}\n\n"
        
        f"### National Resistance Trend\n"
        f"**Posterior Slope**: {national_slope:.4f} log-odds/year [95% HDI: {national_slope_hdi_lower:.4f} to {national_slope_hdi_upper:.4f}].\n"
    )
    
    if national_slope_hdi_lower > 0:
        observation += (
            f"**Interpretation**: ✓ **Statistically Significant Increase** (P(β > 0) ~ 0.95+). "
            f"{antibiotic_name} resistance is accelerating nationally. This is an urgent public health signal.\n"
        )
    elif national_slope_hdi_upper < 0:
        observation += (
            f"**Interpretation**: ✓ **Statistically Significant Decrease** (P(β < 0) ~ 0.95+). "
            f"{antibiotic_name} resistance is declining nationally. Verify this reflects true control, not surveillance artifacts.\n"
        )
    else:
        observation += (
            f"**Interpretation**: ⚠️ **Uncertain National Trend** (Credible interval spans zero). "
            f"Data are insufficient to declare definitive increase or decrease. Extend surveillance or increase resolution.\n"
        )
    
    observation += (
        f"\n### Species-Level Trajectories (Granular Analysis)\n\n"
        f"**Critical Threshold: P(slope > 0) > 0.95 = Statistically Significant Acceleration**\n\n"
        f"**High-Confidence Accelerating Species** ({len(accelerating)} species):\n"
    )
    
    if len(accelerating) > 0:
        for idx, row in accelerating.nlargest(5, 'Median_Slope_Resistance_Change_per_Year').iterrows():
            observation += (
                f"- **{row['Species']}**: {row['Median_Slope_Resistance_Change_per_Year']:.4f} log-odds/year, P(slope>0) = {row['Probability_Slope_Positive']:.3f}. "
                f"Steep trajectory—immediate targeted intervention (culture surveillance, screening, infection control).\n"
            )
        
        if outlier_species is not None and outlier_contribution > 1.5:
            observation += (
                f"\n**⚠️ OUTLIER ALERT**: {outlier_species['Species']} is disproportionately driving the national increase "
                f"({outlier_contribution:.1f}x relative contribution). "
                f"**Strategic Question**: Is this species epidemic locally? Was there a lab changeover? "
                f"If localized, national intervention may not be needed; if ecosystem-wide, urgent action required.\n"
            )
    else:
        observation += "- None. All species show stable or declining resistance.\n"
    
    observation += (
        f"\n**High-Confidence Decelerating Species** ({len(decelerating)} species):\n"
    )
    
    if len(decelerating) > 0:
        for idx, row in decelerating.nsmallest(3, 'Median_Slope_Resistance_Change_per_Year').iterrows():
            observation += (
                f"- **{row['Species']}**: {row['Median_Slope_Resistance_Change_per_Year']:.4f} log-odds/year, P(slope>0) = {row['Probability_Slope_Positive']:.3f}. "
                f"Resistance declining—investigate successful interventions (e.g., stewardship initiatives, reduced antibiotic use, improved infection control).\n"
            )
    else:
        observation += "- None. All species show stable or rising resistance.\n"
    
    observation += (
        f"\n**Uncertain Trends** ({len(uncertain)} species, 0.05 < P < 0.95):\n"
        f"- {len(uncertain)} species with ambiguous trajectory. Recommend extended surveillance or investigate episodic outbreaks, lab protocol changes.\n"
    )
    
    implications = {
        'policymakers': (
            f"{'**ALERT**:' if national_slope_hdi_lower > 0 else '**MONITORING**:'} National resistance to {antibiotic_name} is "
            f"{'increasing.' if national_slope > 0 else 'stable or declining.'} "
            f"Prioritize surveillance and intervention for {accelerating.iloc[0]['Species'] if len(accelerating) > 0 else 'vulnerable species'}. "
            f"If {len(accelerating)} species show significant acceleration, coordinate multi-facility stewardship. "
            f"Consider antimicrobial use restrictions in high-resistance settings."
        ),
        'clinicians': (
            f"Temporal data show {'escalating' if national_slope > 0 else 'improving'} resistance to {antibiotic_name}. "
            f"Update empirical prescribing guidelines, especially for {len(accelerating)} rapidly evolving species. "
            f"Consider combination therapy or alternatives for high-risk pathogens. "
            f"{'Verify any resistance declines before relaxedprescribing.' if len(decelerating) > 0 else ''} "
        ),
        'researchers': (
            f"Species trajectories reveal heterogeneous resistance evolution for {antibiotic_name}. "
            f"Investigate mechanisms: {accelerating.iloc[0]['Species'] if len(accelerating) > 0 else 'organisms'} (clonal expansion, plasmid fitness, mutational pathways). "
            f"{'For declining species: characterize successful interventions.' if len(decelerating) > 0 else ''} "
            f"{'Species outlier detected—validate via independent sampling.' if outlier_species is not None and outlier_contribution > 1.5 else ''}"
        ),
        'public': (
            f"Resistance to {antibiotic_name} is {'on the rise' if national_slope > 0 else 'stabilizing'} in Nigeria. "
            f"Take antibiotics only as prescribed, complete the full course, and support infection prevention measures (handwashing, sanitation). "
            f"Support research into new antibiotics and alternatives."
        )
    }
    
    return observation, implications


# ============================================================================
# UTILITY: Get all available antibiotics in bhm_results
# ============================================================================

def get_available_antibiotics():
    # Return list of antibiotics with BHM results available.
    bhm_dir = "bhm_results"
    if not os.path.exists(bhm_dir):
        return []
    
    antibiotics = [d for d in os.listdir(bhm_dir) 
                   if os.path.isdir(os.path.join(bhm_dir, d)) and not d.startswith('.')]
    return sorted(antibiotics)
