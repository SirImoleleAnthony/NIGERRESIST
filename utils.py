import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress, chi2_contingency
import warnings
import streamlit as st
warnings.filterwarnings('ignore')

@st.cache_data
def load_data(filepath="data/amr_nigeria_full_data.csv"):

    df_wide = pd.read_csv(filepath)
    
    # Identify all columns that end with '_I' (interpretation columns)
    interp_cols = [col for col in df_wide.columns if col.endswith('_I')]
    # The corresponding MIC columns are the same names without '_I'
    mic_cols = [col.replace('_I', '') for col in interp_cols]
    
    # Keep all other columns (non-antibiotic specific)
    id_vars = [col for col in df_wide.columns if col not in interp_cols and col not in mic_cols]
    
    # Melt the interpretation columns
    df_interp = pd.melt(
        df_wide,
        id_vars=id_vars,
        value_vars=interp_cols,
        var_name='Antibiotic_Interpretation',
        value_name='Interpretation'
    )
    # Extract antibiotic name from column name (e.g., 'Amikacin_I' -> 'Amikacin')
    df_interp['Antibiotic'] = df_interp['Antibiotic_Interpretation'].str.replace('_I', '')
    
    # Melt the MIC columns
    df_mic = pd.melt(
        df_wide,
        id_vars=['Isolate Id'],  # only need a key to merge later
        value_vars=mic_cols,
        var_name='Antibiotic',
        value_name='MIC'
    )
    
    # Merge the two melted dataframes on 'Isolate Id' and 'Antibiotic'
    df_long = pd.merge(df_interp, df_mic, on=['Isolate Id', 'Antibiotic'], how='left')
    
    # Convert MIC to numeric, handling ">" or "<" values
    # For simplicity, we replace ">" with +0.5 (or remove and use as float)
    def clean_mic(val):
        if pd.isna(val):
            return np.nan
        if isinstance(val, str):
            # Handle ">64", ">32", etc.
            if '>' in val:
                try:
                    return float(val.replace('>', '')) + 0.5
                except:
                    return np.nan
            # Handle "<0.5" etc. (maybe treat as half the value)
            if '<' in val:
                try:
                    return float(val.replace('<', '')) / 2
                except:
                    return np.nan
            # Otherwise convert to float
            try:
                return float(val)
            except:
                return np.nan
        return val
    
    df_long['MIC'] = df_long['MIC'].apply(clean_mic)
    
    # Ensure proper data types
    df_long['Year'] = df_long['Year'].astype(int)
    # Optional: keep 'Interpretation' as string
    # We'll also drop rows where Interpretation is missing (though we keep MIC missing allowed)
    df_long = df_long.dropna(subset=['Interpretation'])
    
    # Optionally, keep only the columns needed for analysis (to save memory)
    keep_cols = [
        'Isolate Id', 'Study', 'Species', 'Family', 'Country', 'Gender',
        'Age Group', 'Speciality', 'Source', 'Year', 'Phenotype', 'Genes',
        'Genotype', 'Antibiotic', 'MIC', 'Interpretation'
    ]
    df_long = df_long[keep_cols]
    
    return df_long

# Calculate resistance rate for a given DataFrame
def calculate_resistance_rate(df):
    total = len(df)
    resistant = (df['Interpretation'] == 'Resistant').sum()
    return (resistant / total * 100) if total > 0 else 0, total

# Get resistance rates by category 
def get_category_rates(df, category_col):
    rates = df.groupby(category_col).agg(
        resistant=('Interpretation', lambda x: (x == 'Resistant').sum()),
        total=('Interpretation', 'count')
    ).reset_index()
    rates['rate'] = (rates['resistant'] / rates['total'] * 100).round(1)
    rates = rates.sort_values('rate', ascending=False)
    return rates

# Trends description 
def describe_trend(yearly_df, antibiotic_name):
    if len(yearly_df) < 3:
        return f"Insufficient data to determine a reliable trend for {antibiotic_name} (only {len(yearly_df)} years)."
    slope, intercept, r_value, p_value, std_err = linregress(yearly_df['Year'], yearly_df['percent'])
    direction = "increasing" if slope > 0 else "decreasing"
    sig = "statistically significant" if p_value < 0.05 else "not statistically significant"
    change = yearly_df['percent'].iloc[-1] - yearly_df['percent'].iloc[0]
    first_year = int(yearly_df['Year'].iloc[0])
    last_year = int(yearly_df['Year'].iloc[-1])
    desc = (f"Over the period {first_year}–{last_year}, resistance to {antibiotic_name} "
            f"shows a {direction} trend of {abs(slope):.2f}% per year ({sig}, p={p_value:.3f}). "
            f"Overall, resistance { 'increased' if change > 0 else 'decreased' } from {yearly_df['percent'].iloc[0]:.1f}% "
            f"to {yearly_df['percent'].iloc[-1]:.1f}%, a {abs(change):.1f} percentage point change.")
    return desc

# Describe categorical rates
def describe_categorical_rates(rates_df, category_name, metric='resistance'):
    if rates_df.empty:
        return f"No data available for {category_name}."
    highest = rates_df.iloc[0]
    lowest = rates_df.iloc[-1]
    desc = (f"Resistance rates vary considerably across {category_name}. "
            f"The highest rate is observed in **{highest[category_name]}** at **{highest['rate']:.1f}%** "
            f"({highest['resistant']} resistant out of {highest['total']} isolates). "
            f"The lowest rate is in **{lowest[category_name]}** at **{lowest['rate']:.1f}%** "
            f"({lowest['resistant']} resistant out of {lowest['total']} isolates).")
    # Add more details for categories with large differences
    if len(rates_df) > 2:
        median_rate = rates_df['rate'].median()
        desc += f" The median resistance rate across categories is {median_rate:.1f}%."
    return desc

# Trend Over Time
def plot_trend(df, antibiotic_name, species=None):
    mask = pd.Series([True] * len(df), index=df.index)
    if species:
        mask &= df['Species'] == species
    df_filtered = df[mask]

    if df_filtered.empty:
        return None, f"No data available for {antibiotic_name}" + (f" in species {species}" if species else "")

    yearly = df_filtered.groupby('Year')['Interpretation'].agg(
        resistant=lambda x: (x == 'Resistant').sum(),
        total='count'
    ).reset_index()
    yearly['percent'] = (yearly['resistant'] / yearly['total']) * 100

    # Create plot
    fig = px.line(yearly, x='Year', y='percent',
                  title=f'Resistance to {antibiotic_name} Over Time' + (f' ({species})' if species else ''),
                  labels={'percent': 'Resistance (%)'})
    fig.add_scatter(x=yearly['Year'], y=yearly['percent'], mode='markers', marker=dict(size=8), showlegend=False)
    fig.update_layout(yaxis_range=[0,100])

    # Generate detailed observation
    obs = describe_trend(yearly, antibiotic_name)

    # Add contextual information about the data
    obs += f" The analysis is based on {len(df_filtered)} isolates tested for {antibiotic_name}."
    if species:
        obs += f" For {species}, the sample size ranges from {yearly['total'].min()} to {yearly['total'].max()} isolates per year."

    # Calculate slope (trend) from the yearly data
    if len(yearly) > 1:
        slope_result = linregress(yearly['Year'], yearly['percent'])
        slope = slope_result.slope
        direction = 'upward' if slope > 0 else 'downward'
    else:
        slope = 0
        direction = 'stable'

    # Implications (as a separate string or dictionary)
    implications = {
        'policymakers': f"Based on the observed trend, policymakers should {'intensify' if slope>0 else 'maintain'} surveillance and consider {'tightening' if slope>0 else 'continuing'} stewardship programs for {antibiotic_name}.",
        'clinicians': f"Clinicians should {'be cautious' if slope>0 else 'stay vigilant'} with {antibiotic_name} prescribing, as resistance is {'rising' if slope>0 else 'declining'} at a rate of {abs(slope):.2f}% per year.",
        'researchers': f"The {direction} trend provides an opportunity to investigate the drivers behind the change, such as antibiotic usage patterns or infection control measures.",
        'public': f"Patients should always complete prescribed antibiotics and never share them, as this helps slow the {'increase' if slope>0 else 'decline'} in resistance."
    }

    return fig, obs, implications


# Resistance by Species
def plot_by_species(df, antibiotic_name, top_n=10):
    """Bar chart of resistance rates by species."""
    mask = df['Antibiotic'] == antibiotic_name
    df_filtered = df[mask]
    if df_filtered.empty:
        return None, f"No data for {antibiotic_name}.", {}

    rates = get_category_rates(df_filtered, 'Species')
    rates = rates.head(top_n)

    fig = px.bar(rates, x='Species', y='rate',
                 title=f'Resistance to {antibiotic_name} by Species (top {top_n})',
                 labels={'rate': 'Resistance (%)', 'Species': ''})
    fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0,100])

    # Detailed observation
    obs = describe_categorical_rates(rates, 'Species')
    # Add species with highest total isolates
    top_species_total = rates.nlargest(1, 'total').iloc[0]
    obs += f" The most frequently tested species was **{top_species_total['Species']}** with {top_species_total['total']} isolates."

    # Implications
    implications = {
        'policymakers': f"Focus stewardship programs on the species with highest resistance rates, particularly **{rates.iloc[0]['Species']}** ({rates.iloc[0]['rate']:.1f}% resistant).",
        'clinicians': f"When treating infections caused by **{rates.iloc[0]['Species']}**, consider alternative antibiotics or combination therapy due to high resistance.",
        'researchers': f"Investigate the mechanisms driving high resistance in **{rates.iloc[0]['Species']}** and whether these are transferable to other species.",
        'public': f"Understanding which bacteria are most resistant helps healthcare providers choose the right antibiotics."
    }
    return fig, obs, implications


# Resistance by Age Group
def plot_by_age(df, antibiotic_name):
    """Bar chart of resistance by age group."""
    mask = df['Antibiotic'] == antibiotic_name
    df_filtered = df[mask]
    if df_filtered.empty:
        return None, f"No data for {antibiotic_name}.", {}

    #age_order = ['0 to 2 Years', '3 to 12 Years', '13 to 18 Years', '19 to 64 Years', '65+ Years', 'Unknown']
    age_order = ['0 to 2 Years', '3 to 12 Years', '13 to 18 Years', '19 to 64 Years', '65 to 84 Years', '85 and Over', 'Unknown']
    rates = get_category_rates(df_filtered, 'Age Group')
    
    # Filter out NaN values
    rates = rates[rates['Age Group'].notna()]
    
    # Get description before reordering (while sorted by rate)
    obs = describe_categorical_rates(rates, 'Age Group')
    
    # Reorder for display purposes only
    rates['Age Group'] = pd.Categorical(rates['Age Group'], categories=age_order, ordered=True)
    rates_display = rates.sort_values('Age Group')

    fig = px.bar(rates_display, x='Age Group', y='rate',
                 title=f'Resistance to {antibiotic_name} by Age Group',
                 labels={'rate': 'Resistance (%)'})
    fig.update_layout(yaxis_range=[0,100])

    # Add explanation if children vs adults
    child_groups = ['0 to 2 Years', '3 to 12 Years']
    adult_groups = ['19 to 64 Years', '65 to 84 Years', '85 and Over']
    # adults= ['19 to 64 Years', '65+ Years']
    child_rate = rates_display[rates_display['Age Group'].isin(child_groups)]['rate'].mean()
    adult_rate = rates_display[rates_display['Age Group'].isin(adult_groups)]['rate'].mean()
    obs += f" On average, children under 13 have a resistance rate of {child_rate:.1f}%, while adults have {adult_rate:.1f}%."

    # Use rates (sorted by resistance) to get highest group for implications
    highest_group = rates.iloc[0]['Age Group']
    implications = {
        'policymakers': f"Age‑specific guidelines may be needed, especially for the most affected age group: **{highest_group}**.",
        'clinicians': f"Consider age when selecting empiric therapy; {highest_group} patients show the highest resistance.",
        'researchers': f"Explore age‑related factors (immune maturity, comorbidities, prior antibiotic exposure) that contribute to resistance differences.",
        'public': f"Age can influence antibiotic effectiveness; always inform your healthcare provider about the patient's age."
    }
    return fig, obs, implications

# Resistance by Gender
def plot_by_gender(df, antibiotic_name):
    """Bar chart of resistance by gender."""
    mask = df['Antibiotic'] == antibiotic_name
    df_filtered = df[mask]
    if df_filtered.empty:
        return None, f"No data for {antibiotic_name}.", {}

    rates = get_category_rates(df_filtered, 'Gender')
    fig = px.bar(rates, x='Gender', y='rate',
                 title=f'Resistance to {antibiotic_name} by Gender',
                 labels={'rate': 'Resistance (%)'})
    fig.update_layout(yaxis_range=[0,100])

    # Detailed observation
    if len(rates) == 2:
        male_rate = rates[rates['Gender']=='Male']['rate'].values[0]
        female_rate = rates[rates['Gender']=='Female']['rate'].values[0]
        diff = male_rate - female_rate
        if diff > 0:
            obs = f"Males show a {diff:.1f} percentage point higher resistance rate ({male_rate:.1f}%) compared to females ({female_rate:.1f}%)."
        elif diff < 0:
            obs = f"Females show a {abs(diff):.1f} percentage point higher resistance rate ({female_rate:.1f}%) compared to males ({male_rate:.1f}%)."
        else:
            obs = f"Resistance rates are similar between males ({male_rate:.1f}%) and females ({female_rate:.1f}%)."
        obs += f" The analysis includes {rates[rates['Gender']=='Male']['total'].values[0]} male isolates and {rates[rates['Gender']=='Female']['total'].values[0]} female isolates."
    else:
        obs = describe_categorical_rates(rates, 'Gender')

    implications = {
        'policymakers': f"Consider gender‑specific awareness campaigns if disparities are significant.",
        'clinicians': f"Be aware of potential gender differences; however, empiric therapy should not be solely based on gender.",
        'researchers': f"Investigate biological (e.g., hormonal) or behavioral factors that could explain the observed differences.",
        'public': f"Both men and women can be affected by resistant infections; seek medical advice promptly."
    }
    return fig, obs, implications


# Resistance by Specialty
def plot_by_specialty(df, antibiotic_name, top_n=10):
    """Bar chart of resistance by clinical specialty."""
    mask = df['Antibiotic'] == antibiotic_name
    df_filtered = df[mask]
    if df_filtered.empty:
        return None, f"No data for {antibiotic_name}.", {}

    rates = get_category_rates(df_filtered, 'Speciality')
    rates = rates.head(top_n)

    fig = px.bar(rates, x='Speciality', y='rate',
                 title=f'Resistance to {antibiotic_name} by Speciality (top {top_n})',
                 labels={'rate': 'Resistance (%)'})
    fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0,100])

    obs = describe_categorical_rates(rates, 'Speciality')
    # Add context about high-risk specialties
    high_risk = rates.iloc[0]['Speciality']
    obs += f" Specialties like **{high_risk}** may require intensified infection control measures."

    implications = {
        'policymakers': f"Target stewardship interventions to high‑resistance specialties such as **{high_risk}**.",
        'clinicians': f"Be extra cautious when prescribing in specialties with high resistance; consider consulting infectious disease specialists.",
        'researchers': f"Study the reasons for high resistance in specific specialties (e.g., device use, patient acuity, antibiotic usage).",
        'public': f"Resistance rates can vary by hospital ward; trust your healthcare team to choose the right antibiotic."
    }
    return fig, obs, implications

# Resistance by Source
def plot_by_source(df, antibiotic_name, top_n=10):
    """Bar chart of resistance by infection source."""
    mask = df['Antibiotic'] == antibiotic_name
    df_filtered = df[mask]
    if df_filtered.empty:
        return None, f"No data for {antibiotic_name}.", {}

    rates = get_category_rates(df_filtered, 'Source')
    rates = rates.head(top_n)

    fig = px.bar(rates, x='Source', y='rate',
                 title=f'Resistance to {antibiotic_name} by Source (top {top_n})',
                 labels={'rate': 'Resistance (%)'})
    fig.update_layout(xaxis_tickangle=-45, yaxis_range=[0,100])

    obs = describe_categorical_rates(rates, 'Source')
    # Add explanation about possible reasons
    top_source = rates.iloc[0]['Source']
    obs += f" The highest resistance is observed in **{top_source}** infections, possibly due to biofilm formation or higher antibiotic pressure in those sites."

    implications = {
        'policymakers': f"Allocate resources for infection prevention in sources with highest resistance (e.g., **{top_source}**).",
        'clinicians': f"Consider source‑specific empirical therapy; for **{top_source}** infections, alternative antibiotics may be needed.",
        'researchers': f"Investigate the microbial ecology and resistance mechanisms in high‑resistance sources.",
        'public': f"Different types of infections may require different antibiotics; always follow your doctor's advice."
    }
    return fig, obs, implications

# Genotype–Phenotype Correlation (by Gene)
def plot_by_gene(df, antibiotic_name, gene):
    """Bar chart comparing resistance rates with and without a specific gene."""
    mask = (df['Antibiotic'] == antibiotic_name) & (df['Genes'].notna())
    df_filtered = df[mask].copy()
    if df_filtered.empty:
        return None, f"No data for {antibiotic_name} with gene information.", {}

    df_filtered['gene_present'] = df_filtered['Genes'].str.contains(gene, na=False)
    rates = get_category_rates(df_filtered, 'gene_present')
    rates['gene_present'] = rates['gene_present'].map({True: f'With {gene}', False: f'Without {gene}'})

    fig = px.bar(rates, x='gene_present', y='rate',
                 title=f'Resistance to {antibiotic_name} by {gene} Presence',
                 labels={'rate': 'Resistance (%)', 'gene_present': ''})
    fig.update_layout(yaxis_range=[0,100])

    if len(rates) == 2:
        with_gene = rates[rates['gene_present']==f'With {gene}']['rate'].values[0]
        without_gene = rates[rates['gene_present']==f'Without {gene}']['rate'].values[0]
        diff = with_gene - without_gene
        obs = f"Isolates carrying the **{gene}** gene have a {diff:.1f} percentage point higher resistance rate ({with_gene:.1f}%) compared to those without the gene ({without_gene:.1f}%). "
        obs += f"This suggests that **{gene}** is a strong predictor of resistance to {antibiotic_name} in this dataset."
    else:
        obs = f"Insufficient data to compare the effect of {gene} on resistance to {antibiotic_name}."

    implications = {
        'policymakers': f"Molecular surveillance targeting **{gene}** can help predict resistance trends.",
        'clinicians': f"If rapid testing detects **{gene}**, consider avoiding {antibiotic_name} in favor of alternative agents.",
        'researchers': f"Investigate the mechanisms by which **{gene}** confers resistance and its potential for horizontal transfer.",
        'public': f"Laboratory tests that identify resistance genes help doctors select the most effective antibiotic."
    }
    return fig, obs, implications

# MIC Distribution
def plot_mic_distribution(df, antibiotic_name, species=None):
    """Box plot of MIC values by susceptibility."""
    mask = (df['Antibiotic'] == antibiotic_name) & (df['MIC'].notna())
    if species:
        mask &= df['Species'] == species
    df_filtered = df[mask].copy()

    if df_filtered.empty:
        return None, f"No MIC data for {antibiotic_name}" + (f" in {species}" if species else ""), {}

    # Convert MIC to numeric (handling ">" etc. – we assume preprocessing)
    df_filtered['MIC'] = pd.to_numeric(df_filtered['MIC'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['MIC'])

    fig = px.box(df_filtered, x='Interpretation', y='MIC',
                 title=f'MIC Distribution for {antibiotic_name}' + (f' ({species})' if species else ''),
                 labels={'MIC': 'MIC (µg/mL)', 'Interpretation': 'Susceptibility'})
    fig.update_yaxes(type="log")  # MIC often log scale

    # Calculate median MIC values
    resistant_mic = df_filtered[df_filtered['Interpretation']=='Resistant']['MIC']
    susceptible_mic = df_filtered[df_filtered['Interpretation']=='Susceptible']['MIC']
    if len(resistant_mic) > 0 and len(susceptible_mic) > 0:
        med_res = resistant_mic.median()
        med_sus = susceptible_mic.median()
        obs = f"Resistant isolates have a median MIC of {med_res:.1f} µg/mL, which is {med_res/med_sus:.1f} times higher than the median for susceptible isolates ({med_sus:.1f} µg/mL). "
        obs += f"This large separation indicates that the current breakpoint effectively distinguishes resistant from susceptible strains."
    else:
        obs = "Insufficient data to compare MIC distributions."

    implications = {
        'policymakers': f"Breakpoints should be periodically reviewed to ensure they reflect clinical outcomes.",
        'clinicians': f"For isolates with MICs close to the breakpoint, consider higher doses or combination therapy.",
        'researchers': f"MIC distributions can help identify emerging resistance trends before they become clinically significant.",
        'public': f"MIC testing helps determine the exact concentration needed to kill bacteria, guiding precise treatment."
    }
    return fig, obs, implications

# Distribution of Bacterial Species by Gender
def plot_species_by_gender(df, antibiotic_name, top_n=10):
    """Bar chart of species counts by gender."""
    mask = df['Antibiotic'] == antibiotic_name
    df_filtered = df[mask]
    if df_filtered.empty:
        return None, f"No data for {antibiotic_name}.", {}

    # Top species overall
    top_species = df_filtered['Species'].value_counts().head(top_n).index
    df_top = df_filtered[df_filtered['Species'].isin(top_species)]
    counts = df_top.groupby(['Species', 'Gender']).size().reset_index(name='Count')

    fig = px.bar(counts, x='Species', y='Count', color='Gender',
                 title=f'Species Distribution by Gender ({antibiotic_name})',
                 barmode='group')
    fig.update_layout(xaxis_tickangle=-45)

    # Observation: most common species per gender
    male_top = df_top[df_top['Gender']=='Male']['Species'].value_counts().head(1)
    female_top = df_top[df_top['Gender']=='Female']['Species'].value_counts().head(1)
    if not male_top.empty and not female_top.empty:
        obs = f"The most common species in males is **{male_top.index[0]}** with {male_top.values[0]} isolates. "
        obs += f"In females, **{female_top.index[0]}** is most common with {female_top.values[0]} isolates. "
        obs += f"This suggests that certain species may have a predilection for one gender, possibly due to anatomical or behavioral factors."
    else:
        obs = f"Species distribution varies by gender. The bar chart shows the counts for the top {top_n} species."

    implications = {
        'policymakers': f"Gender‑specific infection prevention strategies may be needed if certain species disproportionately affect one gender.",
        'clinicians': f"Consider gender when assessing the likelihood of specific pathogens.",
        'researchers': f"Investigate why certain species are more common in one gender (e.g., urinary tract infections in females).",
        'public': f"Men and women may be prone to different infections; awareness can help in early recognition."
    }
    return fig, obs, implications

# Distribution of Bacterial Species by Age Group
def plot_species_by_age(df, antibiotic_name, top_n=10):
    """Bar chart of species counts by age group."""
    mask = df['Antibiotic'] == antibiotic_name
    df_filtered = df[mask]
    if df_filtered.empty:
        return None, f"No data for {antibiotic_name}.", {}

    top_species = df_filtered['Species'].value_counts().head(top_n).index
    df_top = df_filtered[df_filtered['Species'].isin(top_species)]
    age_order = ['0 to 2 Years', '3 to 12 Years', '13 to 18 Years', '19 to 64 Years', '65 to 84 Years', '85 and Over', 'Unknown']
    counts = df_top.groupby(['Species', 'Age Group']).size().reset_index(name='Count')
    counts['Age Group'] = pd.Categorical(counts['Age Group'], categories=age_order, ordered=True)
    counts = counts.sort_values('Age Group')

    fig = px.bar(counts, x='Species', y='Count', color='Age Group',
                 title=f'Species Distribution by Age Group ({antibiotic_name})',
                 barmode='group')
    fig.update_layout(xaxis_tickangle=-45)

    # Observation: children vs adults
    children = ['0 to 2 Years', '3 to 12 Years']
    adults = ['19 to 64 Years', '65 to 84 Years', '85 and Over']
    child_counts = df_top[df_top['Age Group'].isin(children)]['Species'].value_counts()
    adult_counts = df_top[df_top['Age Group'].isin(adults)]['Species'].value_counts()
    obs = f"In children (≤12 years), the most common species is **{child_counts.index[0]}** ({child_counts.values[0]} isolates). "
    obs += f"In adults (19–64 years), **{adult_counts.index[0]}** is most common ({adult_counts.values[0]} isolates). "
    obs += f"This age‑related shift may reflect differences in immune maturity or exposure patterns."

    implications = {
        'policymakers': f"Pediatric and adult guidelines should reflect the different pathogen profiles.",
        'clinicians': f"Empiric therapy may need to be age‑specific based on the most likely pathogens.",
        'researchers': f"Study the age‑related factors influencing pathogen acquisition.",
        'public': f"Children and adults may get different types of bacterial infections; healthcare providers consider this when treating."
    }
    return fig, obs, implications

# Species Distribution Over Years
def plot_species_over_years(df, antibiotic_name, top_n=5):
    """Line chart of species counts over years."""
    mask = df['Antibiotic'] == antibiotic_name
    df_filtered = df[mask]
    if df_filtered.empty:
        return None, f"No data for {antibiotic_name}.", {}

    top_species = df_filtered['Species'].value_counts().head(top_n).index
    df_top = df_filtered[df_filtered['Species'].isin(top_species)]
    yearly = df_top.groupby(['Year', 'Species']).size().reset_index(name='Count')

    fig = px.line(yearly, x='Year', y='Count', color='Species',
                  title=f'Species Distribution Over Years ({antibiotic_name})',
                  markers=True)

    # Observation: species trends
    trends = []
    for species in top_species:
        sp_data = yearly[yearly['Species'] == species]
        if len(sp_data) >= 3:
            slope, _, _, p, _ = linregress(sp_data['Year'], sp_data['Count'])
            if p < 0.05:
                direction = "increasing" if slope > 0 else "decreasing"
                trends.append(f"{species} is significantly {direction} (p={p:.3f})")
            else:
                trends.append(f"{species} shows no significant trend (p={p:.3f})")
    if trends:
        obs = "Trend analysis over the study period: " + "; ".join(trends)
    else:
        obs = "Insufficient years to determine trends for the top species."

    implications = {
        'policymakers': f"Monitor emerging species trends to anticipate changes in antibiotic needs.",
        'clinicians': f"Be aware of shifts in predominant pathogens over time.",
        'researchers': f"Investigate the drivers behind species frequency changes (e.g., outbreaks, changes in diagnostic practices).",
        'public': f"Understanding which bacteria are becoming more common helps public health prepare."
    }
    return fig, obs, implications

# Species Distribution by Susceptibility
def plot_species_by_susceptibility(df, antibiotic_name, top_n=10):
    """Stacked bar chart of species counts by interpretation."""
    mask = df['Antibiotic'] == antibiotic_name
    df_filtered = df[mask]
    if df_filtered.empty:
        return None, f"No data for {antibiotic_name}.", {}

    top_species = df_filtered['Species'].value_counts().head(top_n).index
    df_top = df_filtered[df_filtered['Species'].isin(top_species)]
    counts = df_top.groupby(['Species', 'Interpretation']).size().reset_index(name='Count')

    color_map = {'Resistant': 'red', 'Intermediate': 'yellow', 'Susceptible': 'green'}
    fig = px.bar(counts, x='Species', y='Count', color='Interpretation',
                 title=f'Species Distribution by Susceptibility to {antibiotic_name}',
                 barmode='stack', color_discrete_map=color_map)
    fig.update_layout(xaxis_tickangle=-45)

    # Observation: species with highest resistance, considering sample size
    res_rates = df_top.groupby('Species').agg(
        resistant=('Interpretation', lambda x: (x == 'Resistant').sum()),
        total=('Interpretation', 'count')
    ).reset_index()
    res_rates['rate'] = (res_rates['resistant'] / res_rates['total'] * 100)
    
    # Filter to species with at least a minimum sample size for meaningful comparison
    min_sample = max(5, res_rates['total'].quantile(0.25))  # At least 25th percentile of sample sizes
    res_rates_filtered = res_rates[res_rates['total'] >= min_sample].sort_values('rate', ascending=False)
    
    # Calculate statistics
    mean_rate = res_rates_filtered['rate'].mean()
    median_rate = res_rates_filtered['rate'].median()
    min_rate = res_rates_filtered['rate'].min()
    max_rate = res_rates_filtered['rate'].max()
    rate_range = max_rate - min_rate
    
    if not res_rates_filtered.empty:
        top_res = res_rates_filtered.iloc[0]['Species']
        top_rate = res_rates_filtered.iloc[0]['rate']
        top_count = res_rates_filtered.iloc[0]['total']
        top_resistant = int(res_rates_filtered.iloc[0]['resistant'])
        
        # Get 2nd and 3rd highest for comparison
        second_res = res_rates_filtered.iloc[1]['Species'] if len(res_rates_filtered) > 1 else None
        second_rate = res_rates_filtered.iloc[1]['rate'] if len(res_rates_filtered) > 1 else None
        rate_diff_2nd = top_rate - second_rate if second_res else 0
        
        third_res = res_rates_filtered.iloc[2]['Species'] if len(res_rates_filtered) > 2 else None
        third_rate = res_rates_filtered.iloc[2]['rate'] if len(res_rates_filtered) > 2 else None
        
        # Get lowest resistance
        lowest_res = res_rates_filtered.iloc[-1]['Species']
        lowest_rate = res_rates_filtered.iloc[-1]['rate']
        
        obs = f"**{top_res}** exhibits the highest resistance rate at {top_rate:.1f}% ({top_resistant} resistant out of {int(top_count)} isolates). "
        
        if second_res:
            obs += f"This is {rate_diff_2nd:.1f} percentage points higher than **{second_res}** ({second_rate:.1f}%), "
            if third_res:
                obs += f"and {top_rate - third_rate:.1f} percentage points higher than **{third_res}** ({third_rate:.1f}%). "
            else:
                obs += f"indicating a notable distinction in resistance profiles. "
        
        obs += f"Among the {len(res_rates_filtered)} species analyzed, resistance rates range from {min_rate:.1f}% (**{lowest_res}**) to {max_rate:.1f}%, with a mean of {mean_rate:.1f}% and median of {median_rate:.1f}%. "
        obs += f"This {rate_range:.1f} percentage point spread indicates substantial variation in susceptibility patterns across species, suggesting species-specific factors (intrinsic resistance, clonal spread, or selective pressure) may play significant roles in resistance to {antibiotic_name}."
    else:
        # If no species meet minimum threshold, just use highest rate overall
        res_rates_sorted = res_rates.sort_values('rate', ascending=False)
        top_res = res_rates_sorted.iloc[0]['Species']
        top_rate = res_rates_sorted.iloc[0]['rate']
        top_count = res_rates_sorted.iloc[0]['total']
        obs = f"**{top_res}** has the highest resistance rate at {top_rate:.1f}% ({int(res_rates_sorted.iloc[0]['resistant'])} resistant out of {int(top_count)} isolates), "
        obs += f"followed by other species as shown. This indicates that certain species are more likely to carry resistance to {antibiotic_name}."

    implications = {
        'policymakers': f"Species-specific surveillance and targeted stewardship programs should prioritize high-resistance species like **{top_res}**, while monitoring low-resistance species for emerging resistance.",
        'clinicians': f"When **{top_res}** is identified, consider alternative antibiotics. Culture and susceptibility testing is essential to guide therapy, especially for high-risk species.",
        'researchers': f"The wide variation in resistance rates ({rate_range:.1f}%) across species warrants investigation into species-specific resistance mechanisms, horizontal gene transfer patterns, and clonal lineage prevalence for {antibiotic_name}.",
        'public': f"Different bacteria species have different natural resistance patterns. Your healthcare provider uses culture results to identify the specific bacteria and choose the most effective antibiotic."
    }
    return fig, obs, implications

# MIC Distribution per Species
def plot_mic_by_species(df, antibiotic_name, top_n=10):
    """Box plot of MIC values by species."""
    mask = (df['Antibiotic'] == antibiotic_name) & (df['MIC'].notna())
    df_filtered = df[mask].copy()
    if df_filtered.empty:
        return None, f"No MIC data for {antibiotic_name}.", {}

    # Convert MIC to numeric
    df_filtered['MIC'] = pd.to_numeric(df_filtered['MIC'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['MIC'])

    top_species = df_filtered['Species'].value_counts().head(top_n).index
    df_top = df_filtered[df_filtered['Species'].isin(top_species)]

    color_map = {'Resistant': 'red', 'Intermediate': 'yellow', 'Susceptible': 'green'}
    fig = px.box(df_top, x='Species', y='MIC', color='Interpretation',
                 title=f'MIC Distribution by Species for {antibiotic_name}',
                 labels={'MIC': 'MIC (µg/mL)'}, color_discrete_map=color_map)
    fig.update_layout(xaxis_tickangle=-45)
    fig.update_yaxes(type="log")

    # Observation: species with highest median MIC among resistant isolates
    resistant = df_top[df_top['Interpretation'] == 'Resistant']
    if not resistant.empty:
        med_mic = resistant.groupby('Species')['MIC'].median().sort_values(ascending=False)
        top_species_mic = med_mic.index[0]
        top_mic = med_mic.iloc[0]
        obs = f"Among resistant isolates, **{top_species_mic}** shows the highest median MIC ({top_mic:.1f} µg/mL). "
        obs += f"This suggests that for this species, higher doses or alternative agents may be needed."
    else:
        obs = "No resistant isolates with MIC data available."

    implications = {
        'policymakers': f"Breakpoints should consider species‑specific MIC distributions.",
        'clinicians': f"For species with high MICs, even if categorized as susceptible, consider using higher doses or combination therapy.",
        'researchers': f"Study the mechanisms that lead to elevated MICs in certain species.",
        'public': f"MIC testing helps ensure the right dose of antibiotic is used."
    }
    return fig, obs, implications


# Species Susceptibility Trend Over Time
def plot_species_trend_by_susceptibility(df, antibiotic_name, species):
    """Line chart of susceptibility categories over time for a given species."""
    mask = (df['Antibiotic'] == antibiotic_name) & (df['Species'] == species)
    df_species = df[mask].copy()
    if df_species.empty:
        return None, f"No data for {antibiotic_name} in species {species}.", {}

    yearly = df_species.groupby(['Year', 'Interpretation']).size().reset_index(name='Count')
    total_per_year = yearly.groupby('Year')['Count'].transform('sum')
    yearly['Percent'] = (yearly['Count'] / total_per_year) * 100

    color_map = {'Resistant': 'red', 'Intermediate': 'yellow', 'Susceptible': 'green'}
    fig = px.line(yearly, x='Year', y='Percent', color='Interpretation',
                  title=f'Susceptibility Trend for {species} to {antibiotic_name}',
                  markers=True, color_discrete_map=color_map)
    fig.update_layout(yaxis_range=[0,100])

    # Trend for resistant category
    resistant_data = yearly[yearly['Interpretation'] == 'Resistant']
    if len(resistant_data) >= 3:
        slope, _, _, p, _ = linregress(resistant_data['Year'], resistant_data['Percent'])
        if slope > 0 and p < 0.05:
            obs = f"Resistance in **{species}** is significantly increasing at a rate of {slope:.2f}% per year (p={p:.3f}). "
        elif slope < 0 and p < 0.05:
            obs = f"Resistance in **{species}** is significantly decreasing at a rate of {abs(slope):.2f}% per year (p={p:.3f}). "
        else:
            obs = f"No significant trend in resistance for **{species}** (p={p:.3f}). "
        obs += f"Over the study period, the proportion of resistant isolates has changed from {resistant_data['Percent'].iloc[0]:.1f}% to {resistant_data['Percent'].iloc[-1]:.1f}%."
    else:
        obs = f"Insufficient years to determine a trend for {species}."

    implications = {
        'policymakers': f"Monitor this species closely; if resistance is increasing, consider targeted interventions.",
        'clinicians': f"For {species} infections, be aware of the evolving resistance pattern.",
        'researchers': f"Investigate the reasons behind the trend in this species (e.g., clonal spread, plasmid acquisition).",
        'public': f"Some bacteria change their resistance over time; regular surveillance helps doctors stay informed."
    }
    return fig, obs, implications

