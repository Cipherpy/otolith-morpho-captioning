import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os

def analyze_feature_descriptions(csv_file, features_file):
    """
    Analyze a CSV file containing specimen data with descriptions that match features
    listed in a separate text file. Count occurrences of features and unique species,
    then visualize the results.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing specimen data
    features_file : str
        Path to the text file containing feature definitions
    """
    # Load the CSV data
    try:
        df = pd.read_csv(csv_file)
        print(len(df))
        print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        return None
    
    # Check if required columns exist
    required_cols = ["Description", "Class"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print("Available columns:", df.columns.tolist())
        return None
    
    # Load the features list from text file
    try:
        with open(features_file, 'r') as f:
            feature_lines = f.readlines()
        
        # Parse features (assuming one feature per line)
        features = [line.strip() for line in feature_lines if line.strip()]
        print(f"Loaded {len(features)} features from text file")
    except Exception as e:
        print(f"Error loading features file: {str(e)}")
        return None
    
    # Initialize counters for each feature
    feature_specimen_counts = {feature: 0 for feature in features}
    feature_species_counts = {feature: set() for feature in features}
    
    # Count occurrences of each feature in the descriptions
    for _, row in df.iterrows():
        description = str(row['Description'])
        species = row['Class']
        
        for feature in features:
            # Check if the feature appears in the description
            if re.search(r'\b' + re.escape(feature) + r'\b', description, re.IGNORECASE):
                feature_specimen_counts[feature] += 1
                feature_species_counts[feature].add(species)
    
    # Convert sets to counts
    feature_unique_species_counts = {feature: len(species_set) for feature, species_set in feature_species_counts.items()}
    
    # Create a dataframe for plotting
    plot_data = pd.DataFrame({
        'Feature': list(feature_specimen_counts.keys()),
        'Specimen Count': list(feature_specimen_counts.values()),
        'Unique Species Count': list(feature_unique_species_counts.values())
    })
    
    # Sort by specimen count for better visualization
    plot_data = plot_data.sort_values('Specimen Count', ascending=False)
    
    # Print the counts
    print("\nFeature occurrence counts:")
    for feature, count in feature_specimen_counts.items():
        species_count = feature_unique_species_counts[feature]
        print(f"{feature}: {count} specimens, {species_count} unique species")
    
    # Create directory for visualizations
    os.makedirs('feature_analysis', exist_ok=True)
    
    # Create bar plot comparing specimen counts and species counts
    plt.figure(figsize=(12, 8))
    
    # Set width of bars
    barWidth = 0.4
    
    # Set positions of bars on X axis
    r1 = np.arange(len(plot_data))
    r2 = [x + barWidth for x in r1]
    
      # Create bars
    plt.bar(r1, plot_data['Specimen Count'], width=barWidth, label='Specimen Count', color='blue', alpha=0.7)
    plt.bar(r2, plot_data['Unique Species Count'], width=barWidth, label='Unique Species Count', color='green', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('Features', fontweight='bold', fontsize=12)
    plt.ylabel('Count', fontweight='bold', fontsize=12)
    plt.title('Specimen and Unique Species Counts by Feature', fontweight='bold', fontsize=14)
    
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth/2 for r in range(len(plot_data))], plot_data['Feature'], rotation=45, ha='right')
    
    # Create legend
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('feature_analysis/feature_counts_comparison.png', dpi=300)
    
    # Create a horizontal bar chart for better readability with long feature names
    plt.figure(figsize=(10, max(8, len(plot_data) * 0.4)))  # Adjust height based on number of features
    
    # Create horizontal bars
    plt.barh(plot_data['Feature'], plot_data['Specimen Count'], color='blue', alpha=0.7, label='Specimen Count')
    plt.barh(plot_data['Feature'], plot_data['Unique Species Count'], color='green', alpha=0.7, 
             left=plot_data['Specimen Count'], label='Unique Species Count')
    
    # Add labels and title
    plt.xlabel('Count', fontweight='bold', fontsize=12)
    plt.ylabel('Features', fontweight='bold', fontsize=12)
    plt.title('Specimen and Unique Species Counts by Feature', fontweight='bold', fontsize=14)
    
    # Create legend
    plt.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('feature_analysis/feature_counts_horizontal.png', dpi=300)
    
    # Create a separate bar chart for each count type (better for comparison across features)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # Specimen count
    sns.barplot(x='Feature', y='Specimen Count', data=plot_data, ax=ax1, palette='Blues_d')
    ax1.set_title('Number of Specimens by Feature')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Species count
    sns.barplot(x='Feature', y='Unique Species Count', data=plot_data, ax=ax2, palette='Greens_d')
    ax2.set_title('Number of Unique Species by Feature')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('feature_analysis/feature_counts_separate.png', dpi=300)
    
    # Create a heatmap for top species and features
    # Get the top 10 most common species
    top_species = df['Class'].value_counts().nlargest(10).index.tolist()
    
    # Create a matrix of species vs features
    heatmap_data = []
    for species in top_species:
        species_rows = df[df['Class'] == species]
        
        row_data = {'Class': species}
        
        for feature in features:
            # Count specimens of this species that have this feature
            count = 0
            for _, specimen_row in species_rows.iterrows():
                description = str(specimen_row['Description'])
                if re.search(r'\b' + re.escape(feature) + r'\b', description, re.IGNORECASE):
                    count += 1
            
            row_data[feature] = count
        
        heatmap_data.append(row_data)
    
    # Convert to DataFrame
    heatmap_df = pd.DataFrame(heatmap_data)
    heatmap_df.set_index('Class', inplace=True)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt='d')
    plt.title('Feature Distribution Among Top 10 Species')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('feature_analysis/species_feature_heatmap.png', dpi=300)
    
    # Return the data for further analysis if needed
    return {
        'full_data': df,
        'feature_counts': feature_specimen_counts,
        'species_counts': feature_unique_species_counts,
        'plot_data': plot_data
    }

if __name__ == "__main__":
    # Paths to your files
    csv_file = input("Enter the path to your CSV file: ")
    features_file = input("Enter the path to your features text file: ")
    
    # Run the analysis
    results = analyze_feature_descriptions(csv_file, features_file)
    
    if results:
        print("\nAnalysis complete. Visualizations saved in 'feature_analysis/' directory.")