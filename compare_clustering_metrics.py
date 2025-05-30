import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
import os

# Create directory for comparison results
os.makedirs('comparison_results', exist_ok=True)

# Load the preprocessed data
print("Loading preprocessed dataset...")
df = pd.read_csv('processed_data/student_depression_processed.csv')
print(f"Dataset shape: {df.shape}")

# Extract features and target
features = df.drop(['id', 'Depression'], axis=1).columns.tolist()
X = df[features].values
y = df['Depression'].values  # Depression as reference label

# Create binary depression label for external metrics (0 for no depression, 1 for depression)
binary_depression = (df['Depression'] > 0).astype(int)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Load K-means model results (we'll need to recreate them since we don't have the labels file)
print("\nRecreating K-means clustering model...")
kmeans_analysis = pd.read_csv('clustering_results/kmeans_cluster_analysis.csv')
n_kmeans_clusters = len(kmeans_analysis)
print(f"K-means used {n_kmeans_clusters} clusters")

# Recreate K-means model
kmeans = KMeans(n_clusters=n_kmeans_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Load Hierarchical clustering results
print("\nRecreating Hierarchical clustering model...")
hc_analysis = pd.read_csv('hierarchical_results/hierarchical_cluster_analysis.csv')
n_hc_clusters = len(hc_analysis)
print(f"Hierarchical clustering used {n_hc_clusters} clusters")

# Recreate hierarchical clustering model
hc = AgglomerativeClustering(n_clusters=n_hc_clusters)
hc_labels = hc.fit_predict(X_scaled)

# Create a function to calculate internal metrics
def calculate_internal_metrics(X, labels):
    try:
        silhouette = silhouette_score(X, labels)
    except:
        silhouette = np.nan
    
    try:
        db_score = davies_bouldin_score(X, labels)
    except:
        db_score = np.nan
    
    try:
        ch_score = calinski_harabasz_score(X, labels)
    except:
        ch_score = np.nan
    
    return {
        'silhouette_score': silhouette,
        'davies_bouldin_score': db_score,
        'calinski_harabasz_score': ch_score
    }

# Calculate cohesion (within-cluster sum of squares)
def calculate_cohesion(X, labels):
    unique_labels = np.unique(labels)
    wss = 0
    
    for label in unique_labels:
        if label == -1:  # Skip noise points if any
            continue
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        wss += np.sum(np.square(cluster_points - centroid))
    
    return wss

# Calculate separation (between-cluster sum of squares)
def calculate_separation(X, labels):
    unique_labels = np.unique(labels)
    overall_centroid = np.mean(X, axis=0)
    bss = 0
    
    for label in unique_labels:
        if label == -1:  # Skip noise points if any
            continue
        cluster_points = X[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        bss += len(cluster_points) * np.sum(np.square(centroid - overall_centroid))
    
    return bss

# Calculate external metrics (using Depression as reference)
def calculate_external_metrics(true_labels, cluster_labels):
    # For clustering, we need to map cluster labels to target classes
    # We'll assign each cluster to the majority class within it
    
    # Create a mapping from cluster label to predicted class
    cluster_to_class = {}
    unique_clusters = np.unique(cluster_labels)
    
    for cluster in unique_clusters:
        if cluster == -1:  # Skip noise points
            continue
        mask = (cluster_labels == cluster)
        if np.sum(mask) > 0:  # Ensure cluster has points
            # Assign the majority class
            cluster_to_class[cluster] = int(np.round(np.mean(true_labels[mask])))
    
    # Map cluster labels to predicted classes
    predicted_labels = np.zeros_like(true_labels)
    for i, cluster in enumerate(cluster_labels):
        if cluster in cluster_to_class:
            predicted_labels[i] = cluster_to_class[cluster]
        else:
            # For noise points or unmapped clusters, guess the majority class
            predicted_labels[i] = int(np.round(np.mean(true_labels)))
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Calculate metrics for both methods
print("\nCalculating metrics for K-means clustering...")
kmeans_internal_metrics = calculate_internal_metrics(X_scaled, kmeans_labels)
kmeans_cohesion = calculate_cohesion(X_scaled, kmeans_labels)
kmeans_separation = calculate_separation(X_scaled, kmeans_labels)
kmeans_external_metrics = calculate_external_metrics(binary_depression, kmeans_labels)

print("\nCalculating metrics for Hierarchical clustering...")
hc_internal_metrics = calculate_internal_metrics(X_scaled, hc_labels)
hc_cohesion = calculate_cohesion(X_scaled, hc_labels)
hc_separation = calculate_separation(X_scaled, hc_labels)
hc_external_metrics = calculate_external_metrics(binary_depression, hc_labels)

# Create a summary DataFrame
metrics_summary = pd.DataFrame({
    'Metric': [
        'Silhouette Score', 
        'Davies-Bouldin Score', 
        'Calinski-Harabasz Score',
        'Cohesion (Lower is better)',
        'Separation (Higher is better)',
        'Accuracy',
        'Precision',
        'Recall',
        'F1 Score'
    ],
    'K-means': [
        kmeans_internal_metrics['silhouette_score'],
        kmeans_internal_metrics['davies_bouldin_score'],
        kmeans_internal_metrics['calinski_harabasz_score'],
        kmeans_cohesion,
        kmeans_separation,
        kmeans_external_metrics['accuracy'],
        kmeans_external_metrics['precision'],
        kmeans_external_metrics['recall'],
        kmeans_external_metrics['f1_score']
    ],
    'Hierarchical': [
        hc_internal_metrics['silhouette_score'],
        hc_internal_metrics['davies_bouldin_score'],
        hc_internal_metrics['calinski_harabasz_score'],
        hc_cohesion,
        hc_separation,
        hc_external_metrics['accuracy'],
        hc_external_metrics['precision'],
        hc_external_metrics['recall'],
        hc_external_metrics['f1_score']
    ]
})

print("\nMetrics Comparison:")
print(metrics_summary)

# Save metrics to CSV
metrics_summary.to_csv('comparison_results/clustering_metrics_comparison.csv', index=False)

# Create bar charts for comparison
plt.figure(figsize=(15, 10))

# Create subplots for different metric groups
metrics_groups = [
    ('Internal Validation Metrics', ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score']),
    ('Cohesion and Separation', ['Cohesion (Lower is better)', 'Separation (Higher is better)']),
    ('External Validation Metrics', ['Accuracy', 'Precision', 'Recall', 'F1 Score'])
]

for i, (group_name, metrics) in enumerate(metrics_groups):
    plt.subplot(len(metrics_groups), 1, i+1)
    
    # Filter metrics for this group
    group_df = metrics_summary[metrics_summary['Metric'].isin(metrics)]
    
    # Set up positions for bars
    x = np.arange(len(group_df['Metric']))
    width = 0.35
    
    # Create bars
    kmeans_bars = plt.bar(x - width/2, group_df['K-means'], width, label='K-means')
    hc_bars = plt.bar(x + width/2, group_df['Hierarchical'], width, label='Hierarchical')
    
    # Customize plot
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(group_name)
    plt.xticks(x, group_df['Metric'], rotation=45, ha='right')
    plt.legend()
    
    # Add value labels on top of bars
    for bars in [kmeans_bars, hc_bars]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                plt.text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()

plt.savefig('comparison_results/clustering_metrics_comparison.png')
plt.close()

# Write a comparison summary
with open('comparison_results/clustering_comparison_summary.md', 'w') as f:
    f.write("# Clustering Methods Comparison: K-means vs. Hierarchical\n\n")
    
    f.write("## Overview\n")
    f.write("This document compares the performance of K-means and Hierarchical clustering on the student depression dataset.\n\n")
    
    f.write("## Metrics Comparison\n\n")
    f.write(metrics_summary.to_markdown(index=False))
    f.write("\n\n")
    
    f.write("## Interpretation\n\n")
    
    # Silhouette score (higher is better)
    if kmeans_internal_metrics['silhouette_score'] > hc_internal_metrics['silhouette_score']:
        silhouette_winner = "K-means"
    else:
        silhouette_winner = "Hierarchical"
        
    f.write(f"### Silhouette Score\n")
    f.write(f"**{silhouette_winner}** shows a better silhouette score, indicating better-defined and more separated clusters. ")
    f.write("Silhouette score ranges from -1 to 1, with higher values indicating better cluster separation and cohesion.\n\n")
    
    # Davies-Bouldin (lower is better)
    if kmeans_internal_metrics['davies_bouldin_score'] < hc_internal_metrics['davies_bouldin_score']:
        db_winner = "K-means"
    else:
        db_winner = "Hierarchical"
        
    f.write(f"### Davies-Bouldin Score\n")
    f.write(f"**{db_winner}** has a lower Davies-Bouldin index, indicating better cluster separation. ")
    f.write("Lower values indicate better clustering with more distinct, compact clusters.\n\n")
    
    # Calinski-Harabasz (higher is better)
    if kmeans_internal_metrics['calinski_harabasz_score'] > hc_internal_metrics['calinski_harabasz_score']:
        ch_winner = "K-means"
    else:
        ch_winner = "Hierarchical"
        
    f.write(f"### Calinski-Harabasz Score\n")
    f.write(f"**{ch_winner}** shows a higher Calinski-Harabasz score, indicating better-defined clusters. ")
    f.write("Higher values indicate better clustering with more distinct, dense clusters.\n\n")
    
    # Cohesion (lower is better)
    if kmeans_cohesion < hc_cohesion:
        cohesion_winner = "K-means"
    else:
        cohesion_winner = "Hierarchical"
        
    f.write(f"### Cohesion (Within-cluster sum of squares)\n")
    f.write(f"**{cohesion_winner}** achieves better cohesion (lower within-cluster sum of squares). ")
    f.write("Lower values indicate more compact clusters with points closer to their centroids.\n\n")
    
    # Separation (higher is better)
    if kmeans_separation > hc_separation:
        separation_winner = "K-means"
    else:
        separation_winner = "Hierarchical"
        
    f.write(f"### Separation (Between-cluster sum of squares)\n")
    f.write(f"**{separation_winner}** shows better separation between clusters. ")
    f.write("Higher values indicate better clustering with more distinct clusters.\n\n")
    
    # F1 Score (higher is better)
    if kmeans_external_metrics['f1_score'] > hc_external_metrics['f1_score']:
        f1_winner = "K-means"
    else:
        f1_winner = "Hierarchical"
        
    f.write(f"### F1 Score\n")
    f.write(f"When using depression as a reference label, **{f1_winner}** achieves a better F1 score. ")
    f.write("This suggests that this method better identifies patterns related to depression status.\n\n")
    
    # Overall conclusion
    f.write("## Overall Conclusion\n\n")
    
    # Count winners
    kmeans_wins = sum(1 for winner in [silhouette_winner, db_winner, ch_winner, cohesion_winner, separation_winner, f1_winner] if winner == "K-means")
    hc_wins = sum(1 for winner in [silhouette_winner, db_winner, ch_winner, cohesion_winner, separation_winner, f1_winner] if winner == "Hierarchical")
    
    if kmeans_wins > hc_wins:
        overall_winner = "K-means"
        strengths = "simplicity, efficiency, and ability to identify global structure"
        weaknesses = "assumption of spherical clusters and sensitivity to outliers"
    else:
        overall_winner = "Hierarchical"
        strengths = "ability to identify nested cluster structures and handle irregular cluster shapes"
        weaknesses = "computational complexity and sensitivity to the distance metric and linkage method"
    
    f.write(f"Based on the majority of metrics, **{overall_winner}** appears to be the better clustering approach for this student depression dataset. ")
    f.write(f"Its strengths in {strengths} make it more suitable for this particular application, despite {weaknesses}.\n\n")
    
    f.write("Both methods provide valuable insights, but for the specific task of identifying patterns related to depression, ")
    f.write(f"the {overall_winner} approach provides more coherent and well-separated clusters that better align with depression indicators.\n")

print("\nComparison complete. Results saved to 'comparison_results' directory.") 