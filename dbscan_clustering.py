import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import os

# Create directory for DBSCAN results
os.makedirs('dbscan_results', exist_ok=True)

# Load the preprocessed data
print("Loading preprocessed dataset...")
df = pd.read_csv('processed_data/student_depression_processed.csv')
print(f"Dataset shape: {df.shape}")

# Select features for clustering
# Remove ID and target variable (Depression)
features = df.drop(['id', 'Depression'], axis=1).columns.tolist()
X = df[features].values

print(f"Selected {len(features)} features for clustering")
print("Features:", features)

# Standardize the data
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal eps parameter using k-distance graph
print("\nFinding optimal epsilon parameter...")
n_samples = min(5000, X_scaled.shape[0])  # Use a subset for k-distance computation to avoid memory issues
sample_indices = np.random.choice(X_scaled.shape[0], n_samples, replace=False)
X_sample = X_scaled[sample_indices]

# Compute k-distance graph
k = 5  # number of neighbors to consider
nbrs = NearestNeighbors(n_neighbors=k).fit(X_sample)
distances, indices = nbrs.kneighbors(X_sample)

# Sort distances in ascending order
distances = np.sort(distances[:, k-1])

# Plot k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(range(len(distances)), distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'Distance to {k}th nearest neighbor')
plt.title('K-distance Graph')
plt.grid(True)
plt.savefig('dbscan_results/k_distance_graph.png')
plt.close()

# Try a wider range of eps values and min_samples
eps_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
min_samples_values = [10, 15, 20, 25, 30]

# Create a dataframe to store results
results = []

print("\nTesting different DBSCAN parameters...")
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X_scaled)
        
        # Number of clusters (excluding noise)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        # Number of noise points
        n_noise = list(cluster_labels).count(-1)
        
        # Proportion of noise points
        noise_ratio = n_noise / len(cluster_labels)
        
        # Calculate silhouette score if there are at least 2 clusters and not all points are noise
        silhouette_avg = -1  # Default value if silhouette score can't be computed
        if n_clusters >= 2 and n_noise < len(cluster_labels):
            try:
                # Filter out noise points for silhouette calculation
                mask = cluster_labels != -1
                if sum(mask) > n_clusters:  # Need more points than clusters for silhouette score
                    silhouette_avg = silhouette_score(X_scaled[mask], cluster_labels[mask])
            except:
                pass  # Keep default value if calculation fails
        
        print(f"DBSCAN eps={eps}, min_samples={min_samples}: {n_clusters} clusters, {n_noise} noise points ({noise_ratio:.2%}), silhouette={silhouette_avg:.3f}")
        
        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'silhouette_score': silhouette_avg
        })

# Convert results to dataframe
results_df = pd.DataFrame(results)
print("\nDBSCAN parameter testing results:")
print(results_df)

# Save results
results_df.to_csv('dbscan_results/dbscan_parameter_testing.csv', index=False)

# Automated parameter selection logic
# Look for parameters that give 2-10 clusters with noise ratio < 30% and best silhouette score
candidate_params = results_df[
    (results_df['n_clusters'] >= 2) & 
    (results_df['n_clusters'] <= 10) & 
    (results_df['noise_ratio'] < 0.3) &
    (results_df['silhouette_score'] > 0)  # Valid silhouette scores only
]

if not candidate_params.empty:
    # Select parameters with the best silhouette score
    optimal_result = candidate_params.sort_values(by='silhouette_score', ascending=False).head(1)
else:
    # Fallback to original strategy if no good candidates with silhouette score
    optimal_result = results_df[(results_df['n_clusters'] >= 2) & 
                                (results_df['n_clusters'] <= 10) & 
                                (results_df['noise_ratio'] < 0.3)]
    
    if optimal_result.empty:
        # If still no ideal parameters, choose ones with reasonable clusters and lowest noise
        optimal_result = results_df[(results_df['n_clusters'] >= 2) & 
                                   (results_df['n_clusters'] <= 20)].sort_values(by='noise_ratio', ascending=True).head(1)
        
        if optimal_result.empty:
            # Last resort: choose parameters with most clusters and least noise
            optimal_result = results_df.sort_values(by=['n_clusters', 'noise_ratio'], 
                                                   ascending=[False, True]).head(1)

optimal_eps = optimal_result.iloc[0]['eps']
optimal_min_samples = int(optimal_result.iloc[0]['min_samples'])

print(f"\nSelected optimal parameters: eps={optimal_eps}, min_samples={optimal_min_samples}")
print(f"This produces {optimal_result.iloc[0]['n_clusters']} clusters with {optimal_result.iloc[0]['noise_ratio']:.2%} noise points")
if optimal_result.iloc[0]['silhouette_score'] > 0:
    print(f"Silhouette score: {optimal_result.iloc[0]['silhouette_score']:.3f}")

# Apply DBSCAN with optimal parameters
print("\nPerforming DBSCAN clustering with optimal parameters...")
dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Evaluate clustering quality
n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_dbscan_noise = list(dbscan_labels).count(-1)
dbscan_noise_ratio = n_dbscan_noise / len(dbscan_labels)

print(f"DBSCAN produced {n_dbscan_clusters} clusters with {n_dbscan_noise} noise points ({dbscan_noise_ratio:.2%})")

# Check DBSCAN results
if n_dbscan_clusters <= 2 or n_dbscan_clusters > 20 or dbscan_noise_ratio > 0.3 or dbscan_noise_ratio < 0.01:
    print("\nDBSCAN did not produce satisfactory results. Trying OPTICS algorithm...")
    
    # Create a list of OPTICS parameters to try
    optics_params = [
        {'min_samples': 15, 'xi': 0.05, 'min_cluster_size': 50},
        {'min_samples': 10, 'xi': 0.03, 'min_cluster_size': 30},
        {'min_samples': 5, 'xi': 0.01, 'min_cluster_size': 20}
    ]
    
    best_optics = None
    best_score = -1
    best_labels = None
    best_n_clusters = 0
    best_noise_ratio = 1.0
    
    print("\nTesting different OPTICS parameters:")
    for params in optics_params:
        print(f"Testing OPTICS with {params}...")
        optics = OPTICS(
            min_samples=params['min_samples'], 
            xi=params['xi'], 
            min_cluster_size=params['min_cluster_size'],
            metric='euclidean'
        )
        labels = optics.fit_predict(X_scaled)
        
        # Count clusters and noise points
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        noise_ratio = n_noise / len(labels)
        
        # Calculate silhouette score if possible
        sil_score = -1
        if n_clusters >= 2 and n_noise < len(labels):
            try:
                # Filter out noise points for silhouette calculation
                mask = labels != -1
                if sum(mask) > n_clusters and sum(mask) > 0:  # Need more points than clusters
                    sil_score = silhouette_score(X_scaled[mask], labels[mask])
            except:
                pass
        
        print(f"OPTICS with {params}: {n_clusters} clusters, {n_noise} noise points ({noise_ratio:.2%}), silhouette={sil_score:.3f}")
        
        # Score this clustering based on number of clusters and noise ratio
        # We want a reasonable number of clusters (3-15) and moderate noise ratio (0.05-0.3)
        is_good_clustering = (3 <= n_clusters <= 15) and (0.05 <= noise_ratio <= 0.3)
        
        # Update best parameters if this is better
        if is_good_clustering and (sil_score > best_score or best_score == -1):
            best_optics = optics
            best_score = sil_score
            best_labels = labels
            best_n_clusters = n_clusters
            best_noise_ratio = noise_ratio
    
    # If we found good OPTICS parameters, use them
    if best_optics is not None:
        print(f"\nSelected OPTICS clustering with {best_n_clusters} clusters and {best_noise_ratio:.2%} noise ratio")
        cluster_labels = best_labels
        clustering_method = "OPTICS"
    else:
        # Try one more time with more aggressive parameters
        print("\nTrying OPTICS with more aggressive parameters...")
        optics = OPTICS(min_samples=3, xi=0.005, min_cluster_size=10, metric='euclidean')
        cluster_labels = optics.fit_predict(X_scaled)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        noise_ratio = n_noise / len(cluster_labels)
        
        print(f"Final OPTICS attempt: {n_clusters} clusters, {n_noise} noise points ({noise_ratio:.2%})")
        
        if n_clusters >= 3:
            clustering_method = "OPTICS"
        else:
            # If all else fails, fall back to DBSCAN
            print("\nFalling back to DBSCAN results")
            cluster_labels = dbscan_labels
            clustering_method = "DBSCAN"
else:
    # DBSCAN produced reasonable results, use them
    print("DBSCAN produced reasonable clustering results.")
    cluster_labels = dbscan_labels
    clustering_method = "DBSCAN"

# Assign clusters to dataframe
df['Cluster'] = cluster_labels

# Replace -1 (noise) with a more descriptive label for the report
df['Cluster_Label'] = df['Cluster'].apply(lambda x: 'Noise' if x == -1 else f'Cluster {x}')

# Count instances in each cluster
cluster_counts = df['Cluster_Label'].value_counts()
print("\nCluster distribution:")
for cluster, count in cluster_counts.items():
    print(f"{cluster}: {count} instances ({count/len(df)*100:.2f}%)")

# Save the clustering method used for reporting
with open('dbscan_results/clustering_method_used.txt', 'w') as f:
    f.write(clustering_method)
    
# Create directory for results using the appropriate method name
results_dir = 'optics_results' if clustering_method == 'OPTICS' else 'dbscan_results'
os.makedirs(results_dir, exist_ok=True)

# Analyze clusters
print("\nAnalyzing clusters...")
# Group by cluster label to include noise points as a separate group
cluster_analysis = df.groupby('Cluster_Label').mean()[features]
print("\nCluster centers (mean values for each feature):")
print(cluster_analysis)

# Save cluster analysis to CSV
cluster_analysis.to_csv(f'{results_dir}/cluster_analysis.csv')

# Visualize clusters using PCA for dimensionality reduction
print("\nVisualizing clusters using PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 10))
# Create a custom colormap with grey (-1) for noise points
unique_clusters = sorted(df['Cluster'].unique())
# Use a cyclic colormap for the clusters
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_clusters)))
# Replace the color for noise (-1) with grey
if -1 in unique_clusters:
    noise_idx = unique_clusters.index(-1)
    colors[noise_idx] = [0.7, 0.7, 0.7, 1.0]  # Grey

for i, cluster in enumerate(unique_clusters):
    mask = df['Cluster'] == cluster
    label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                color=colors[i], 
                s=50 if cluster == -1 else 70,
                alpha=0.5 if cluster == -1 else 0.8,
                label=label)

plt.title(f'{clustering_method} Clustering')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(f'{results_dir}/pca_visualization.png')
plt.close()

# Analyze relationship between clusters and depression
print("\nAnalyzing relationship between clusters and depression...")
depression_by_cluster = df.groupby('Cluster_Label')['Depression'].mean().sort_values()
print("\nAverage depression score by cluster:")
print(depression_by_cluster)

plt.figure(figsize=(12, 7))
ax = depression_by_cluster.plot(kind='bar')
plt.title('Average Depression Score by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Depression Score')
plt.axhline(y=df['Depression'].mean(), color='red', linestyle='--', label='Overall Average')
plt.grid(True, alpha=0.3, axis='y')
plt.legend()

# Add values on top of bars
for i, v in enumerate(depression_by_cluster):
    ax.text(i, v + 0.02, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig(f'{results_dir}/depression_by_cluster.png')
plt.close()

# Analyze key features by cluster
print("\nAnalyzing key features by cluster...")
# Select a subset of important features to visualize
key_features = ['Age', 'Academic Pressure', 'CGPA', 'Study Satisfaction', 
                'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
                'Family History of Mental Illness', 'Combined_Stress_Index']
key_features = [f for f in key_features if f in features]

# Create a heatmap of cluster centers for key features
plt.figure(figsize=(14, 10))
key_cluster_centers = cluster_analysis[key_features]

# Normalize the data for better visualization
key_cluster_centers_scaled = pd.DataFrame(
    scaler.fit_transform(key_cluster_centers),
    index=key_cluster_centers.index,
    columns=key_cluster_centers.columns
)

# Create a heatmap
sns.heatmap(key_cluster_centers_scaled, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Normalized Cluster Centers for Key Features')
plt.tight_layout()
plt.savefig(f'{results_dir}/key_features_heatmap.png')
plt.close()

# Summarize the characteristics of each cluster
print("\nCluster Characteristics Summary:")
for cluster_label in sorted(df['Cluster_Label'].unique()):
    print(f"\n{cluster_label}:")
    
    # Skip if not in the normalized dataframe (shouldn't happen, but just in case)
    if cluster_label not in key_cluster_centers_scaled.index:
        continue
    
    # Get the top 5 distinctive features for this cluster (highest absolute z-scores)
    cluster_features = key_cluster_centers_scaled.loc[cluster_label].abs().sort_values(ascending=False)
    top_features = cluster_features.head(5).index.tolist()
    
    for feature in top_features:
        raw_value = key_cluster_centers.loc[cluster_label, feature]
        scaled_value = key_cluster_centers_scaled.loc[cluster_label, feature]
        direction = "high" if scaled_value > 0 else "low"
        print(f"  - {feature}: {direction} ({raw_value:.2f}, z-score: {scaled_value:.2f})")
    
    # Depression info for this cluster
    cluster_depression = df[df['Cluster_Label'] == cluster_label]['Depression'].mean()
    overall_depression = df['Depression'].mean()
    depression_diff = cluster_depression - overall_depression
    
    depression_status = "higher" if depression_diff > 0 else "lower"
    print(f"  - Depression: {depression_status} than average by {abs(depression_diff):.2f} ({cluster_depression:.2f} vs. {overall_depression:.2f} overall)")

# Create a 3D PCA visualization to better show cluster separation
print("\nCreating 3D visualization of clusters...")
pca_3d = PCA(n_components=3)
X_pca_3d = pca_3d.fit_transform(X_scaled)

# Create the 3D plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster
for i, cluster in enumerate(unique_clusters):
    mask = df['Cluster'] == cluster
    label = 'Noise' if cluster == -1 else f'Cluster {cluster}'
    ax.scatter(X_pca_3d[mask, 0], X_pca_3d[mask, 1], X_pca_3d[mask, 2],
               color=colors[i],
               s=30 if cluster == -1 else 50,
               alpha=0.5 if cluster == -1 else 0.8,
               label=label)

ax.set_title(f'3D PCA Visualization of {clustering_method} Clusters')
ax.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.2%} variance)')
ax.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.2%} variance)')
ax.legend()
plt.savefig(f'{results_dir}/3d_pca_visualization.png')
plt.close()

print(f"\n{clustering_method} clustering analysis complete. Results saved to '{results_dir}' directory.") 