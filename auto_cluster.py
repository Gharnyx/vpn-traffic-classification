# auto_cluster_vpn.py
# Automatically clusters your flows into VPN-like and non-VPN-like groups
# No manual labels needed - uses all numeric features

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ────────────────────────────────────────────────
    #  CHANGE THIS PATH TO YOUR ACTUAL CSV FILE
    # ────────────────────────────────────────────────
    csv_path = r"D:\traffic_nfstream.csv"

    print("Loading CSV file...")
    df = pd.read_csv(csv_path)

    # Keep only numeric columns (drop strings like IPs, app names, etc.)
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    print(f"Loaded {len(numeric_df)} flows with {numeric_df.shape[1]} numeric features")

    # Clean data: replace inf/nan with 0 (common in flow stats)
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Features for clustering
    X = numeric_df.values

    # Scale features (very important for clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ────────────────────────────────────────────────
    # Run KMeans clustering (2 clusters = VPN vs non-VPN)
    # ────────────────────────────────────────────────
    print("Running KMeans clustering (2 groups)...")
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Add cluster labels to dataframe (0 or 1)
    numeric_df['Cluster'] = kmeans.labels_
    df['Cluster'] = kmeans.labels_   # also add to original df

    # Simple quality score (higher = better separation)
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_scaled, kmeans.labels_)
    print(f"\nClustering quality (Silhouette Score): {score:.3f}")
    print("   > 0.5 = good separation, 0.2–0.5 = ok, <0.2 = weak")

    # ────────────────────────────────────────────────
    # Show average values per cluster (helps you interpret)
    # ────────────────────────────────────────────────
    print("\nCluster 0 average values:")
    print(numeric_df[numeric_df['Cluster'] == 0].mean(numeric_only=True).round(3))

    print("\nCluster 1 average values:")
    print(numeric_df[numeric_df['Cluster'] == 1].mean(numeric_only=True).round(3))

    # ────────────────────────────────────────────────
    # Visualize clusters in 2D (PCA reduction)
    # ────────────────────────────────────────────────
    print("\nCreating cluster visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1],
                    hue=numeric_df['Cluster'],
                    palette='viridis',
                    alpha=0.7,
                    s=60)
    plt.title('Automatic Clustering: Flows grouped into 2 groups (likely VPN vs non-VPN)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster (0 or 1)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(r"D:\vpn_clusters_plot.png")
    plt.show()

    # ────────────────────────────────────────────────
    # Save result with cluster labels
    # ────────────────────────────────────────────────
    output_path = r"D:\traffic_with_clusters.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved clustered results to: {output_path}")
    print("You can now open this file in Excel and see the 'Cluster' column (0 or 1)")
    print("Compare the two clusters — one should have more regular IAT, steady rates, etc. (VPN)")