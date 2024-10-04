# src/pca_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def run_pca(file_path):
    # Step 1: Load the dataset
    data = pd.read_csv(file_path)

    # Step 2: Encode the 'nurmcal' column
    label_encoder = LabelEncoder()
    data['nurmcal'] = label_encoder.fit_transform(data['nurmcal'])

    # Step 3: Define features and target variable
    X = data.drop('total_bill', axis=1)  # Features
    y = data['total_bill']  # Target variable

    # Step 4: Create a PCA Pipeline
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),  # Feature scaling
        ('pca', PCA(n_components=2))   # PCA for dimensionality reduction
    ])

    # Step 5: Fit the pipeline on the feature set
    X_pca = pipeline.fit_transform(X)

    # Step 6: Create a DataFrame for PCA results
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])

    # Step 7: Plotting the PCA results
    plt.figure(figsize=(10, 6))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], c=data['nurmcal'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='nurmcal (Encoded)')
    plt.title('PCA of Tips Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()

    # Step 8: Save the visualization
    plt.savefig('pca_tips_viz.png')  # Save the visualization as a PNG file
    plt.show()

if __name__ == "__main__":
    run_pca(r"data/tips - tips.csv")
