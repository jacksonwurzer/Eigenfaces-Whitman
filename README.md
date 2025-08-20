# Face Analysis Using Eigenfaces and PCA

*Important Note: This project uses face images from the Whitman College dataset. These are historical images of students. The raw   photos are **not included** in the repository to protect privacy. Only the numerical data representation is provided. Please handle the data responsibly.

# Project Overview

This project uses Principal Component Analysis (PCA) to extract eigenfaces from a dataset of vectorized face images. The goal of this project was to explore how faces can be represented in a lower-dimensional space by projecting them onto the principal components. I then was interested in how to use the principal components (eigenfaces) to reconstruct a given face. I also used the first 25 principal components to select a face and then locate the most similar face using the Euclidean distance in PCA space. 



# Main Features

- Dataset: Vectorized grayscale face images (Faces.mat dataset)

- Dimensionality Reduction: PCA via the SVD

- Eigenfaces: Principal components representing dominant patterns of facial variation

- Face Reconstruction: Rebuild faces using top k eigenfaces (k = 5, 10, 15)

- Similarity Search: Identify most similar faces using Euclidean distance in PCA space

- Visualization: Display mean face, eigenfaces, reconstructed faces, and similar faces.


  ## Files
  - Faces.mat -- faces data stored in MATLAB file
  - 
