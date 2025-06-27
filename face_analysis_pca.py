"""

Author: Jackson Wurzer  
Created: Feb 20, 2025  
Updated Last: June 20, 2025  
Contact: jacksonwurzer@outlook.com

Description:
------------
This script implements the Eigenfaces method for face recognition using Principal Component Analysis (PCA).
It performs dimensionality reduction using the economy-sized (reduced) Singular Value Decomposition (SVD) 
and demonstrates face reconstruction using a subset of eigenfaces, and finally a method for locating 
the most similar face to a chosen query face using the Euclidean distance in PCA (eigenface) space.  

Main Features:
--------------
- Computes the mean face from a dataset of vectorized face images
- Extracts principal components (eigenfaces) via SVD
- Visualizes the top eigenfaces and their negative variants
- Reconstructs a face using varying numbers of eigenfaces (5, 10, 15)
- Projects all face images onto the first two principal components for 2D visualization


Libraries:
-------------
- numpy
- scipy
- matplotlib

Usage:
------
This project uses face images from the Whitman College dataset. These are historical images of students. 
Only the numerical data representation is provided. Please handle the data responsibly.


Code is adapted from Professor Douglas Hundley's course materials at Whitman College 
and on his website: http://people.whitman.edu/~hundledr/courses/M350.html

"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Load in face data
mat_contents = scipy.io.loadmat('Faces.mat')
Y = mat_contents['Y']  # Each column is a vectorized face image
Y = np.array(Y, dtype=np.float64)  # Convert to float 
m = 294  # Image width
n = 262  # Image height
boys = mat_contents['boys']  # Indices for male faces
girls = mat_contents['girls']  # Indices for female faces

# Compute the mean face
avgFace = np.mean(Y, axis=1)  

# Display the mean face
plt.figure()
plt.imshow(avgFace.reshape((n, m)).T, cmap='gray')  # Reshape to original image dimensions
plt.title('Mean Face')
plt.axis('off')
plt.show()

# Mean center the data
Xm = Y - avgFace[:, np.newaxis]  

# Perform Singular Value Decomposition
U, S, VT = np.linalg.svd(Xm, full_matrices=0)  # U contains eigenfaces

# Display the first 4 eigenfaces
fig, axs = plt.subplots(2, 2)
axs = axs.ravel()
for i in range(4):
    axs[i].imshow(np.reshape(U[:, i], (n, m)).T, cmap='gray')  # Reshape to display
    axs[i].set_title(f'Eigenface {i+1}')
    axs[i].axis('off')
plt.show()

# Display the negative versions of the first 4 eigenfaces
fig, axs = plt.subplots(2, 2)
axs = axs.ravel()
for i in range(4):
    axs[i].imshow(np.reshape(-U[:, i], (n, m)).T, cmap='gray')  # Negative direction
    axs[i].set_title(f'Negative Eigenface {i+1}')
    axs[i].axis('off')
plt.show()

# Reconstruct a randomly selected face using different numbers of eigenfaces
np.random.seed(40)  # For reproducibility
random_index = np.random.randint(Y.shape[1])
original_face = Y[:, random_index]  # Select a face at random

k_values = [5, 10, 15]  # Number of eigenfaces used for reconstruction

fig, axs = plt.subplots(2, 2)
axs = axs.ravel()
axs[0].imshow(original_face.reshape((n, m)).T, cmap='gray')  # Original image
axs[0].set_title("Original")
axs[0].axis("off")

# Reconstruct the image with increasing number of principal components
i = 0
for k in k_values:
    U_k = U[:, :k]
    coeffs = U_k.T @ (original_face - avgFace)  # Project onto top-k eigenfaces
    recon_face = avgFace + U_k @ coeffs  # Reconstruct face from projection
    axs[i + 1].imshow(recon_face.reshape((n, m)).T, cmap='gray')  # Show reconstruction
    axs[i + 1].set_title(f"k = {k}")
    axs[i + 1].axis("off")
    i += 1
plt.show()


# --- Face Similarity Search using Euclidean Distance ---

# Choose a face at random to use as the query
np.random.seed(0)
query_index = np.random.randint(Y.shape[1])
query_face = Y[:, query_index]

# Number of principal components (eigenfaces) to use 
k_sim = 25 # Using 25 
U_k = U[:, :k_sim]

# Center the query face and project into PCA space
query_centered = query_face - avgFace
query_proj = U_k.T @ query_centered  # shape: (k_sim,)

# Project all centered faces into the same PCA space
dataset_proj = U_k.T @ Xm  # shape: (k_sim, num_faces)

# Compute Euclidean distances between query and all other faces
distances = np.linalg.norm(dataset_proj - query_proj[:, np.newaxis], axis=0)

# Exclude the query face itself
distances[query_index] = np.inf

# Find the index of the closest match
most_similar_index = np.argmin(distances)

# Plot original query face and most similar face side-by-side
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(query_face.reshape((n, m)).T, cmap='gray')
axs[0].set_title("Query Face")
axs[0].axis('off')

axs[1].imshow(Y[:, most_similar_index].reshape((n, m)).T, cmap='gray')
axs[1].set_title("Most Similar Face")
axs[1].axis('off')

plt.suptitle(f"Similarity Match using Top {k_sim} Eigenfaces", fontsize=14)
plt.show()

