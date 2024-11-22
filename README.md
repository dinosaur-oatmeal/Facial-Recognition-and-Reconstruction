# Face Recognition and Reconstruction with PCA

This project demonstrates the implementation of Principal Component Analysis (PCA) for face recognition and reconstruction, leveraging the Eigenfaces method. The program reduces the dimensionality of a dataset of facial images, achieving significant storage savings while maintaining high reconstruction fidelity.

## Features
- **Dimensionality Reduction:** Compresses facial images with up to **61% storage savings** using 150 principal components.
- **Eigenfaces Visualization:** Visualizes the most significant eigenfaces for interpretability.
- **Image Reconstruction:** Reconstructs original images with **high fidelity** (90%+ similarity).

## File Overview
`eigenfaces.py`: Main script implementing PCA and visualization.
`requirements.txt`: List of dependencies.
`ATTfaces/faces/`: Directory for input facial images.

## Metrics and Results
Storage Savings: Achieved 61% reduction in data storage with 150 principal components.
Reconstruction Fidelity: Retained over 90% visual similarity to original images.

## Tools and Libraries
* Python
* NumPy
* Matplotlib
* scikit-learn
* scikit-image
* PIL