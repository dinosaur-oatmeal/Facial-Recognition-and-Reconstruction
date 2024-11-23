import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image

# ***Compute Principal Component Analysis function***
def compute_pca(data, data_mean, n_comps):

    imageAmount, height, width = data.shape

    # ***1. Subtract the mean image from all images***
    dataMinusMean = data - data_mean

    # ***2. Vectorize the images to a 2D matrix***
    print("reshaping")
    dataReshape = dataMinusMean.reshape(imageAmount, height * width)
    print("Reshaped Data Dimensions:", dataReshape.shape)

    # ***3. Create the covariance matrix***
    matrixCovariance = np.cov(dataReshape, rowvar=True)
    print("Covariance Dimensions:", matrixCovariance.shape)
    
    # ***4. Compute the eigendecomposition***
    print("finding eigenvalues and eigenvectors")
    eigenvalues, eigenvectors = np.linalg.eig(matrixCovariance)

    # ***5. Calculate the eigenfaces***
    print("sorting data")
    dataSorted = np.argsort(-eigenvalues)[:n_comps]
    eigenvalues = eigenvalues[dataSorted]
    eigenvectors = eigenvectors[:, dataSorted]
    print("Eigenvectors Dimensions:", eigenvectors.shape)

    print("finding eigenfaces")
    eigenfaces = dataReshape.T @ eigenvectors
    print("Eigenfaces Dimensions:", eigenfaces.shape)
    
    # ***6. Normalize eigenfaces***
    print("normalizing eigenfaces")
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)
    
    # ***7. Compute the weights***
    weights = dataReshape @ eigenfaces
    print("Weights Dimensions", weights.shape)
    
    # Compute sizes in bytes
    original_size = dataReshape.nbytes
    compressed_size = eigenfaces[:, :n_comps].nbytes + weights.nbytes
    size_reduction = (1 - compressed_size / original_size) * 100
    print(f"Original Data Size: {original_size / 1e6:.2f} MB")
    print(f"Compressed Data Size: {compressed_size / 1e6:.2f} MB")
    print(f"Size Reduction: {size_reduction:.2f}%")
    
    # ***8. Return the normalized eigenfaces and weights***
    return eigenfaces.T, weights

# ***Reconstruct face function***
def reconstruct(weights, eigenfaces, X_mean, img_size, img_idx, n_comps):

    # Find weights for the image and number of eigenfaces
    imageWeights = weights[img_idx, :n_comps]
    print("Image Weights Dimensions", imageWeights.size)

    # Find the eigenfaces selected for reconstruction
    eigenfacesSelected = eigenfaces[:n_comps, :]
    print("Eigenfaces Selected Dimensions", eigenfacesSelected.size)

    # Reconstruct the image
    reconstruct = imageWeights @ eigenfacesSelected
    print("Reconstructed Dimensions", reconstruct.size);

    # Add mean back to the image!
    reconstruct += X_mean

    # Reshape back into 2D
    recovered_img = reconstruct.reshape(img_size)
    print("Recovered Image Dimensions", recovered_img.size)
    print("image dimensions should be 92 * 112 = 10304")

    return recovered_img

def main():

    # Load raw images
    data_dir = "ATTfaces/faces/"
    file_names = os.listdir(data_dir)
    images = [np.asarray(Image.open(data_dir + file_names[i])) for i in range(len(file_names))]
    images = np.array(images)

    # Visualize some of the faces
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Sample Faces from the Dataset", fontsize=16)
    grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0)

    for ax, im in zip(grid, images[:64]):
        ax.imshow(im)
        ax.axis('off')

    plt.show()

    # Calculate the mean face
    face_mean = images.mean(0)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(face_mean)
    ax.set_title("Average Face from Dataset")

    plt.show()

    num_images, height, width = images.shape
    img_idx = 1
    n_comps = 200

    # Compute PCA and visualize
    eigenfaces, weights = compute_pca(images, face_mean, n_comps)

    # Visualize some eigenfaces
    fig = plt.figure(figsize=(4, 4))
    fig.suptitle("Top 16 Eigenfaces", fontsize=16)
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0)

    for ax, im in zip(grid, eigenfaces[:16]):
        ax.imshow(im.reshape(height, width))
        ax.axis('off')

    plt.show()

    reconstructed_img = reconstruct(weights, eigenfaces, face_mean.reshape(-1), [height, width], img_idx, n_comps)

    # Compute similarities between original and reconstructed image
    correlation = np.corrcoef(face_mean, reconstructed_img)[0, 1]
    print(f"Correlation Coefficient: {correlation:.4f}")

    # Visualize original and reconstructed
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(images[img_idx])
    ax2.imshow(reconstructed_img)
    ax1.set_title("Original Face")
    ax2.set_title("Reconstructed Face")

    plt.show()

if __name__ == "__main__":

    main()