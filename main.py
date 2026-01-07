import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.filters import median
from skimage.morphology import ball, disk # Import disk for 2D footprint

# 1. Import the load_image and edge_detection functions from the image_utils module.
from image_utils import load_image, edge_detection

# 2. Define the image_path
image_path = '/content/bird.jpg'

# 3. Load the image
original_image = load_image(image_path)

if original_image is not None:
    print(f"Original image loaded successfully from {image_path}")

    # 4. Apply a median filter to the loaded image for noise suppression.
    # Apply median filter to each channel if it's a color image
    clean_image = np.zeros_like(original_image, dtype=np.uint8)
    for i in range(original_image.shape[2]): # Iterate over color channels
        # Use disk(3) for 2D filtering on each channel
        clean_image[:, :, i] = median(original_image[:, :, i], disk(3))
    print("Median filter applied.")

    # 5. Pass the median-filtered image to your imported edge_detection function
    edge_magnitude_image = edge_detection(clean_image)
    print("Edge detection performed.")

    # 6. To determine a suitable threshold for binarization, display a histogram of the edgeMAG values.
    plt.figure(figsize=(8, 5))
    plt.hist(edge_magnitude_image.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Histogram of Edge Magnitude Values')
    plt.xlabel('Edge Magnitude')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    print("Histogram displayed. Please examine it to choose a threshold.")

    # Based on observation from the histogram (e.g., from the previous run's `edge_magnitude_image` values),
    # let's pick a threshold. This value might need adjustment after seeing the histogram.
    # For this example, I'll use 50 as a reasonable starting point based on typical edge magnitudes.
    threshold = 50 # Example threshold, adjust after viewing histogram
    print(f"Using threshold: {threshold}")

    # 7. Convert the edgeMAG array into a binary array
    edge_binary = (edge_magnitude_image > threshold).astype(np.uint8) * 255 # Convert to 0 or 255 for saving
    print("Edge image binarized.")

    # 8. Create a figure with subplots to display the original image,
    # the median-filtered image, the edgeMAG image (grayscale), and the binary edge image.
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(clean_image)
    axes[1].set_title('Median Filtered Image')
    axes[1].axis('off')

    axes[2].imshow(edge_magnitude_image, cmap='gray')
    axes[2].set_title('Edge Magnitude')
    axes[2].axis('off')

    axes[3].imshow(edge_binary, cmap='gray')
    axes[3].set_title('Binary Edges (Threshold=' + str(threshold) + ')')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()
    print("Images displayed.")

    # 9. Save the edge_binary array as a .png file
    edge_image_pil = Image.fromarray(edge_binary)
    edge_image_pil.save('my_edges.png')
    print("Binary edge image saved as 'my_edges.png'.")
else:
    print("Image loading failed. Cannot proceed with processing.")
