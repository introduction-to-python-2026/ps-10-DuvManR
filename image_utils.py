from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(file_path):
  """
  Loads a color image from the given file path and converts it into a NumPy array.

  Args:
    file_path (str): The path to the image file.

  Returns:
    np.array: The image as a NumPy array.
  """
  try:
    img = Image.open(file_path)
    img_array = np.array(img)
    return img_array
  except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found. Please upload an image file to /content/ and update the path.")
    return None
  except Exception as e:
    print(f"An error occurred while loading the image: {e}")
    return None

def edge_detection(image_array):
  """
  Performs edge detection on a 3-channel color image array.

  Args:
    image_array (np.array): The input color image as a NumPy array.

  Returns:
    np.array: The magnitude of the edges (edgeMAG) as a NumPy array.
  """
  # 1. Convert to grayscale
  # Average the three color channels. Ensure it's converted to float for calculations.
  grayscale_image = np.mean(image_array, axis=2).astype(float)

  # 2. Create kernelY for vertical changes
  kernelY = np.array([
      [1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1]
  ])

  # 3. Create kernelX for horizontal changes
  kernelX = np.array([
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]
  ])

  # 4. Apply each filter using convolve2d with zero padding
  # 'mode='same'' ensures output has the same shape as input
  # 'boundary='fill'', fillvalue=0.0 provides zero padding
  edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0.0)
  edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0.0)

  # 5. Compute edgeMAG
  edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

  return edgeMAG
