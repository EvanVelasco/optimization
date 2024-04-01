from PIL import Image
import numpy as np

# Open the image
image = Image.open("rommie.jpeg")

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the image to a numpy array
image_array = np.array(gray_image)

# Normalize pixel values to range from 0 to 1
normalized_image = image_array / 255.0

# Print the normalized matrix

np.savetxt("rommie.txt", normalized_image, fmt='%1.4f', delimiter='\t')
