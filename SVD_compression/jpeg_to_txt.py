from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Open the image
image = Image.open("images/tech.jpeg")

# Convert the image to grayscale
gray_image = image.convert('L')

# Convert the image to a numpy array
image_array = np.array(gray_image)

# Normalize pixel values to range from 0 to 1
normalized_image = image_array / 255.0

# Print the normalized matrix

np.savetxt("tech.txt", normalized_image, fmt='%1.4f', delimiter='\t')


X = np.loadtxt("tech.txt")

plt.gray()
plt.imshow(X)
plt.title(f'Original Image')
plt.savefig(f'images/original.jpeg')
plt.show()