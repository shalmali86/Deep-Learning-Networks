import numpy as np  # Importing numpy for numerical operations
import matplotlib.pyplot as plt  # Importing matplotlib for plotting images
import cv2  # Importing OpenCV for image processing
from tensorflow.keras.models import Sequential  # Importing Sequential API for building a model layer-by-layer
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D  # Importing layers for CNN model

# Reading an image in grayscale mode
image = cv2.imread('/content/dot.png', cv2.IMREAD_GRAYSCALE)

# Checking the shape of the original image
image.shape

# Resizing the image to 28x28 pixels
image_resize = cv2.resize(image, (28, 28))

# Reshaping the image to (1, 28, 28, 1) to be compatible with the Conv2D input
image_input = image_resize.reshape(1, 28, 28, 1)

# Defining a simple 3x3 filter for edge detection
filter1 = np.array([[[[-1]], [[-1]], [[-1]]],
                    [[[-1]], [[8]], [[-1]]],
                    [[[-1]], [[-1]], [[-1]]]])

# Building a Sequential model
model = Sequential()
# Adding a Conv2D layer with the custom filter and no activation (just applying convolution)
model.add(Conv2D(1, kernel_size=(3, 3), input_shape=(28, 28, 1), activation=None, use_bias=False))
# Setting the weights of the Conv2D layer to the custom filter defined above
model.layers[0].set_weights([filter1])

# Applying the convolutional filter to the input image
conv_output = model.predict(image_input)

# Removing single-dimensional entries from the output array (from (1, 26, 26, 1) to (26, 26))
conv_output = np.squeeze(conv_output)

# Plotting the original resized image and the filtered image
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_resize, cmap='gray')  # Showing the resized input image
ax[0].axis('off')
ax[1].imshow(conv_output, cmap='gray')  # Showing the convolutional output image
ax[1].axis('off')
plt.show()


# Loading necessary libraries again for the second part
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Importing functions to load and convert images

# Loading a grayscale image, resizing it to 128x128 pixels
img = load_img('/content/African_Bush_Elephant.jpg', target_size=(128, 128), color_mode='grayscale')
# Converting the image to a NumPy array for model compatibility
img_array = img_to_array(img)
# Expanding dimensions to make it suitable for model input (from (128,128,1) to (1,128,128,1))
img_array = np.expand_dims(img_array, axis=0)

# Displaying the image array shape
img_array

# Normalizing the image array values to a range of 0-1 by dividing by 255
image_array = img_array / 255.0

# Building a Sequential model with a single Conv2D layer
model = Sequential()
# Adding a Conv2D layer with 32 filters, 3x3 kernel, ReLU activation, and same padding
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1), padding='same'))
# Compiling the model with Adam optimizer and categorical crossentropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Generating the feature maps by passing the image through the Conv2D layer
conv_output = model.predict(image_array)
# Squeezing the output to remove unnecessary dimensions
features_maps = np.squeeze(conv_output)

# Plotting each of the 32 feature maps
fig, axarr = plt.subplots(4, 8, figsize=(15, 10))  # Creating a grid of subplots
for i in range(32):  # Looping through each feature map
  ax = axarr[i // 8, i % 8]  # Selecting subplot based on index
  ax.imshow(features_maps[:, :, i], cmap='gray')  # Displaying each feature map in grayscale
  ax.axis('off')  # Hiding axes for cleaner visualization
plt.show()  # Displaying the feature maps
