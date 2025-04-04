from keras import load_model
from keras import image
import numpy as np

IMG_SIZE = 224

# Load the trained model
model = load_model('cat_dog_model.h5')

# Load and preprocess the image
img_path = 'path_to_your_image.jpg'
img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale like training

# Predict
prediction = model.predict(img_array)

# Output
if prediction[0] > 0.5:
    print("It's a dog!")
else:
    print("It's a cat!")