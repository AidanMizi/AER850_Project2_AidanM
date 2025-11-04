from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# load the model
model_path = 'DCNN_model.h5'  
DCNN_model = load_model(model_path)

test_image_paths = [
    r"Data\test\crack\test_crack.jpg",
    r"Data\test\missing-head\test_missinghead.jpg",
    r"Data\test\paint-off\test_paintoff.jpg"
]

class_names = ['crack', 'missing-head', 'paint-off']  

# for loop to iterate through each test image and display the results
for img_path in test_image_paths:
    img = image.load_img(img_path, target_size=(500, 500))
    img_array = image.img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0) 

    # predict the class
    predictions = DCNN_model.predict(img_array)
    predicted_class = np.argmax(predictions)  # find the class with highest probability
    confidence = predictions[0][predicted_class]

    # display result
    print(f"Image: {img_path}")
    print(f"Predicted classification: {class_names[predicted_class]} with confidence {confidence:.2f}")

    # display the image with the prediction
    plt.imshow(img)
    plt.title(f"Predicted classification: {class_names[predicted_class]} ({confidence:.2f})")
    plt.axis('off')
    plt.show()