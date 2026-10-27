
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from tensorflow.keras.utils import load_img, img_to_array

# 1. Load the trained CNN model
model = tf.keras.models.load_model(
    "models/best_mask_model.keras"
)

# 2. Load class names in the correct order
with open("models/class_names.json", "r") as file:
    class_names = json.load(file)

# 3. Select the new image
image_path = Path("test_images/person.jpg")

if not image_path.exists():
    raise FileNotFoundError(
        f"Image not found: {image_path.resolve()}"
    )

# 4. Load and resize image to 128 x 128
image = load_img(
    image_path,
    target_size=(128, 128),
    color_mode="rgb"
)

# 5. Convert image into a NumPy array
image_array = img_to_array(image)

# 6. Normalize pixel values, just like training
image_array = image_array / 255.0

# 7. Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# 8. Predict the class
predictions = model.predict(image_array, verbose=0)

predicted_index = np.argmax(predictions[0])
print("All probabilities:", predictions[0])
predicted_class = class_names[predicted_index]
confidence = float(predictions[0][predicted_index]) * 100

# 9. Display the result
print("Predicted class:", predicted_class)
print(f"Confidence: {confidence:.2f}%")

# 10. Display the image and prediction
plt.imshow(image)
plt.title(
    f"{predicted_class} ({confidence:.2f}%)"
)
plt.axis("off")
plt.show()
