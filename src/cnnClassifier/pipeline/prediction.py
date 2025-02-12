import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.verbose_name = {
            0: 'No_DR',
            1: 'Mild',
            2: 'Moderate',
            3: 'Severe',
            4: 'Proliferate_DR'
        }
        self.model = load_model(os.path.join("model", "model.h5"))

    def predict_label(self):
        # Load and preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255.0
        test_image = test_image.reshape(1, 224, 224, 3)

        # Get prediction probabilities
        predictions = self.model.predict(test_image)
        
        # Get class and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = np.max(predictions) * 100  # Convert to percentage
        
        # Format output string
        return (
            f"{self.verbose_name[predicted_class_idx]} "
            f"with confidence of {confidence:.2f}%"
        )