import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
    def get_base_model(self):
        self.model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)
    @staticmethod
    def _prepare_full_model(model, classes):
    # Add Global Average Pooling layer
        gap_layer = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    
    # Add the prediction layer
        prediction = tf.keras.layers.Dense(
        units=classes,
        activation="softmax"
        )(gap_layer)
    
    # Create the full model
        full_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=prediction
    )
    
    # Compile the full model
        full_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    
    # Display the model summary
        full_model.summary()
        return full_model

    
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)

    
        
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

