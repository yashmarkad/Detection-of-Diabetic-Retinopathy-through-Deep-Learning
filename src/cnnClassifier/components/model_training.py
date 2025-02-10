import os
from pathlib import Path
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import TrainingConfig
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
        )

        train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
            **datagenerator_kwargs,
             
        )

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            class_mode='categorical',
            **dataflow_kwargs
        )

        
    def test_valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
        )

        test_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Optionally get this from config
        testdir = r"C:/CDAC_PROJECT/Final_project_retinopathy/artifacts/data_ingestion/testing"

        self.test_generator = test_datagenerator.flow_from_directory(
            directory=testdir,
            shuffle=True,
            class_mode='categorical',
            **dataflow_kwargs
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        # # Ensure the generators are initialized
        # if not hasattr(self, 'train_generator'):
        #     self.train_valid_generator()
        # if not hasattr(self, 'test_generator'):
        #     self.test_valid_generator()

        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            validation_data=self.test_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
