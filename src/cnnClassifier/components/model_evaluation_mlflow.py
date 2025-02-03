from sklearn.metrics import classification_report
import mlflow
import mlflow.keras
import tensorflow as tf
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from cnnClassifier.entity.config_entity import EvaluationConfig
import os
os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/yashmarkad/Detection-of-Diabetic-Retinopathy-through-Deep-Learning.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="yashmarkad"
os.environ["MLFLOW_TRACKING_PASSWORD"]="507858fbefdda06e761fb4e268bab2342fa77d35"



class Evaluation:
    def __init__(self, config):
        self.config = config
    
    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.30
        )
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            shuffle=False
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        self.score = self.model.evaluate(self.valid_generator)
        self.compute_metrics()
        self.save_score()
    
    def compute_metrics(self):
        y_true = self.valid_generator.classes
        y_pred_probs = self.model.predict(self.valid_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        report = classification_report(y_true, y_pred, output_dict=True, target_names=self.valid_generator.class_indices.keys())
        
        self.precision = report['weighted avg']['precision']
        self.recall = report['weighted avg']['recall']
        self.f1_score = report['weighted avg']['f1-score']
    
    def save_score(self):
        scores = {
            "loss": self.score[0],
            "accuracy": self.score[1],
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score
        }
        save_json(path=Path("scores.json"), data=scores)
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics({
                "loss": self.score[0],
                "accuracy": self.score[1],
                "precision": self.precision,
                "recall": self.recall,
                "f1_score": self.f1_score
            })
            
            if tracking_url_type_store != "file":
                mlflow.keras.log_model(self.model, "model", registered_model_name="MobileNetV2")
            else:
                mlflow.keras.log_model(self.model, "model")


