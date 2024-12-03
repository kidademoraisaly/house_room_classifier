import pytest
import tensorflow as tf
from room_classifier.models.cnn_model import RoomClassificationModel

def test_model_creation():
    model = RoomClassificationModel(num_classes=6)
    model.build_model()
    
    assert model.model is not None, "Model should be created successfully"
    assert len(model.model.layers) > 0, "Model should have layers"

def test_model_compilation():
    model = RoomClassificationModel(num_classes=6)
    model.build_model()
    
    assert model.model.optimizer is not None, "Model should have an optimizer"
    assert model.model.loss is not None, "Model should have a loss function"
