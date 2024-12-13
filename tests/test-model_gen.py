# =====================================================
# 1. Imports
# =====================================================

import pytest
import tensorflow as tf
from room_classifier.models.cnn_model import RoomClassificationModel

# =====================================================
# 2. Testing Model Creation
# =====================================================

def test_model_creation():
    model = RoomClassificationModel(num_classes=5) # Initialize the model with 6 classes
    model.build_model() # Build the model architecture
    
    assert model.model is not None, "Model should be created successfully" # Ensures that the model object (model.model) is successfully created and is not None
    assert len(model.model.layers) > 0, "Model should have layers"         # Verifies that the model contains at least one layer

# =====================================================
# 3. Testing Model Compilation (Optimizer + Loss Function)
# =====================================================

def test_model_compilation():
    model = RoomClassificationModel(num_classes=6) # Initialize the model with 6 classes
    model.build_model() # Build the model architecture
    
    assert model.model.optimizer is not None, "Model should have an optimizer" # Ensures that the model has been compiled with an optimizer (e.g., Adam, RMSprop)
    assert model.model.loss is not None, "Model should have a loss function"   # Verifies that a loss function is set during compilation
