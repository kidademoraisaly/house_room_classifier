# =====================================================
# 1. Imports
# =====================================================

import pytest
from room_classifier.data.preprocessing import prepare_data_generators # Responsible for preparing data generators (e.g., ImageDataGenerator) for training, validation, and testing datasets

# =====================================================
# 2. Test Function
# =====================================================

def test_data_generators():
    data_dir = 'data/processed'  # Ensure this path exists during testing

    # Step 1: Call the Function Under Test
    
    train_gen, val_gen, test_gen = prepare_data_generators(data_dir) # This function is expected to return: Training data, Validation data and Test data generators
    
    # Step 2: Check if Generators Are Created

    assert train_gen is not None, "Training generator should be created"
    assert val_gen is not None, "Validation generator should be created"
    assert test_gen is not None, "Test generator should be created"
    
    # Step 3: Check if Classes Are Detected

    assert len(train_gen.class_indices) > 0, "Should detect room classes"
