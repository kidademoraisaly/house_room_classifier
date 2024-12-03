import pytest
from room_classifier.data.preprocessing import prepare_data_generators

def test_data_generators():
    data_dir = 'data/processed'  # Ensure this path exists during testing
    
    train_gen, val_gen, test_gen = prepare_data_generators(data_dir)
    
    assert train_gen is not None, "Training generator should be created"
    assert val_gen is not None, "Validation generator should be created"
    assert test_gen is not None, "Test generator should be created"
    
    assert len(train_gen.class_indices) > 0, "Should detect room classes"
