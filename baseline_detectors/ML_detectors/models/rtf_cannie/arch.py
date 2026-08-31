# ============================================================================
# RTF-CANNIE ARCHITECTURE
# This model uses Google's CANINE transformer (character-level processing)
# Pre-trained model is loaded from HuggingFace transformers library
# ============================================================================

"""
RTF-CANNIE Architecture

This module doesn't define custom neural network classes because it uses
Google's pre-trained CANINE model directly from the transformers library.

The architecture consists of:
1. CANINE-S base model (google/canine-s)
   - Character-level transformer (no tokenization needed)
   - Processes up to 2048 characters
   - Pre-trained on multilingual data

2. Random Transformer Forest (RTF) Ensemble
   - 5 independent CANINE models
   - Each trained on bootstrap samples
   - Predictions aggregated by averaging probabilities

Key Papers:
- CANINE: "CANINE: Pre-training an Efficient Tokenization-Free Encoder
           for Language Representation" (Clark et al., 2021)
- RTF: "An Ensemble of Pre-trained Transformer Models For Imbalanced
        Multiclass Malware Classification" (Demirkıran et al.)

Usage:
    from transformers import CanineForSequenceClassification

    model = CanineForSequenceClassification.from_pretrained(
        'google/canine-s',
        num_labels=6,
        ignore_mismatched_sizes=True
    )

Model Architecture (CANINE-S):
    - Input: Character sequences (0-2048 chars)
    - Embedding: Character embeddings
    - Encoder: 12 transformer layers
    - Pooler: [CLS] token pooling
    - Classifier: Linear layer to num_classes

Ensemble Strategy:
    - Train 5 models with stratified bootstrap sampling
    - Each model votes on predictions
    - Final prediction = argmax(average_probabilities)

Hyperparameters (from paper):
    - max_length: 2048 characters
    - learning_rate: 3e-5
    - weight_decay: 1e-3
    - batch_size: 8
    - num_epochs: 5
    - warmup_ratio: 0.1
"""

# No custom architecture needed - uses HuggingFace transformers directly
# The actual model is instantiated in __init__.py using:
#
#   from transformers import CanineForSequenceClassification
#
#   model = CanineForSequenceClassification.from_pretrained(
#       'google/canine-s',
#       num_labels=num_classes,
#       ignore_mismatched_sizes=True
#   )

__all__ = []  # No classes exported from this module
