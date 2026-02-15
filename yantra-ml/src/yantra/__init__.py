"""
YANTRA: Deterministic Neural Network on Finite Groups

Детерминированная нейронная сеть на циклических группах Zₙ.
100% воспроизводимость, формальная верификация, exhaustive search.

Example:
    >>> from yantra import AFSMClassifier
    >>> from yantra.datasets import generate_xor_dataset
    >>> 
    >>> X_train, y_train = generate_xor_dataset(n_samples=40)
    >>> clf = AFSMClassifier(k_vec=(2, 4))
    >>> clf.train(X_train, y_train)
    >>> 
    >>> X_test, y_test = generate_xor_dataset(n_samples=20)
    >>> accuracy = clf.evaluate(X_test, y_test)
    >>> print(f"Accuracy: {accuracy:.1%}")
"""

__version__ = "0.1.0"
__author__ = "YANTRA Team"
__license__ = "MIT"

from .classifier import AFSMClassifier, AFSMConfig
from .datasets import generate_xor_dataset, generate_two_blobs_dataset

__all__ = [
    'AFSMClassifier',
    'AFSMConfig',
    'generate_xor_dataset',
    'generate_two_blobs_dataset',
    '__version__',
]
