#!/usr/bin/env python3
"""
XOR Classification Demo - 100% Deterministic

Демонстрация детерминированной нейросети на задаче XOR.
Запустите несколько раз - результат будет идентичен!

Usage:
    python examples/01_xor_demo.py
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yantra import AFSMClassifier
from yantra.datasets import generate_xor_dataset


def main():
    print("=" * 70)
    print("YANTRA: XOR Classification Demo (100% Deterministic)")
    print("=" * 70)
    print()
    
    # Генерация данных XOR
    print("📊 Generating XOR dataset...")
    X_train, y_train = generate_xor_dataset(n_samples=40, noise=0.05)
    X_test, y_test = generate_xor_dataset(n_samples=20, noise=0.05)
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print()
    
    # Создание классификатора
    print("🔧 Creating AFSM classifier...")
    clf = AFSMClassifier(
        k_vec=(2, 4),           # Конфигурация локусов
        activation='vortex_step',  # Алгебраическая активация
        num_steps=1             # Количество шагов
    )
    print(f"   Configuration: k_vec={clf.k_vec}")
    print(f"   Activation: {clf.activation}")
    print()
    
    # Обучение (exhaustive search)
    print("🎯 Training (exhaustive search)...")
    training_info = clf.train(X_train, y_train)
    print(f"   Candidates evaluated: {training_info['candidates_evaluated']}")
    print(f"   Method: Exhaustive search (deterministic)")
    print(f"   Best accuracy: {training_info['best_accuracy']:.1%}")
    print()
    
    # Тестирование
    print("✅ Testing...")
    test_accuracy = clf.evaluate(X_test, y_test)
    print(f"   Test accuracy: {test_accuracy:.1%}")
    print()
    
    # Предсказания
    print("🔍 Sample predictions:")
    for i in range(min(5, len(X_test))):
        x = X_test[i]
        y_true = y_test[i]
        y_pred = clf.predict(x)
        status = "✓" if y_pred == y_true else "✗"
        print(f"   {status} Input: {x} → Predicted: {y_pred}, True: {y_true}")
    print()
    
    # Confusion matrix
    print("📊 Confusion Matrix:")
    predictions = [clf.predict(x) for x in X_test]
    cm = compute_confusion_matrix(predictions, y_test, num_classes=2)
    print(f"   [[{cm[0][0]:2d}, {cm[0][1]:2d}]")
    print(f"    [{cm[1][0]:2d}, {cm[1][1]:2d}]]")
    print()
    
    # Детерминизм
    if test_accuracy == 1.0:
        print("🎉 PERFECT! 100% accuracy on XOR!")
        print()
        print("💡 Key point: Run this script multiple times.")
        print("   The result will be IDENTICAL - no random seed needed!")
    else:
        print(f"✓ Accuracy: {test_accuracy:.1%}")
    
    print()
    print("=" * 70)
    print("Try running this script 10 times - results are always identical!")
    print("=" * 70)


def compute_confusion_matrix(predictions, labels, num_classes):
    """Вычисление confusion matrix."""
    cm = [[0] * num_classes for _ in range(num_classes)]
    
    for pred, true in zip(predictions, labels):
        cm[true][pred] += 1
    
    return cm


if __name__ == "__main__":
    main()
