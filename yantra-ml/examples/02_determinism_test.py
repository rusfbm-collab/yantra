#!/usr/bin/env python3
"""
Determinism Test - 10 Identical Runs

Запускает обучение 10 раз на одинаковых данных.
Результат: все предсказания идентичны!

Usage:
    python examples/02_determinism_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from yantra import AFSMClassifier
from yantra.datasets import generate_xor_dataset


def main():
    print("=" * 70)
    print("YANTRA: Determinism Test (10 Runs)")
    print("=" * 70)
    print()
    
    # Фиксированные данные для всех запусков
    print("📊 Generating fixed dataset...")
    X_train, y_train = generate_xor_dataset(n_samples=40, noise=0.05)
    X_test, y_test = generate_xor_dataset(n_samples=20, noise=0.05)
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    print()
    
    # 10 запусков
    print("🔄 Running 10 training sessions...")
    print()
    
    all_predictions = []
    all_accuracies = []
    
    for run in range(1, 11):
        # Создаем НОВЫЙ классификатор
        clf = AFSMClassifier(k_vec=(2, 4), activation='vortex_step', num_steps=1)
        
        # Обучаем на ТЕХ ЖЕ данных
        clf.train(X_train, y_train)
        
        # Получаем предсказания
        predictions = tuple(clf.predict(x) for x in X_test)
        accuracy = clf.evaluate(X_test, y_test)
        
        all_predictions.append(predictions)
        all_accuracies.append(accuracy)
        
        # Показываем первые 5 предсказаний
        pred_str = str(list(predictions[:5]))
        print(f"   Run {run:2d}: {pred_str} ... accuracy={accuracy:.1%}")
    
    print()
    
    # Анализ детерминизма
    print("📊 Determinism Analysis:")
    print()
    
    # Сколько уникальных результатов?
    unique_predictions = len(set(all_predictions))
    unique_accuracies = len(set(all_accuracies))
    
    print(f"   Unique prediction patterns: {unique_predictions}/10")
    print(f"   Unique accuracy values:     {unique_accuracies}/10")
    print()
    
    # Проверка
    if unique_predictions == 1 and unique_accuracies == 1:
        print("✅ DETERMINISM VERIFIED!")
        print()
        print("   🎯 All 10 runs produced IDENTICAL results")
        print("   🎯 No random seed needed")
        print("   🎯 100% reproducibility guaranteed")
        print()
        
        if all_accuracies[0] == 1.0:
            print("🎉 BONUS: 100% accuracy achieved!")
    else:
        print("❌ DETERMINISM FAILED!")
        print(f"   Expected: 1 unique result")
        print(f"   Got: {unique_predictions} unique results")
    
    print()
    print("=" * 70)
    print("Compare with PyTorch: different results on each run")
    print("YANTRA: same results every time!")
    print("=" * 70)


if __name__ == "__main__":
    main()
