"""
Dataset Generation для YANTRA

Генерация синтетических датасетов для тестирования.
"""

from typing import List, Tuple
import random


def generate_xor_dataset(
    n_samples: int = 40,
    noise: float = 0.05
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Генерация XOR датасета.
    
    XOR функция:
        (0, 0) → 0
        (0, 1) → 1
        (1, 0) → 1
        (1, 1) → 0
    
    Args:
        n_samples: Количество примеров
        noise: Уровень шума (0.0 - 0.5)
    
    Returns:
        (X, y) где X - входы, y - метки
    
    Example:
        >>> X, y = generate_xor_dataset(n_samples=100)
        >>> print(len(X), len(y))
        100 100
    """
    X = []
    y = []
    
    # Генерируем примеры из всех 4 квадрантов
    for _ in range(n_samples):
        # Случайный квадрант
        quadrant = random.randint(0, 3)
        
        if quadrant == 0:  # (0, 0) → 0
            x1 = random.uniform(0, 0.5 - noise)
            x2 = random.uniform(0, 0.5 - noise)
            label = 0
        elif quadrant == 1:  # (0, 1) → 1
            x1 = random.uniform(0, 0.5 - noise)
            x2 = random.uniform(0.5 + noise, 1.0)
            label = 1
        elif quadrant == 2:  # (1, 0) → 1
            x1 = random.uniform(0.5 + noise, 1.0)
            x2 = random.uniform(0, 0.5 - noise)
            label = 1
        else:  # (1, 1) → 0
            x1 = random.uniform(0.5 + noise, 1.0)
            x2 = random.uniform(0.5 + noise, 1.0)
            label = 0
        
        X.append((x1, x2))
        y.append(label)
    
    return X, y


def generate_two_blobs_dataset(
    n_samples: int = 40
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """Генерация Two Blobs датасета (линейно разделимый).
    
    Два кластера:
        Blob 1: центр (0.25, 0.25), label=0
        Blob 2: центр (0.75, 0.75), label=1
    
    Args:
        n_samples: Количество примеров
    
    Returns:
        (X, y) где X - входы, y - метки
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Выбираем кластер
        cluster = random.randint(0, 1)
        
        if cluster == 0:
            # Кластер 1: около (0.25, 0.25)
            x1 = random.gauss(0.25, 0.1)
            x2 = random.gauss(0.25, 0.1)
            label = 0
        else:
            # Кластер 2: около (0.75, 0.75)
            x1 = random.gauss(0.75, 0.1)
            x2 = random.gauss(0.75, 0.1)
            label = 1
        
        # Обрезаем значения в [0, 1]
        x1 = max(0.0, min(1.0, x1))
        x2 = max(0.0, min(1.0, x2))
        
        X.append((x1, x2))
        y.append(label)
    
    return X, y


__all__ = ['generate_xor_dataset', 'generate_two_blobs_dataset']
