"""
YANTRA AFSM Classifier - Минимальная реализация

Детерминированный классификатор на конечных группах.
"""

from __future__ import annotations
from typing import List, Tuple, Any, Dict
from dataclasses import dataclass


@dataclass
class AFSMConfig:
    """Конфигурация AFSM нейрона."""
    k_vec: Tuple[int, ...]  # Вектор локусов
    activation: str = 'vortex_step'
    num_steps: int = 1


class AFSMClassifier:
    """Детерминированный классификатор на AFSM нейронах.
    
    Использует exhaustive search вместо градиентного спуска.
    Гарантирует 100% воспроизводимость результатов.
    
    Example:
        >>> clf = AFSMClassifier(k_vec=(2, 4))
        >>> clf.train(X_train, y_train)
        >>> accuracy = clf.evaluate(X_test, y_test)
        >>> print(f"Accuracy: {accuracy:.1%}")
    """
    
    def __init__(
        self,
        k_vec: Tuple[int, ...] = (2, 4),
        activation: str = 'vortex_step',
        num_steps: int = 1
    ):
        """
        Args:
            k_vec: Конфигурация локусов (размеры циклических групп)
            activation: Тип активации ('vortex_step' или 'identity')
            num_steps: Количество шагов AFSM
        """
        self.k_vec = k_vec
        self.activation = activation
        self.num_steps = num_steps
        self.params = None  # Обученные параметры
    
    def train(
        self,
        X: List[Tuple[float, float]],
        y: List[int]
    ) -> Dict[str, Any]:
        """Обучение через exhaustive search.
        
        Args:
            X: Обучающие примеры [(x1, x2), ...]
            y: Метки классов [0, 1, ...]
        
        Returns:
            Информация об обучении
        """
        # Дискретизация входов в Zₙ
        X_discrete = self._discretize(X)
        
        # Exhaustive search в пространстве кандидатов
        candidates = self._generate_candidates()
        
        best_accuracy = 0.0
        best_params = None
        
        for params in candidates:
            # Оценка кандидата
            accuracy = self._evaluate_params(params, X_discrete, y)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
        
        # Сохраняем лучшие параметры
        self.params = best_params
        
        return {
            'candidates_evaluated': len(candidates),
            'best_accuracy': best_accuracy,
            'method': 'exhaustive_search'
        }
    
    def predict(self, x: Tuple[float, float]) -> int:
        """Предсказание класса для одного примера.
        
        Args:
            x: Входной пример (x1, x2)
        
        Returns:
            Предсказанный класс (0 или 1)
        """
        if self.params is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Дискретизация
        x_discrete = self._discretize_single(x)
        
        # Применяем параметры
        if self.activation == 'vortex_step':
            # Vortex step activation
            state = self._apply_vortex(x_discrete, self.params)
        else:
            # Identity (линейный случай)
            state = x_discrete
        
        # Извлекаем класс из состояния
        class_pred = self._state_to_class(state)
        
        return class_pred
    
    def evaluate(
        self,
        X: List[Tuple[float, float]],
        y: List[int]
    ) -> float:
        """Оценка accuracy на тестовых данных.
        
        Args:
            X: Тестовые примеры
            y: Истинные метки
        
        Returns:
            Accuracy (0.0 - 1.0)
        """
        predictions = [self.predict(x) for x in X]
        correct = sum(1 for pred, true in zip(predictions, y) if pred == true)
        return correct / len(y)
    
    def _discretize(self, X: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Дискретизация непрерывных входов в Zₙ."""
        n = self.k_vec[0]  # Размер первого локуса
        
        X_discrete = []
        for x1, x2 in X:
            # Простая дискретизация: порог 0.5
            d1 = 0 if x1 < 0.5 else 1
            d2 = 0 if x2 < 0.5 else 1
            X_discrete.append((d1, d2))
        
        return X_discrete
    
    def _discretize_single(self, x: Tuple[float, float]) -> Tuple[int, int]:
        """Дискретизация одного примера."""
        x1, x2 = x
        d1 = 0 if x1 < 0.5 else 1
        d2 = 0 if x2 < 0.5 else 1
        return (d1, d2)
    
    def _generate_candidates(self) -> List[Dict[str, Any]]:
        """Генерация кандидатов для exhaustive search.
        
        Для XOR задачи: 2^6 = 64 кандидата (малое пространство поиска)
        """
        candidates = []
        
        # Все возможные комбинации выходов для 4 входов
        # Вход: (d1, d2) ∈ {0,1}² → 4 возможных состояния
        # Выход: class ∈ {0, 1}
        
        # Перебираем все функции {0,1}² → {0,1}
        # Всего 2^4 = 16 функций
        
        for output_pattern in range(16):
            # Декодируем паттерн в таблицу
            table = {}
            for i in range(4):
                # i-й вход: (i//2, i%2)
                inp = (i // 2, i % 2)
                # i-й бит паттерна
                out = (output_pattern >> i) & 1
                table[inp] = out
            
            candidates.append({'table': table})
        
        # Для vortex_step активации: дополнительные кандидаты
        if self.activation == 'vortex_step':
            # Добавляем трансформированные версии
            for base_pattern in range(16):
                for transform in range(4):  # 4 типа трансформаций
                    table = {}
                    for i in range(4):
                        inp = (i // 2, i % 2)
                        out = (base_pattern >> i) & 1
                        # Применяем трансформацию
                        if transform == 1:
                            out = 1 - out  # Инверсия
                        elif transform == 2:
                            out = (out + inp[0]) % 2  # XOR с первым битом
                        elif transform == 3:
                            out = (out + inp[1]) % 2  # XOR со вторым битом
                        table[inp] = out
                    
                    candidates.append({'table': table, 'transform': transform})
        
        return candidates
    
    def _evaluate_params(
        self,
        params: Dict[str, Any],
        X: List[Tuple[int, int]],
        y: List[int]
    ) -> float:
        """Оценка точности для заданных параметров."""
        table = params['table']
        
        correct = 0
        for x, y_true in zip(X, y):
            y_pred = table.get(x, 0)
            if y_pred == y_true:
                correct += 1
        
        return correct / len(y)
    
    def _apply_vortex(
        self,
        x: Tuple[int, int],
        params: Dict[str, Any]
    ) -> Tuple[int, int]:
        """Применение vortex step activation."""
        table = params['table']
        # Для простоты: vortex = table lookup
        return x
    
    def _state_to_class(self, state: Tuple[int, int]) -> int:
        """Извлечение класса из состояния."""
        if self.params is None:
            return 0
        
        table = self.params['table']
        return table.get(state, 0)


# Экспорт
__all__ = ['AFSMClassifier', 'AFSMConfig']
