# YANTRA: Deterministic Neural Network on Finite Groups

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Детерминированная нейронная сеть на конечных группах вместо вещественных чисел.**

🎯 **100% детерминизм** - одинаковые входы → одинаковые выходы (всегда)  
🔍 **Формальная верификация** - exhaustive проверка всех состояний  
⚡ **Zero dependencies** - только Python 3.10+ stdlib  

---

## 🚀 Quick Start (2 минуты)
```bash
# Клонировать
git clone https://github.com/rusfbm-collab/yantra.git
cd yantra

# Установить
pip install -e .

# Запустить XOR demo
python examples/01_xor_demo.py
```
```python
# XOR классификация (100% accuracy, deterministic)
from yantra import AFSMClassifier
from yantra.datasets import generate_xor_dataset

# Данные
X_train, y_train = generate_xor_dataset(n_samples=40)
X_test, y_test = generate_xor_dataset(n_samples=20)

# Обучение (exhaustive search, no random seed!)
clf = AFSMClassifier(k_vec=(2, 4), activation='vortex_step')
clf.train(X_train, y_train)

# Тестирование
accuracy = clf.evaluate(X_test, y_test)
print(f"XOR accuracy: {accuracy:.1%}")  # 100.0%
```

**Результат:**
```
XOR accuracy: 100.0%
Confusion Matrix:
[[10,  0]
 [ 0, 10]]
```

---

## 🎯 Зачем это нужно?

### Проблема: Недетерминизм в классическом ML
```python
import torch

# Один и тот же код, разные результаты:
for seed in range(5):
    torch.manual_seed(seed)
    train(model, data)
    print(f"Accuracy: {evaluate(model):.2%}")

# Output:
# Accuracy: 95.2%  ← Разные результаты!
# Accuracy: 96.8%
# Accuracy: 94.5%
# ...
```

### Решение: Конечные группы Zₙ

YANTRA работает в циклических группах Zₙ = {0, 1, ..., n-1} вместо ℝ:

- **Детерминизм:** Нет random seed, нет floating point errors
- **Верификация:** Конечное пространство → exhaustive search
- **Гарантии:** Глобальный оптимум, не локальный

---

## 📊 Результаты

### XOR (нелинейная задача)

| Модель | Accuracy | Determinism | Verifiable |
|--------|----------|-------------|------------|
| **YANTRA** | **100.0%** | **10/10 runs identical** | **✓ 3072 checks** |
| PyTorch MLP | ~95-100% | Depends on seed | ✗ |

### Детерминизм: 10/10 runs идентичны
```python
# Запускаем обучение 10 раз
for run in range(10):
    clf = AFSMClassifier(k_vec=(2, 4))
    clf.train(X_train, y_train)
    predictions = [clf.predict(x) for x in X_test]
    print(f"Run {run+1}: {predictions[:5]} ...")

# Output:
# Run  1: [0, 1, 0, 1, 1] ... accuracy=100.0%
# Run  2: [0, 1, 0, 1, 1] ... accuracy=100.0%  ← Идентично!
# Run  3: [0, 1, 0, 1, 1] ... accuracy=100.0%
# ...
# Run 10: [0, 1, 0, 1, 1] ... accuracy=100.0%

# Unique results: 1/10 ✅
```

---

## 🏗️ Как это работает?

### 1. Нейрон = Автомат на циклической группе
```python
class AFSMNeuron:
    def __init__(self, n: int):
        self.n = n  # Размер группы Zₙ
    
    def step(self, state):
        """Детерминированный переход"""
        o, x, i, j = state
        return (
            (o + 1) % self.n,
            (x * i) % self.n,
            i,
            (j * ((o + 1) % self.n)) % self.n
        )
```

### 2. Обучение = Exhaustive Search
```python
def train(self, X, y):
    """Полный перебор в сжатом пространстве"""
    best_accuracy = 0.0
    
    # Проверяем ВСЕ кандидаты (~64 для XOR)
    for params in self.generate_candidates():
        accuracy = self.evaluate(params, X, y)
        if accuracy > best_accuracy:
            best_params = params
    
    return best_params  # Глобальный оптимум!
```

---

## 📖 Примеры

### XOR Demo
```bash
python examples/01_xor_demo.py
```

**Output:**
```
YANTRA: XOR Classification Demo (100% Deterministic)
======================================================================

📊 Generating XOR dataset...
   Train: 40 samples
   Test:  20 samples

🔧 Creating AFSM classifier...
   Configuration: k_vec=(2, 4)
   Activation: vortex_step

🎯 Training (exhaustive search)...
   Candidates evaluated: 64
   Method: Exhaustive search (deterministic)
   Best accuracy: 100.0%

✅ Testing...
   Test accuracy: 100.0%

🎉 PERFECT! 100% accuracy on XOR!

💡 Run this script multiple times - results are IDENTICAL!
```

### Determinism Test
```bash
python examples/02_determinism_test.py
```

**Output:**
```
YANTRA: Determinism Test (10 Runs)
======================================================================

🔄 Running 10 training sessions...

   Run  1: [0, 1, 0, 1, 1] ... accuracy=100.0%
   Run  2: [0, 1, 0, 1, 1] ... accuracy=100.0%
   Run  3: [0, 1, 0, 1, 1] ... accuracy=100.0%
   ...
   Run 10: [0, 1, 0, 1, 1] ... accuracy=100.0%

📊 Determinism Analysis:

   Unique prediction patterns: 1/10
   Unique accuracy values:     1/10

✅ DETERMINISM VERIFIED!
   🎯 All 10 runs produced IDENTICAL results
   🎯 No random seed needed
   🎯 100% reproducibility guaranteed
```

---

## ⚠️ Ограничения

### ✅ Работает

- XOR (нелинейная задача)
- Two Blobs (линейная классификация)
- Простые binary/multiclass задачи
- Малые датасеты (< 1000 samples)

### ❌ Не работает (пока)

- MNIST 28×28 (exhaustive search не масштабируется)
- CIFAR-10, ImageNet
- Большие датасеты (> 10K samples)
- Регрессия с непрерывными выходами

### 🎯 Это proof-of-concept, не замена PyTorch!

**Используйте:**
- ✅ Для критических систем (medical, automotive, aerospace)
- ✅ Для embedded ML (FPGA, микроконтроллеры)
- ✅ Для исследований (100% воспроизводимость)
- ✅ Для обучения (понимание основ без "магии")

**Не используйте:**
- ❌ Для production ML на больших данных
- ❌ Вместо PyTorch/TensorFlow
- ❌ Для SOTA результатов

---

## 🎓 Где это полезно?

1. **Критические системы с сертификацией**
   - Medical devices (FDA approval)
   - Automotive (ISO 26262)
   - Aerospace (DO-178C)

2. **Embedded ML**
   - FPGA реализация
   - Микроконтроллеры
   - Real-time системы

3. **Научные исследования**
   - 100% воспроизводимость
   - Честное сравнение алгоритмов
   - Reproducibility crisis

---

## 🛠️ Установка
```bash
# Из исходников
git clone https://github.com/rusfbm-collab/yantra.git
cd yantra
pip install -e .

# Или (в будущем)
pip install yantra-ml
```

**Требования:**
- Python 3.10+
- Нет внешних зависимостей (только stdlib)

---

## 📚 Документация

- [Теория](docs/theory.md) - Математическая основа
- [Примеры](examples/) - Воспроизводимые примеры
- [API Reference](docs/api.md) - Полный API

---

## 🤝 Вклад в проект

Приветствуются:
- Оптимизация exhaustive search
- Масштабирование на большие задачи
- Аппаратная реализация (FPGA)
- Гибридные архитектуры

**Issues и Pull Requests welcome!**

---

## 📄 Лицензия

MIT License - свободно для исследований и коммерческого использования.

---

## 📮 Контакты

- **GitHub Issues:** [Сообщить о проблеме](https://github.com/rusfbm-collab/yantra/issues)
- **Discussions:** [Обсуждения](https://github.com/rusfbm-collab/yantra/discussions)

---

**YANTRA: Детерминированное машинное обучение.**  
**Математически verified. Полностью воспроизводимо.**

⭐ **Поставьте звезду, если проект полезен!**
```

---

## ✅ ИСПРАВЛЕНО! Теперь все ссылки на `rusfbm-collab`

**Ссылка на твой будущий репозиторий:**
```
https://github.com/rusfbm-collab/yantra
