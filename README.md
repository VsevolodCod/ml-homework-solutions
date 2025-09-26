# Machine Learning Course 2025 - Homework Solutions

Репозиторий с решениями домашних заданий по курсу машинного обучения 2025 года.

## Структура проекта

```
ml-course-2025/
├── homeworks/
│   ├── дз/                    # Исходные задания
│   │   ├── hw01_classification_and_attention/
│   │   ├── hw02_cross_entropy/
│   │   ├── hw03_qlearning/
│   │   └── hw04_policy_gradient/
│   └── решения/               # Мои решения
│       ├── дз_1/             # Classification & Attention
│       ├── дз_2/             # Cross Entropy
│       ├── дз_3/             # Q-Learning
│       └── дз_4/             # Policy Gradient
└── README.md
```

## Домашние задания

### ДЗ 1: Classification and Attention
**Папка:** `homeworks/решения/дз_1/`

- **Основные файлы:**
  - `hw-1.py` - Python скрипт с реализацией
  - `template_p01.py` - Шаблон задания
  - `submission_dict_fmnist_task_1.json` - Файл для сдачи
  - `hw_fmnist_data_dict.npy` - Данные FashionMNIST

- **Описание:** Реализация классификации изображений с использованием механизма внимания (attention) на датасете FashionMNIST.

### ДЗ 2: Cross Entropy
**Папка:** `homeworks/решения/дз_2/`

- **Основные файлы:**
  - `template_crossentropy.py` - Основная реализация
  - `template_crossentropy01.py` - Альтернативная версия

- **Описание:** Изучение и реализация функции потерь Cross Entropy для задач классификации.

### ДЗ 3: Q-Learning
**Папка:** `homeworks/решения/дз_3/`

- **Основные файлы:**
  - `template_qlearning.py` - Шаблон и реализация алгоритма

- **Описание:** Реализация алгоритма Q-Learning для обучения с подкреплением в игровой среде.

### ДЗ 4: Policy Gradient (REINFORCE)
**Папка:** `homeworks/решения/дз_4/`

- **Основные файлы:**
  - `template_reinforce.py` - Шаблон алгоритма REINFORCE
  - `sessions_to_send.json` - Данные сессий для отправки

- **Описание:** Реализация алгоритма Policy Gradient (REINFORCE) для решения задачи CartPole в OpenAI Gym.

## Технологии

- **Python 3.x**
- **PyTorch** - для нейронных сетей
- **NumPy** - для численных вычислений
- **Jupyter Notebook** - для интерактивной разработки
- **OpenAI Gym** - для сред обучения с подкреплением
- **Matplotlib/Seaborn** - для визуализации
- **Jupyter Notebook** - для интерактивной разработки
- **OpenAI Gym** - для сред обучения с подкреплением
- **Matplotlib/Seaborn** - для визуализации


## Как запустить

1. **Клонировать репозиторий:**
   ```bash
   git clone https://github.com/VsevolodCod/ml-homework-solutions.git
   cd ml-homework-solutions
   ```

2. **Установить зависимости:**
   ```bash
   pip install torch numpy matplotlib jupyter gym
   ```

3. **Запустить Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **Открыть нужное домашнее задание** в соответствующей папке `homeworks/решения/дз_X/`


## Контакты

Если есть вопросы по решениям - пишите:  https://t.me/Comando1207 .

---
*Курс машинного обучения 2025 | Все права защищены*