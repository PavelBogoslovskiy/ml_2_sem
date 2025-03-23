from __future__ import annotations

from typing import List

import numpy as np

from descents import BaseDescent
from descents import get_descent


class LinearRegression:
    """
    Класс линейной регрессии.

    Parameters
    ----------
    descent_config : dict
        Конфигурация градиентного спуска.
    tolerance : float, optional
        Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional
        Критерий остановки по количеству итераций. По умолчанию равен 300.

    Attributes
    ----------
    descent : BaseDescent
        Экземпляр класса, реализующего градиентный спуск.
    tolerance : float
        Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int
        Критерий остановки по количеству итераций.
    loss_history : List[float]
        История значений функции потерь на каждой итерации.

    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
        """
        :param descent_config: gradient descent config
        :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
        :param max_iter: stopping criterion for iterations (int)
        """
        self.descent: BaseDescent = get_descent(descent_config)

        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.loss_history: List[float] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        self : LinearRegression
            Возвращает экземпляр класса с обученными весами.

        """
        # Инициализация весов. 
        # Большой разницы между нулями и случайными числами вроде нет для линейной регрессии
        # эти веса вообще не исп

        self.iterations = 0

        self.loss_history.append(self.calc_loss(x, y))
        for iter in range(1, self.max_iter + 1):
            self.iterations = iter
            # Вычисление градиента
            gradient = self.descent.calc_gradient(x, y)
            if np.any(np.isnan(gradient)):
                print("В векторе весов появлявились значения NaN")
                break
            
            # Обновление весов
            diff_w = self.descent.update_weights(gradient)

            curr_loss = self.calc_loss(x, y)
            if curr_loss == float('inf'):
                print("Бесконечно больной лосс")
                return self
            
            # Историчность loss
            self.loss_history.append(curr_loss)

            # Проверка на NaN
            if np.any(np.isnan(self.descent.w)):
                print("В векторе весов появлявились значения NaN")
                return self

            # Проверка на сходимость
            if np.linalg.norm(diff_w) < self.tolerance:
                print(f"Достигли сходимости на итерации {iter}")
                return self

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        prediction : np.ndarray
            Массив прогнозируемых значений.
        """
        return self.descent.predict(x)

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Расчёт значения функции потерь для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        loss : float
            Значение функции потерь.
        """
        return self.descent.calc_loss(x, y)
