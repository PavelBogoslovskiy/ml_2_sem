import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Находит оптимальный порог для разбиения вектора признака по критерию Джини.

    Критерий Джини определяется следующим образом:
    .. math::
        Q(R) = -\\frac {|R_l|}{|R|}H(R_l) -\\frac {|R_r|}{|R|}H(R_r),

    где:
    * :math:`R` — множество всех объектов,
    * :math:`R_l` и :math:`R_r` — объекты, попавшие в левое и правое поддерево соответственно.

    Функция энтропии :math:`H(R)`:
    .. math::
        H(R) = 1 - p_1^2 - p_0^2,

    где:
    * :math:`p_1` и :math:`p_0` — доля объектов класса 1 и 0 соответственно.

    Указания:
    - Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    - В качестве порогов, нужно брать среднее двух соседних (при сортировке) значений признака.
    - Поведение функции в случае константного признака может быть любым.
    - При одинаковых приростах Джини нужно выбирать минимальный сплит.
    - Для оптимизации рекомендуется использовать векторизацию вместо циклов.

    Parameters
    ----------
    feature_vector : np.ndarray
        Вектор вещественнозначных значений признака.
    target_vector : np.ndarray
        Вектор классов объектов (0 или 1), длина `feature_vector` равна длине `target_vector`.

    Returns
    -------
    thresholds : np.ndarray
        Отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно разделить на
        два различных поддерева.
    ginis : np.ndarray
        Вектор со значениями критерия Джини для каждого порога в `thresholds`.
    threshold_best : float
        Оптимальный порог для разбиения.
    gini_best : float
        Оптимальное значение критерия Джини.

    """
    sort_ind = np.argsort(feature_vector)
    feature_vector = feature_vector[sort_ind]
    target_vector = target_vector[sort_ind]

    diffs = np.diff(feature_vector)
    mask = diffs != 0
    thresholds = (feature_vector[:-1] + feature_vector[1:]) / 2
    thresholds = thresholds[mask]

    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    total_samples = len(target_vector)
    total_pos = np.sum(target_vector)

    l_pos = np.cumsum(target_vector)[:-1][mask]
    l_cnt = np.arange(1, total_samples)[mask]

    r_pos = total_pos - l_pos
    r_cnt = total_samples - l_cnt

    # убрали пусты е поддеревья
    valid = (l_cnt > 0) & (r_cnt > 0)
    thresholds = thresholds[valid]
    l_cnt = l_cnt[valid]
    r_cnt = r_cnt[valid]
    l_pos = l_pos[valid]
    r_pos = r_pos[valid]

    # джини для левой и правой частей
    l_p1 = l_pos / l_cnt
    l_p0 = 1 - l_p1
    H_l = 1 - l_p1**2 - l_p0**2

    r_p1 = r_pos / r_cnt
    r_p0 = 1 - r_p1
    H_r = 1 - r_p1**2 - r_p0**2

    ginis = - (l_cnt / total_samples) * H_l - (r_cnt / total_samples) * H_r

    best_ind = np.argmax(ginis)
    threshold_best = thresholds[best_ind]
    gini_best = ginis[best_ind]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
    ):
        if any(ft not in {"real", "categorical"} for ft in feature_types):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        """
        Обучение узла дерева решений.

        Если все элементы в подвыборке принадлежат одному классу, узел становится терминальным.

        Parameters
        ----------
        sub_X : np.ndarray
            Подвыборка признаков.
        sub_y : np.ndarray
            Подвыборка меток классов.
        node : dict
            Узел дерева, который будет заполнен информацией о разбиении.

        """
        node["depth"] = depth

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return
        
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {
                    key: clicks.get(key, 0) / count for key, count in counts.items()
                }
                sorted_categories = sorted(ratio, key=ratio.get)
                categories_map = {
                    category: i for i, category in enumerate(sorted_categories)
                }
                feature_vector = np.vectorize(categories_map.get)(sub_X[:, feature])
            else:
                raise ValueError("Некорректный тип признака")

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = threshold
                    categories_map_best = categories_map

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        l_mask = split
        r_mask = ~split

        if (np.sum(l_mask) == 0
            or np.sum(r_mask) == 0
            or (self._min_samples_leaf is not None
                and (np.sum(l_mask) < self._min_samples_leaf
                    or np.sum(r_mask) < self._min_samples_leaf))):
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
        
        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
            node["categories_map"] = categories_map_best
        else:
            raise ValueError("Некорректный тип признака")

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        """
        Рекурсивное предсказание класса для одного объекта по узлу дерева решений.

        Если узел терминальный, возвращается предсказанный класс.
        Если узел не терминальный, выборка передается в соответствующее поддерево для дальнейшего предсказания.

        Parameters
        ----------
        x : np.ndarray
            Вектор признаков одного объекта.
        node : dict
            Узел дерева решений.

        Returns
        -------
        int
            Предсказанный класс объекта.
        """
        if node["type"] == "terminal":
            return node["class"]
        
        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif feature_type == "categorical":
            categories_map = node["categories_map"]
            feature_value = categories_map.get(x[feature], float('inf'))
            
            if feature_value < node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        
        else:
            raise ValueError("Unknown feature type")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
