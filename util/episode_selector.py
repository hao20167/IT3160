import sys

sys.path.append(".")
import numpy as np
from util.sum_tree import SumTree
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from scipy.stats import iqr
from sklearn.model_selection import GridSearchCV


class start_selector(object):
    def __init__(self, start_list, initial_priority_list):
        self.start_list = start_list
        self.start_index_list = range(len(start_list))
        self.current_size = len(start_list)
        self.tree = SumTree(len(self.start_list))
        self.initial_priority_list = initial_priority_list
        for i in self.start_index_list:
            self.tree.update(i, initial_priority_list[i])

    def sample(self):
        batch_index, IS_weight = self.tree.get_batch_index(
            current_size=self.current_size, batch_size=1, beta=0
        )
        batch_index = batch_index.tolist()[0]
        return self.start_list[batch_index], batch_index

    def update_single_priorities(self, index, priority):
        self.tree.update(data_index=index, priority=priority)

    def update_batch_priorities(self, batch_index, priorities):
        for index, priority in zip(batch_index, priorities):
            self.tree.update(data_index=index, priority=priority)

    def sample_determistic(self):
        index = self.initial_priority_list.index(max(self.initial_priority_list))
        start = self.start_index_list[index]
        return start, index


def get_transformation_exp(beta, buy_hold_return_list):
    priority_list = []
    for return_rate in buy_hold_return_list:
        priority_list.append(np.exp(beta * return_rate))
    return priority_list


def get_transformation_sigmoid(beta, buy_hold_return_list):
    priority_list = []
    for return_rate in buy_hold_return_list:
        priority_list.append(1 / (1 + (np.exp((-return_rate * beta)))))
    return priority_list


def get_transformation_even(
    buy_hold_return_list, bandwidth=None, kernel="gaussian", beta=None
):

    if bandwidth is None:
        silverman_bandwidth = get_silverman_bandwidth(buy_hold_return_list)
        log_bandwidths = np.linspace(
            np.log10(0.01 * silverman_bandwidth),
            np.log10(10 * silverman_bandwidth),
            100,
        )
        bandwidths = 10**log_bandwidths
        kde = KernelDensity(kernel=kernel)
        grid = GridSearchCV(kde, {"bandwidth": bandwidths})
        grid.fit(np.array(buy_hold_return_list).reshape(-1, 1))
        bandwidth = grid.best_params_["bandwidth"]
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
        np.array(buy_hold_return_list)[:, np.newaxis]
    )
    log_density = kde.score_samples(np.array(buy_hold_return_list)[:, np.newaxis])
    density = np.exp(log_density)
    weights = 1.0 / density
    weights /= np.sum(weights)
    return weights


def get_transformation_even_based_boltzmann(
    buy_hold_return_list, bandwidth=None, kernel="gaussian", beta=10
):

    if bandwidth is None:
        silverman_bandwidth = get_silverman_bandwidth(buy_hold_return_list)
        log_bandwidths = np.linspace(
            np.log10(0.01 * silverman_bandwidth),
            np.log10(10 * silverman_bandwidth),
            100,
        )
        bandwidths = 10**log_bandwidths
        kde = KernelDensity(kernel=kernel)
        grid = GridSearchCV(kde, {"bandwidth": bandwidths})
        grid.fit(np.array(buy_hold_return_list).reshape(-1, 1))
        bandwidth = grid.best_params_["bandwidth"]
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
        np.array(buy_hold_return_list)[:, np.newaxis]
    )
    log_density = kde.score_samples(np.array(buy_hold_return_list)[:, np.newaxis])
    density = np.exp(log_density)
    weights = 1 / density
    weights /= np.sum(weights)
    final_weights = []
    assert len(weights) == len(buy_hold_return_list)
    for return_rate, weight in zip(buy_hold_return_list, weights):
        final_weights.append(weight * np.exp(beta * return_rate))
    return final_weights


def get_transformation_even_based_sigmoid(
    buy_hold_return_list, bandwidth=None, kernel="gaussian", beta=10
):

    if bandwidth is None:
        silverman_bandwidth = get_silverman_bandwidth(buy_hold_return_list)
        log_bandwidths = np.linspace(
            np.log10(0.01 * silverman_bandwidth),
            np.log10(10 * silverman_bandwidth),
            100,
        )
        bandwidths = 10**log_bandwidths
        kde = KernelDensity(kernel=kernel)
        grid = GridSearchCV(kde, {"bandwidth": bandwidths})
        grid.fit(np.array(buy_hold_return_list).reshape(-1, 1))
        bandwidth = grid.best_params_["bandwidth"]
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
        np.array(buy_hold_return_list)[:, np.newaxis]
    )
    log_density = kde.score_samples(np.array(buy_hold_return_list)[:, np.newaxis])
    density = np.exp(log_density)
    weights = 1 / density
    weights /= np.sum(weights)
    final_weights = []
    assert len(weights) == len(buy_hold_return_list)
    for return_rate, weight in zip(buy_hold_return_list, weights):
        final_weights.append(weight * 1 / (1 + (np.exp((-return_rate * beta)))))
    return final_weights


def get_silverman_bandwidth(data):
    std_dev = np.std(data)
    interquartile_range = iqr(data)
    n = len(data)
    return 1.06 * min(std_dev, interquartile_range / 1.34) * n ** (-1 / 5)


def get_transformation_even_risk(
    buy_hold_return_list, bandwidth=None, kernel="gaussian", beta=None, risk_bond=0.1
):

    lower_risk_bond = risk_bond / 2
    upper_risk_bond = 1 - risk_bond / 2
    upper_value = np.quantile(buy_hold_return_list, upper_risk_bond)
    lower_value = np.quantile(buy_hold_return_list, lower_risk_bond)
    if bandwidth is None:
        silverman_bandwidth = get_silverman_bandwidth(buy_hold_return_list)
        log_bandwidths = np.linspace(
            np.log10(0.01 * silverman_bandwidth),
            np.log10(10 * silverman_bandwidth),
            100,
        )
        bandwidths = 10**log_bandwidths
        kde = KernelDensity(kernel=kernel)
        grid = GridSearchCV(kde, {"bandwidth": bandwidths})
        grid.fit(np.array(buy_hold_return_list).reshape(-1, 1))
        bandwidth = grid.best_params_["bandwidth"]
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
        np.array(buy_hold_return_list)[:, np.newaxis]
    )
    log_density = kde.score_samples(np.array(buy_hold_return_list)[:, np.newaxis])
    density = np.exp(log_density)
    density = density / np.sum(density)
    weights = []
    for return_rate, single_density in zip(buy_hold_return_list, density):
        if return_rate >= lower_value and return_rate <= upper_value:
            weights.append(1 / single_density)
        else:
            weights.append(1)
    weights = np.array(weights)

    return weights


def get_transformation_even_based_boltzmann_risk(
    buy_hold_return_list, bandwidth=None, kernel="gaussian", beta=10, risk_bond=0.1
):

    lower_risk_bond = risk_bond / 2
    upper_risk_bond = 1 - risk_bond / 2
    upper_value = np.quantile(buy_hold_return_list, upper_risk_bond)
    lower_value = np.quantile(buy_hold_return_list, lower_risk_bond)
    if bandwidth is None:
        silverman_bandwidth = get_silverman_bandwidth(buy_hold_return_list)
        log_bandwidths = np.linspace(
            np.log10(0.01 * silverman_bandwidth),
            np.log10(10 * silverman_bandwidth),
            100,
        )
        bandwidths = 10**log_bandwidths
        kde = KernelDensity(kernel=kernel)
        grid = GridSearchCV(kde, {"bandwidth": bandwidths})
        grid.fit(np.array(buy_hold_return_list).reshape(-1, 1))
        bandwidth = grid.best_params_["bandwidth"]
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
        np.array(buy_hold_return_list)[:, np.newaxis]
    )
    log_density = kde.score_samples(np.array(buy_hold_return_list)[:, np.newaxis])
    density = np.exp(log_density)
    density = density / np.sum(density)
    weights = []
    for return_rate, single_density in zip(buy_hold_return_list, density):
        if return_rate >= lower_value and return_rate <= upper_value:
            weights.append(1 / single_density)
        else:
            weights.append(1)
    weights = np.array(weights)
    final_weights = []
    assert len(weights) == len(buy_hold_return_list)
    for return_rate, weight in zip(buy_hold_return_list, weights):
        final_weights.append(weight * np.exp(beta * return_rate))
    return final_weights


def get_transformation_even_based_sigmoid_risk(
    buy_hold_return_list, bandwidth=None, kernel="gaussian", beta=10, risk_bond=0.1
):

    lower_risk_bond = risk_bond / 2
    upper_risk_bond = 1 - risk_bond / 2
    upper_value = np.quantile(buy_hold_return_list, upper_risk_bond)
    lower_value = np.quantile(buy_hold_return_list, lower_risk_bond)
    if bandwidth is None:
        silverman_bandwidth = get_silverman_bandwidth(buy_hold_return_list)
        log_bandwidths = np.linspace(
            np.log10(0.01 * silverman_bandwidth),
            np.log10(10 * silverman_bandwidth),
            100,
        )
        bandwidths = 10**log_bandwidths
        kde = KernelDensity(kernel=kernel)
        grid = GridSearchCV(kde, {"bandwidth": bandwidths})
        grid.fit(np.array(buy_hold_return_list).reshape(-1, 1))
        bandwidth = grid.best_params_["bandwidth"]
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(
        np.array(buy_hold_return_list)[:, np.newaxis]
    )
    log_density = kde.score_samples(np.array(buy_hold_return_list)[:, np.newaxis])
    density = np.exp(log_density)
    density = density / np.sum(density)
    weights = []
    for return_rate, single_density in zip(buy_hold_return_list, density):
        if return_rate >= lower_value and return_rate <= upper_value:
            weights.append(1 / single_density)
        else:
            weights.append(1)
    weights = np.array(weights)
    final_weights = []
    assert len(weights) == len(buy_hold_return_list)
    for return_rate, weight in zip(buy_hold_return_list, weights):
        final_weights.append(weight * 1 / (1 + (np.exp((-return_rate * beta)))))
    return final_weights
