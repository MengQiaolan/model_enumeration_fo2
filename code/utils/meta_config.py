from __future__ import annotations
from collections import defaultdict

from enum import Enum
from functools import reduce
import math
import os
import argparse
import logging
import logzero

import functools

from itertools import combinations
import numpy as np

from logzero import logger
from typing import Callable
from contexttimer import Timer

from wfomc.cell_graph.cell_graph import build_cell_graphs
from wfomc.problems import WFOMCProblem

from wfomc.utils import MultinomialCoefficients, multinomial, \
    multinomial_less_than, RingElement, Rational, round_rational
from wfomc.cell_graph import CellGraph, Cell
from wfomc.context import WFOMCContext, WFOMCContext
from wfomc.parser import parse_input
from wfomc.fol.syntax import Const, Pred, QFFormula, PREDS_FOR_EXISTENTIAL
from wfomc.utils.polynomial import coeff_dict, create_vars, expand

def get_config_weight_standard(cell_graph: CellGraph,
                               cell_config: dict[Cell, int]) -> RingElement:
    res = Rational(1, 1)
    for i, (cell_i, n_i) in enumerate(cell_config.items()):
        if n_i == 0:
            continue
        res = res * cell_graph.get_cell_weight(cell_i) ** n_i
        res = res * cell_graph.get_two_table_weight(
            (cell_i, cell_i)
        ) ** (n_i * (n_i - 1) // 2)
        for j, (cell_j, n_j) in enumerate(cell_config.items()):
            if j <= i:
                continue
            if n_j == 0:
                continue
            res = res * cell_graph.get_two_table_weight(
                (cell_i, cell_j)
            ) ** (n_i * n_j)
    # logger.debug('Config weight: %s', res)
    return res

def _get_config_weights(cell_graph: CellGraph, domain_size: int) \
        -> tuple[list[tuple[int, ...]], list[Rational]]:
    configs = []
    weights = []
    cells = cell_graph.get_cells()
    n_cells = len(cells)
    for partition in multinomial(n_cells, domain_size):
        coef = MultinomialCoefficients.coef(partition)
        cell_config = dict(zip(cells, partition))
        weight = coef * get_config_weight_standard(
            cell_graph, cell_config
        )
        if weight != 0:
            configs.append(partition)
            weights.append(weight)
    return configs, weights

def _adjust_config_weights(configs: list[tuple[int, ...]],
                               weights: list[RingElement],
                               src_cell_graph: CellGraph,
                               dest_cell_graph: CellGraph) -> \
            tuple[list[tuple[int, ...]], list[Rational]]:
    src_cells = src_cell_graph.get_cells()
    dest_cells = dest_cell_graph.get_cells()
    mapping_mat = np.zeros(
        (len(src_cells), len(dest_cells)), dtype=np.int32)
    for idx, cell in enumerate(src_cells):
        dest_idx = dest_cells.index(
            cell.drop_preds(prefixes=PREDS_FOR_EXISTENTIAL)
        )
        mapping_mat[idx, dest_idx] = 1

    adjusted_config_weight = defaultdict(lambda: Rational(0, 1))
    for config, weight in zip(configs, weights):
        adjusted_config_weight[tuple(np.dot(
            config, mapping_mat).tolist())] += weight
    return list(adjusted_config_weight.keys()), \
        list(adjusted_config_weight.values())

def get_template_config(context: WFOMCContext, cell_graph: CellGraph, uni_cell_graph: CellGraph, domain_size) -> set[tuple[int, ...]]:

    original_level = logger.getEffectiveLevel()
    logger.setLevel(logging.CRITICAL)
    try:
        domain_size: int = domain_size
        MultinomialCoefficients.setup(domain_size)
        configs, weights = _get_config_weights(
            cell_graph, domain_size)

        # for config, weight in zip(configs, weights):
        #     print(config, weight)

        template_configs: set[tuple[int, ...]] = set()

        if context.contain_existential_quantifier():
            # Precomputed weights for cell configs
            m = len(context.ext_formulas)
            delta = max(2*m+1, m*(m+1))
            # self.uni_cell_graph.show()
            configs, weights = _adjust_config_weights(
                configs, weights,
                cell_graph, uni_cell_graph
            )
            for config, weight in zip(configs, weights):
                for idx, ni in enumerate(config):
                    if ni > delta:
                        ls = list(config)
                        ls[idx] = delta
                        config = tuple(ls)
                if weight > 0:
                    # print(config, weight)
                    template_configs.add(config)
            cell_graph = uni_cell_graph
        # wfomc = sum(weights)
        return template_configs

    finally:
        logger.setLevel(original_level)

@functools.lru_cache(maxsize=None)
def sat_config(config: tuple):
    return False

def calculate_meta_config_down(config: tuple, meta_cfgs: set[tuple]): 
    if not sat_config(config):
        return False
    is_meta = True
    for i in range(len(config)):
        if config[i] == 0:
            continue
        l = list(config)
        l[i] -= 1
        sub_config = tuple(l)
        if calculate_meta_config(sub_config, meta_cfgs) and l[i] != 0:
            is_meta = False
    if is_meta:
        meta_cfgs.add(config)
    return True


def calculate_meta_config(config: tuple, hold: list, meta_cfgs: set[tuple], delta):
    # hold[i]==True means that we don't need to increase the i-th element
    # since it has been increased to the upper bound 
    # or its incremented config can be derived from a meta_cfg
    for i in range(len(config)):
        if config[i] >= delta or hold[i]:
            hold[i] = True
            continue
        l = list(config)
        l[i] += 1
        inc_config = tuple(l)
        if inc_config in meta_cfgs:
            hold[i] = True
            continue
        if any(all((a >= b and b > 0) or (a == b and b == 0) for a, b in zip(inc_config, meta_cfg))
               for meta_cfg in meta_cfgs):
            hold[i] = True
            continue
        # call sat function as little as possible
        if (not hold[i]) and sat_config(inc_config):
            meta_cfgs.add(inc_config)
            hold[i] = True

    for i in range(len(config)):
        if not hold[i]:
            l = list(config)
            l[i] += 1
            calculate_meta_config(tuple(l), hold.copy(), meta_cfgs, delta)

delta = 5

def generate_config_class(domain_size:int, len_config:int):
    for k_zero in range(len_config):
        if (len_config - k_zero) * (delta - 1) >= domain_size:
            yield (k_zero, 0)
        for k_sup in range(1, len_config + 1 - k_zero):
            if k_sup * delta + (len_config - k_zero - k_sup) > domain_size:
                break
            yield (k_zero, k_sup)

def generate_base_configs(len_config, k_zero, k_sup, n):
    def backtrack(cur_tuple: tuple, zero_left, sup_left):
        if cur_tuple.__len__() == len_config:
            yield cur_tuple
            return
        if len_config - cur_tuple.__len__() >= zero_left + sup_left:
            if zero_left > 0:
                yield from backtrack(cur_tuple + (0,), zero_left-1, sup_left)
            if sup_left > 0:
                yield from backtrack(cur_tuple + (delta,), zero_left, sup_left-1)
            if len_config - cur_tuple.__len__() -1 >= zero_left + sup_left:
                yield from backtrack(cur_tuple + (1,), zero_left, sup_left)
    yield from backtrack(tuple(), k_zero, k_sup)

def calculate_template_config(config: tuple, hold: list, tpl_configs: set[tuple], delta):
    for i in range(len(config)):
        if config[i] == delta or config[i] == 0 or hold[i]:
            hold[i] = True
            continue
        l = list(config)
        l[i] += 1
        inc_config = tuple(l)
        if inc_config in tpl_configs:
            hold[i] = True
            continue
        if any(all((a >= b) for a, b in zip(inc_config, tpl_cfg))
               for tpl_cfg in tpl_configs):
            hold[i] = True
            continue
        # call sat function as little as possible
        if (not hold[i]) and sat_config(inc_config):
            tpl_configs.add(inc_config)
            hold[i] = True

    for i in range(len(config)):
        if not hold[i]:
            l = list(config)
            l[i] += 1
            calculate_template_config(tuple(l), hold.copy(), tpl_configs, delta)


if __name__ == '__main__':
    # meta_cfgs = set()
    # delta = 4
    # calculate_meta_config((0, 0, 0), [False, False, False], meta_cfgs, delta)
    # print(meta_cfgs)

    # for t in generate_tuples(4, 1, 1, 7):
    #     print(t)
        
    len_config = 3
    domain_size = 9
    for cfg_class in generate_config_class(domain_size, len_config):
        for base_config in generate_base_configs(len_config, cfg_class[0], cfg_class[1], 9):
            cfgs = set()
            calculate_meta_config(base_config, [False]*len_config, cfgs, delta)
            for cfg in cfgs:
                print(cfg)