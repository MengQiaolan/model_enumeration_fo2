from __future__ import annotations

import os
import argparse
import logging
import logzero
import functools
from logzero import logger
from contexttimer import Timer
from itertools import combinations

import numpy as np
from wfomc.cython_modules.matrix_utils import swap_rows_cols
from wfomc.parser import parse_input
from wfomc.fol.syntax import AtomicFormula, Const, Pred, X
from wfomc.utils import MultinomialCoefficients, multinomial
from wfomc.cell_graph.components import Cell

from utils.config_sat import ConfigSAT
from utils.enum_context import EnumContext
from utils.enum_utils import Pred_P, Pred_A, remove_aux_atoms


STATISTICS_META_CONFIG: dict[tuple[int], int] = dict()
DOMAIN_SIZE: int = 0
MODEL_COUNT: int = 0
PRINT_MODEL: bool = False
SAT_COUNT: int = 0
DELTA: int = 0

CACHE_OF_MODELS = None

EARLY_STOP: bool = True
MC_LIMIT: int = 1000000

Domain_to_Cell: dict[Const, Cell] = {}
Rel_Dict: dict[tuple[Cell, Cell], list[frozenset[AtomicFormula]]] = {}

Rpred_to_Zpred: dict[Pred, Pred] = {}
Zpred_to_Rpred: dict[Pred, Pred] = {}

Evi_to_Xpred: dict[frozenset[AtomicFormula], Pred] = {}
Xpred_to_Evi: dict[Pred, frozenset[AtomicFormula]] = {}

Config_SAT_pre: ConfigSAT = None

Config_SAT_unary: ConfigSAT = None

class ExitRecursion(Exception):
    pass

def update_PZ_evidence(evidence_dict: dict[Const, Pred],
                      cc_xpred: dict[Pred, int],
                      relation: frozenset[AtomicFormula],
                      first_e: Const, second_e: Const):
    '''
    update P and Z evidence when considering a new relation (2-table)
    '''
    cc_xpred[evidence_dict[first_e]] -= 1
    cc_xpred[evidence_dict[second_e]] -= 1
    for atom in relation:
        # consider the influence of positive R literals to correspoding block (Z)
        if atom.pred.name.startswith('@R') and atom.positive:
            z_pred = Rpred_to_Zpred[atom.pred]
            firt_arg = first_e if atom.args[0] == X else second_e
            if z_pred(X) in Xpred_to_Evi[evidence_dict[firt_arg]]:
                # update x pred when deleting block (z)
                evidence_dict[firt_arg] = Evi_to_Xpred[frozenset({~z_pred(X)}|
                    {atom for atom in Xpred_to_Evi[evidence_dict[firt_arg]] if atom != z_pred(X)})]
    # update x pred when the second element add P unary evidence according to the relation
    evidence_dict[second_e] = Evi_to_Xpred[frozenset({Pred_P(X)}|
                                {atom for atom in Xpred_to_Evi[evidence_dict[second_e]] if atom != ~Pred_P(X)})]
    # update cc after updating evidence
    cc_xpred[evidence_dict[first_e]] += 1
    cc_xpred[evidence_dict[second_e]] += 1

def clean_P_evidence(evidence_dict: dict[Const, Pred], cc_xpred: dict[Pred, int]):
    '''
    clean the P evidence of all elements before sampling a ego sturcture
    '''
    for e, xpred in evidence_dict.items():
        cc_xpred[xpred] -= 1
        evidence_dict[e] = Evi_to_Xpred[frozenset({~Pred_P(X)}|
            {atom for atom in Xpred_to_Evi[xpred]
                    if not atom.pred.name.startswith('@P')})]
        cc_xpred[evidence_dict[e]] += 1

def update_A_evidence(ego_element: Const, evidence_dict: dict[Const, Pred], cc_xpred: dict[Pred, int]):
    '''
    update A evidence after determining the ego element
    '''
    cc_xpred[evidence_dict[ego_element]] -= 1
    evidence_dict[ego_element] = Evi_to_Xpred[frozenset({Pred_A(X)}|
                {atom for atom in Xpred_to_Evi[evidence_dict[ego_element]] if atom != ~Pred_A(X)})]
    cc_xpred[evidence_dict[ego_element]] += 1

def domain_recursion(domain: list[Const],
                     evidence_dict: dict[Const, Pred],
                     cc_xpred: dict[Pred, int],
                     cur_model):
    if domain.__len__() == 0:
        global MODEL_COUNT, PRINT_MODEL
        MODEL_COUNT += 1
        if PRINT_MODEL:
            print(f'Model {MODEL_COUNT}:')
            print(cur_model)
        global CACHE_OF_MODELS
        CACHE_OF_MODELS.append(cur_model)
        if EARLY_STOP and MODEL_COUNT >= MC_LIMIT:
            raise ExitRecursion
        return

    ego_element = domain[0]
    domain = domain[1:]
    # print('    '*(3-len(domain))+f'current element: {ego_element}')
    clean_P_evidence(evidence_dict, cc_xpred)
    update_A_evidence(ego_element, evidence_dict, cc_xpred)
    pair_recursion(ego_element=ego_element, domain_todo=domain, domain_done=[],
                           evidence_dict=evidence_dict, cc_xpred=cc_xpred,
                           cur_model=cur_model)

def pair_recursion(ego_element: Const,
                           domain_todo: list[Const], domain_done: list[Const],
                           evidence_dict: dict[Const, Pred], cc_xpred: dict[Pred, int],
                           cur_model):
    if domain_todo.__len__() == 0:
        cc_xpred[evidence_dict[ego_element]] -= 1
        del evidence_dict[ego_element]
        domain_recursion(domain_done, evidence_dict, cc_xpred, cur_model)
        return
    # cur_element = domain_todo.pop()
    # domain_done.add(cur_element)
    cur_element = domain_todo[0]
    domain_todo = domain_todo[1:]
    domain_done = domain_done + [cur_element]
    for idx, rel in enumerate(Rel_Dict[(Domain_to_Cell[ego_element], Domain_to_Cell[cur_element])]):
        # we need to keep the original evidence set for the next iteration (next relation)
        # as same as the cc_xpred, domain_todo, domain_done
        new_evidence_dict = evidence_dict.copy()
        new_cc_xpred = cc_xpred.copy()
        # update evidence about P and Z according to the selected relation
        update_PZ_evidence(new_evidence_dict, new_cc_xpred, rel, ego_element, cur_element)
        # print('  '+'    '*(3-len(domain_done)-len(domain_todo))+f'    #({ego_element}, {cur_element}) => {set(rel)} => {new_evidence_dict}')
        # use tuple instead of dict (non-hashable) to use lru_cache
        if Config_SAT_pre.check_config_by_cache(tuple([new_cc_xpred[key] for key in sorted(new_cc_xpred.keys(), key=lambda x: int(x.name[2:]))])):
            # print('  '+'    '*(3-len(domain_done)-len(domain_todo))+f'    ({ego_element}, {cur_element}) => {set(rel)} => {new_evidence_dict}')
            new_cur_model = cur_model.copy()
            new_cur_model[ego_element][cur_element] = new_cur_model[cur_element][ego_element] = idx
            pair_recursion(
                ego_element, domain_todo.copy(), domain_done.copy(),
                new_evidence_dict, new_cc_xpred,
                new_cur_model)

def generate_config_class(domain_size:int, len_config:int, delta:int):
    '''
    Generate all possible tuple (k_0, k_delta) where
    k_0 is the number of cells that have no element and
    k_delta is the number of cells that have delta elements.
    '''
    for k_zero in range(len_config):
        if (len_config - k_zero) * (delta - 1) >= domain_size:
            yield (k_zero, 0)
        for k_delta in range(1, len_config + 1 - k_zero):
            if k_delta * delta + (len_config - k_zero - k_delta) > domain_size:
                break
            if delta == 1 and k_delta + k_zero != len_config:
                continue
            yield (k_zero, k_delta)

def generate_base_configs(len_config, k_zero, k_delta, delta):
    '''
    A base config is a config consisting of {0, 1, delta}
    Generate all possible base configurations based on the tuple (k_0, k_delta)
    A base config may be unsat
    '''
    def backtrack(cur_tuple: tuple, zero_left, sup_left):
        if sum(cur_tuple) > DOMAIN_SIZE:
            return
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
    yield from backtrack(tuple(), k_zero, k_delta)

TEMPLATE_CONFIGS = set()
def generate_template_config(base_config: tuple, flag: bool, delta: int, domain_size: int):
    '''
    A template config is a sat config consisting of (0, x, delta) where 1<=x<delta
    generate the template configurations based on the current config.
    '''
    if sum(base_config) > domain_size:
        return
    # flag==True means that the current config is based on a template config
    if not flag:
        # TODO: use sat_cc instead of sat_config
        flag = Config_SAT_unary.check_config_by_pysat(base_config)
    if flag:
        global TEMPLATE_CONFIGS
        if sum(base_config) == domain_size:
            if base_config not in TEMPLATE_CONFIGS:
                TEMPLATE_CONFIGS.add(base_config)
                yield base_config
            return
        elif sum(base_config) < domain_size and any(ni == delta for ni in base_config):
            if base_config not in TEMPLATE_CONFIGS:
                TEMPLATE_CONFIGS.add(base_config)
                yield base_config
        else:
            pass

    for i in range(len(base_config)):
        if base_config[i] == 0 or base_config[i] == delta :
            continue
        if base_config[i]+1 == delta:
            continue
        inc = list(base_config)
        inc[i] += 1
        yield from generate_template_config(tuple(inc), flag, delta, domain_size)

def generate_sat_configs(domain_size, template_config, delta):
    """
    Generate all satisfiable configurations based on the template configuration.
    """
    indeces = list()
    remaining = domain_size - sum(template_config)
    if remaining > 0:
        for idx, ni in enumerate(template_config):
            if ni == delta: # TODO
                indeces.append(idx)
        if len(indeces) == 0:
            raise RuntimeError('The template config is not valid')
        for extra_partition in multinomial(len(indeces), remaining):
            config = list(template_config)
            for idx, extra in zip(indeces, extra_partition):
                config[idx] += extra
            yield tuple(config)
    else:
        yield template_config

def cell_assignment(domain: set[Const], config: tuple[int]):
    """
    Distribute cells to domain elements according to the configuration.
    """
    def backtrack(remaining_elements, current_distribution):
        if len(current_distribution) == len(config):
            if all(len(d) == cnt for d, cnt in zip(current_distribution, config)):
                yield current_distribution
            return

        count = config[len(current_distribution)]
        for comb in combinations(remaining_elements, count):
            next_remaining = remaining_elements - set(comb)
            yield from backtrack(next_remaining, current_distribution + [list(comb)])

    yield from backtrack(domain, [])

def get_domain_order(cells: list[Cell],
                    cell_correlation: dict[tuple[Cell, Cell], int],
                    config: tuple[int]):

    cell_to_index: dict[Cell, int] = dict()
    for index, cell in enumerate(cells):
        cell_to_index[cell] = index

    cell_importance: dict[Cell, int] = dict()
    for i, cell_i in enumerate(cells):
        cell_importance[cell_i] = 0
        if config[i] == 0:
            continue
        for j, cell_j in enumerate(cells):
            if i == j:
                cell_importance[cell_i] += cell_correlation[(cell_i, cell_j)] * (config[j] - 1)
            else:
                cell_importance[cell_i] += cell_correlation[(cell_i, cell_j)] * config[j]

    # sorted_cells = sorted(cell_importance, key=lambda x: cell_importance[x], reverse=True)
    max_cell = max(cell_importance, key=cell_importance.get)

    config = list(config)
    domain_order = [cell_to_index[max_cell]]
    config[cell_to_index[max_cell]] -= 1
    while len(domain_order) != DOMAIN_SIZE:
        max_cell = None
        max_correlation = (-1,)*len(domain_order)
        for cell in cells:
            if config[cell_to_index[cell]] == 0:
                continue
            cur_correlation = ()
            for e in domain_order:
                cur_correlation = (cell_correlation[(cell, cells[e])], ) + cur_correlation
            if cur_correlation > max_correlation:
                max_cell = cell
                max_correlation = cur_correlation
        domain_order.append(cell_to_index[max_cell])
        config[cell_to_index[max_cell]] -= 1

    domain_order.reverse()
    return domain_order

def get_domain_list(domain_order, domain_partition):
    domain_list = []
    domain_partition_copy = []
    for lst in domain_partition:
        domain_partition_copy.append(lst.copy())
    for i in domain_order:
        domain_list.append(domain_partition_copy[i].pop())
    return domain_list

def parse_args():
    parser = argparse.ArgumentParser(
        description='Enumerate models of a given sentence',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True, help='wfomcs file')
    parser.add_argument('--domain_size', '-n', type=int, help='domain_size')
    parser.add_argument('--output_dir', '-o', type=str, default='./check-points')
    parser.add_argument('--print_model', '-p', action='store_true')
    parser.add_argument('--log_level', '-log', type=str, default='INFO')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(int(1e6))
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.log_level == 'D':
        logzero.loglevel(logging.DEBUG)
    elif args.log_level == 'C':
        logzero.loglevel(logging.CRITICAL)
    else:
        logzero.loglevel(logging.INFO)

    logger.setLevel(logging.CRITICAL)
    # logger.setLevel(logging.INFO)

    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')

    problem = parse_input(args.input)
    if args.domain_size:
        problem.domain = {Const(f'e{i}') for i in range(args.domain_size)}
        
    context: EnumContext = EnumContext(problem)

    DELTA = context.delta
    DOMAIN_SIZE = len(context.domain)
    MultinomialCoefficients.setup(DOMAIN_SIZE)

    original_cells: list[Cell] = context.original_cells
    original_original_cell_correlation: dict[tuple[Cell, Cell], int] = context.original_cell_correlation

    # Zpred_to_Rpred = context.zpred_to_rpred
    Rpred_to_Zpred = context.rpred_to_zpred

    Rel_Dict: dict[tuple[Cell, Cell], list[frozenset[AtomicFormula]]] = context.rel_dict
    
    if args.print_model:
        PRINT_MODEL = True
        # output the mapping between index and cell
        print('The compatible 1-types:')
        for idx, cell in enumerate(original_cells):
            print(f' {idx}: {cell}')
        # output the mapping between index and relation of two cells
        print('The compatible 2-types:')
        for k,v in Rel_Dict.items():
            print(f' {k}:')
            for idx, rel in enumerate(v):
                rel = '^'.join([str(atom) for atom in list(remove_aux_atoms(rel))])
                print(f'   {idx}: {rel}')
        print()

    x_preds = context.x_preds
    xpreds_with_P = context.xpreds_with_P
    
    Evi_to_Xpred = context.Evi_to_Xpred
    Xpred_to_Evi = context.Xpred_to_Evi

    auxiliary_uni_formula_cell_graph = context.auxiliary_uni_formula_cell_graph
    auxiliary_cells: list[Cell] = auxiliary_uni_formula_cell_graph.cells

    Config_SAT_pre = ConfigSAT(context.auxiliary_uni_formula,
                           context.auxiliary_ext_formulas,
                           auxiliary_cells,
                           DELTA)
    
    Config_SAT_unary = ConfigSAT(context.original_uni_formula,
                                 context.original_ext_formulas,
                                 context.original_cells,
                                 DELTA, False)

    with Timer() as t_preprocess:
        if False :
            pass
        else: 
            Config_SAT_pre.construct_config_cache(auxiliary_uni_formula_cell_graph, xpreds_with_P, DOMAIN_SIZE)
    preprocess_time = t_preprocess.elapsed

    logger.info('time: %s', preprocess_time)

    with Timer() as t_enumaration:
        try:
            # enumerate satisfiable configs
            ori_len_config = len(context.original_cells)
            ori_delta = context.delta
            for cfg_class in generate_config_class(DOMAIN_SIZE, ori_len_config, ori_delta):
                for base_config in generate_base_configs(ori_len_config, cfg_class[0], cfg_class[1], ori_delta):
                    for tpl_config in generate_template_config(base_config, False, ori_delta, DOMAIN_SIZE):
                        for config in generate_sat_configs(DOMAIN_SIZE, tpl_config, ori_delta):
                            logger.info('The configuration: %s', config)
                            domain_order = get_domain_order(original_cells,
                                                          original_original_cell_correlation,
                                                          config)
                            # init cardinality constraint for each X pred
                            cc_xpred: dict[Pred, int] = {x: 0 for x in x_preds}
                            for cell, num in zip(original_cells, config):
                                cc_xpred[Evi_to_Xpred[context.init_evi_dict[cell]]] += num
                            CACHE_OF_MODELS = []
                            firt_partition:list = None
                            # assign 1-types for all elements
                            for domain_partition in cell_assignment(context.domain, config):
                                if firt_partition != None:
                                    substitute_dict:dict = {}
                                    for first_par, cur_par in zip(firt_partition, domain_partition):
                                        set_1 = set(first_par)
                                        set_2 = set(cur_par)
                                        s1 = set_1 - set_2
                                        s2 = set_2 - set_1
                                        while len(s1) != 0:
                                            substitute_dict[s1.pop()] = s2.pop()

                                    for model in CACHE_OF_MODELS:
                                        new_model = model.copy()
                                        for k, v in substitute_dict.items():
                                            # continue
                                            # row_temp = new_model[k, :].copy()
                                            # new_model[k, :] = new_model[v, :]
                                            # new_model[v, :] = row_temp

                                            # col_temp = new_model[:, k].copy()
                                            # new_model[:, k] = new_model[:, v]
                                            # new_model[:, v] = col_temp
                                            swap_rows_cols(new_model, k, v)
                                        MODEL_COUNT += 1
                                        if PRINT_MODEL:
                                            print(f'Model {MODEL_COUNT}:')
                                            print(new_model)
                                        if EARLY_STOP and MODEL_COUNT > MC_LIMIT:
                                            raise ExitRecursion
                                    continue
                                firt_partition = domain_partition
                                logger.debug('The distribution:')
                                # init evidence set (1-type, block type and negative A)
                                evidence_dict: dict[Const, Pred] = {}
                                for cell, elements in zip(original_cells, domain_partition):
                                    # logger.info(cell, " = ",elements)
                                    for element in elements:
                                        Domain_to_Cell[element] = cell
                                        evidence_dict[element] = Evi_to_Xpred[context.init_evi_dict[cell]]
                                logger.debug('The init evidence: \n%s', evidence_dict)
                                domain_list = get_domain_list(domain_order, domain_partition)
                                init_model = np.full((DOMAIN_SIZE, DOMAIN_SIZE), -1, dtype=np.int8)
                                for idx, element in enumerate(domain_list):
                                    init_model[idx, idx] = original_cells.index(Domain_to_Cell[element])
                                domain_recursion(domain_list, evidence_dict, cc_xpred.copy(), init_model)
                                
        except ExitRecursion:
            logger.info('Early stop, current MC: %s', MODEL_COUNT)
            pass

    enumeration_time = t_enumaration.elapsed
    logger.info('The number of models: %s', MODEL_COUNT)
    logger.info('time: %s', enumeration_time)

    Config_SAT_pre.solver.delete()
    num_meta_cc = 0
    for k, v in Config_SAT_pre.META_CONFIG_DICT.items():
        num_meta_cc += len(v)

    if MODEL_COUNT != 0:
        avg_time = enumeration_time / MODEL_COUNT
    else:
        avg_time = -1
    
    print()
    res = f'domain size: {DOMAIN_SIZE},\npreprocess time(s): {preprocess_time},\nenumeration time(s): {enumeration_time},\nnum of meta configurations: {num_meta_cc},\nnum of enumerated models: {MODEL_COUNT}'
    print(res)