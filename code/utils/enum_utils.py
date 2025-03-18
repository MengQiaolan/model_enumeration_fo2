from __future__ import annotations
from collections import defaultdict

from logzero import logger
from wfomc.fol.sc2 import SC2, to_sc2
from wfomc.fol.utils import new_predicate, convert_counting_formula

from wfomc.network.constraint import CardinalityConstraint
from wfomc.fol.syntax import *
from wfomc.problems import WFOMCProblem
from wfomc.fol.syntax import SKOLEM_PRED_NAME
from wfomc.utils.third_typing import RingElement, Rational

from wfomc.context import WFOMCContext
from wfomc.cell_graph.cell_graph import CellGraph, Cell, OptimizedCellGraphWithPC
from wfomc.network.constraint import PartitionConstraint

from functools import reduce
from wfomc.fol.syntax import AtomicFormula

from collections import defaultdict
from itertools import product

from contexttimer import Timer

from wfomc.utils import multinomial, Rational

from wfomc.cell_graph.utils import conditional_on

from wfomc.cell_graph.cell_graph import build_cell_graphs
from wfomc.network.constraint import PartitionConstraint
from wfomc.utils import MultinomialCoefficients, multinomial_less_than, RingElement, Rational
from wfomc.fol.syntax import Const, Pred, QFFormula

from wfomc.fol.syntax import AtomicFormula, Const, Pred, QFFormula, a, b, c

Pred_A = Pred("@A", 1)
Pred_P = Pred("@P", 1)
Pred_D = Pred("@D", 2)

ENUM_R_PRED_NAME = '@R'
ENUM_Z_PRED_NAME = '@Z'
ENUM_P_PRED_NAME = '@P'
ENUM_X_PRED_NAME = '@X'
ENUM_T_PRED_NAME = '@T'

def build_two_tables(formula: QFFormula, cells: list[Cell]) -> tuple[list[frozenset[AtomicFormula]], dict]:
    models = dict()
    gnd_formula: QFFormula = ground_on_tuple(formula, a, b) & ground_on_tuple(formula, b, a)
    gnd_formula = gnd_formula.substitute({a: X, b: Y})
    gnd_lits = gnd_formula.atoms()
    gnd_lits = gnd_lits.union(frozenset(map(lambda x: ~x, gnd_lits)))
    for model in gnd_formula.models():
        models[model] = 1

    two_tables: dict[tuple[Cell, Cell], list[frozenset[AtomicFormula]]] = dict()
    for i, cell in enumerate(cells):
        models_1 = conditional_on(models, gnd_lits, cell.get_evidences(X))
        for j, other_cell in enumerate(cells):
            models_2 = conditional_on(models_1, gnd_lits, other_cell.get_evidences(Y))
            two_tables[(cell, other_cell)] = []
            
            models_3 = list(models_2.keys())
            def sort(rel):
                r_num = 0
                for atom in rel:
                    if atom.pred.name.startswith(ENUM_R_PRED_NAME):
                        r_num += 1
                res = [0]*r_num
                for atom in rel:
                    if atom.pred.name.startswith(ENUM_R_PRED_NAME) and atom.args[0] != atom.args[1] and atom.positive:
                        res[int(atom.pred.name[2:])] = 1
                return tuple(res)
            models_3.sort(key=sort, reverse=True)
            for model in models_3:
                two_tables[(cell, other_cell)].append(frozenset({atom for atom in model if len(atom.args) == 2 and atom.args[0] != atom.args[1]}))
      
    return list(models.keys()), two_tables

def ground_on_tuple(formula: QFFormula, c1: Const, c2: Const = None) -> QFFormula:
        variables = formula.vars()
        if len(variables) > 2:
            raise RuntimeError(
                "Can only ground out FO2"
            )
        if len(variables) == 1:
            constants = [c1]
        else:
            if c2 is not None:
                constants = [c1, c2]
            else:
                constants = [c1, c1]
        substitution = dict(zip(variables, constants))
        gnd_formula = formula.substitute(substitution)
        return gnd_formula

def remove_aux_atoms(atoms: set[AtomicFormula]) -> set[AtomicFormula]:
    return set(filter(lambda atom: not atom.pred.name.startswith('@'), atoms))

def fast_wfomc_with_pc(opt_cell_graph_pc: OptimizedCellGraphWithPC, 
                       partition_constraint: PartitionConstraint) -> RingElement:
   
    cliques = opt_cell_graph_pc.cliques
    nonind = opt_cell_graph_pc.nonind
    nonind_map = opt_cell_graph_pc.nonind_map

    pred_partitions: list[list[int]] = list(num for _, num in partition_constraint.partition)
    # partition to cliques
    partition_cliques: dict[int, list[int]] = opt_cell_graph_pc.partition_cliques

    res = Rational(0, 1)
    with Timer() as t:
        for configs in product(
            *(list(multinomial_less_than(len(partition_cliques[idx]), constrained_num)) for
                idx, constrained_num in enumerate(pred_partitions))
        ):
            coef = Rational(1, 1)
            remainings = list()
            # config for the cliques
            overall_config = list(0 for _ in range(len(cliques)))
            # {clique_idx: [number of elements of pred1, pred2, ..., predk]}
            clique_configs = defaultdict(list)
            for idx, (constrained_num, config) in enumerate(zip(pred_partitions, configs)):
                remainings.append(constrained_num - sum(config))
                mu = tuple(config) + (constrained_num - sum(config), )
                coef = coef * MultinomialCoefficients.coef(mu)
                for num, clique_idx in zip(config, partition_cliques[idx]):
                    overall_config[clique_idx] = overall_config[clique_idx] + num
                    clique_configs[clique_idx].append(num)

            body = opt_cell_graph_pc.get_i1_weight(
                remainings, overall_config
            )

            for i, clique1 in enumerate(cliques):
                for j, clique2 in enumerate(cliques):
                    if i in nonind and j in nonind:
                        if i < j:
                            body = body * opt_cell_graph_pc.get_two_table_weight(
                                (clique1[0], clique2[0])
                            ) ** (overall_config[nonind_map[i]] *
                                    overall_config[nonind_map[j]])

            for l in nonind:
                body = body * opt_cell_graph_pc.get_J_term(
                    l, tuple(clique_configs[nonind_map[l]])
                )
            res = res + coef * body
    return res