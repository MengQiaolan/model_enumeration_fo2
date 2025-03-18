import functools

from pysat.formula import Atom, Formula
from pysat.solvers import Solver

from itertools import combinations

from wfomc.cell_graph.cell_graph import CellGraph
from wfomc.cell_graph.components import Cell
from wfomc.fol.syntax import X, Y, Pred, QFFormula, QuantifiedFormula, a, b
from wfomc.fol.utils import new_predicate

from wfomc.enum_utils import Pred_A

def get_init_configs(cell_graph: CellGraph, num_exts: int, 
                     A_idx: list[int], P_idx: list[int], 
                     domain_size: int) -> dict[int, list[tuple[int, ...]]]:
    
    cells = cell_graph.get_cells()
    num_cells = len(cells)
    
    # compatible_dict[(cell_idx, R_idx)] is a set of cell indices 
    # that can satisfy the R_idx-th predicate of the cell_idx-th cell
    compatible_dict: dict[tuple[int, int], set[int]] = dict()
    for i in range(num_cells):
        for j in range(num_exts):
            compatible_dict[(i, j)] = set()
    
    # fill sat_dict
    for i, cell_i in enumerate(cells):
        for pred in cell_i.preds:
            if pred.name.startswith('@R') and cell_i.is_positive(pred):
                compatible_dict[(i, int(pred.name[2:]))].add(i)
        for j, cell_j in enumerate(cells):
            if j > i:
                continue
            for rel, weight in cell_graph.get_two_tables((cell_i, cell_j)):
                if weight > 0:
                    for atom in rel:
                        pred_name = atom.pred.name
                        if pred_name.startswith('@R') and atom.positive:
                            if atom.args[0] == a:
                                compatible_dict[(i, int(pred_name[2:]))].add(j)
                            else:
                                compatible_dict[(j, int(pred_name[2:]))].add(i)
    
    A_prefix_len = len(A_idx)
    remaining = num_cells - A_prefix_len
    suffixes: list[tuple[int, ...]] = []
    for k in range(1, domain_size):
        if k > remaining:
            break
        for indices in combinations(range(remaining), k):
            l = [0] * remaining
            for index in indices:
                l[index] = 1
            # there is at least one element whose evidence contain '@P(X)'
            if sum([l[i-A_prefix_len] for i in P_idx]) == 0:
                continue
            suffixes.append(tuple(l))
    
    res = []
    for true_loc in range(A_prefix_len):
        prefix = (0,) * true_loc + (1,) + (0,) * (A_prefix_len - true_loc - 1)
        for suffix in suffixes:
            init_config = prefix + suffix
            cell_set = set([index for index, value in enumerate(init_config) if value != 0])
            flag = True # record if the init_config is possible sat
            for i in range(len(init_config)):
                if init_config[i] == 0:
                    continue
                for r in range(num_exts):
                    need_check = False
                    for pred in cells[i].preds:
                        if pred.name.startswith('@Z') and int(pred.name[2:]) == r:
                            if cells[i].is_positive(pred):
                                need_check = True
                    if need_check:
                        # if there is no cell in cell_set that can satisfy predicate r of i-th cell
                        if len(compatible_dict[(i, r)] & cell_set) == 0:
                            flag = False
                            break
                if not flag:
                    break
            if flag:
                res.append(init_config)
    return res

class ConfigSAT(object):
    def __init__(self, uni_formula: QFFormula,
                 ext_formulas: list[QuantifiedFormula],
                 cells: list[Cell], delta: int, for_precessing: bool = True):
        self.uni_formula = uni_formula
        self.ext_formulas = ext_formulas
        self.cells = cells
        self.delta = delta
        if for_precessing:    
            self.META_CONFIG_DICT: dict[int, set[tuple[int]]] = dict()
            for i, cell in enumerate(cells):
                if cell.is_positive(Pred_A):
                    self.META_CONFIG_DICT[i] = set()
        else:
            self.META_CONFIG_DICT = None        
        
        self.ext_preds = []
        for ext_formula in self.ext_formulas:
            aux_pred = new_predicate(2, '@AUX')
            self.ext_preds.append(aux_pred)
            self.uni_formula = self.uni_formula & (
                ext_formula.quantified_formula.quantified_formula
                .equivalent(aux_pred(X, Y))
            )
        self.cell_graph = CellGraph(self.uni_formula, lambda _: (1, 1))
        self.aligned_cells = list()
        for target_cell in self.cells:
            for cell in self.cell_graph.cells:
                if all(
                    target_cell.is_positive(pred) == cell.is_positive(pred)
                    for pred in target_cell.preds
                ):
                    self.aligned_cells.append(cell)
                    break
        self.clauses = self.build_clauses()
        self.solver = Solver(name='mpl')
        for clause in self.clauses:
            self.solver.add_clause(clause)
        
    def __str__(self) -> str:
        str = 'ConfigSAT\n'
        str += f'cells:\n'
        for cell in self.cell_graph.cells:
            str += f'{cell}\n'
        str += f'delta: {self.delta}\n'
        str += f'ext_preds: {self.ext_preds}\n'
        return str

    @functools.lru_cache(maxsize=None)
    def check_config_by_cache(self, config: tuple[int]) -> bool:
        for meta_cc in self.META_CONFIG_DICT[config.index(1)]:
            if all((a >= b and b > 0) or (a == b and b == 0) for a, b in zip(config, meta_cc)):
                return True
        return False
  
    def check_config_by_pysat(self, config: tuple[int]) -> bool:
        assumptions = []
        for i, num in enumerate(config):
            atom_id = Formula._vpool[Formula._context].obj2id[Atom(f'i{i}_{num}')]
            assumptions.append(atom_id)
        ret = self.solver.solve(assumptions=assumptions)
        return ret
    
    def build_clauses(self):
        cell_graph = self.cell_graph
        cells = self.aligned_cells
        delta = self.delta
        ext_preds = self.ext_preds
        clauses = None
        for i, _ in enumerate(cells):
            for j in range(delta):
                for k in range(j):
                    if clauses is None:
                        clauses = (~Atom(f'c{i}_{j}') | Atom(f'c{i}_{k}'))
                    else:
                        clauses = clauses & (~Atom(f'c{i}_{j}') | Atom(f'c{i}_{k}'))

        for i in range(len(cells)):
            for j in range(delta + 1):
                i_atom = Atom(f'i{i}_{j}') # exactly j elements in cell i
                if j == 0:
                    clauses = clauses & (~i_atom | ~Atom(f'c{i}_0'))
                    clauses = clauses & (i_atom | Atom(f'c{i}_0'))
                else:
                    clauses = clauses & (~i_atom | Atom(f'c{i}_{j-1}'))
                    if j < delta:
                        clauses = clauses & (~i_atom | ~Atom(f'c{i}_{j}'))
            # clauses = clauses & functools.reduce(
            #     lambda x, y: x | y,
            #     [Atom(f'i{i}_{j}') for j in range(delta + 1)]
            # )

        clause = clauses.simplified()

        for i in range(len(cells)):
            cell1 = cells[i]
            for j in range(len(cells)):
                if i > j:
                    continue
                cell2 = cells[j]
                two_tables = [
                    t for t, _ in
                    cell_graph.get_two_tables((cell1, cell2))
                ]
                for k1 in range(delta):
                    for k2 in range(delta):
                        clause = None
                        if i == j:
                            if k1 == k2:
                                tmp = functools.reduce(
                                    lambda x, y: x & y,
                                    (
                                        Atom(f'{pred.name}_{i}_{k1}_{i}_{k2}')
                                        if cell1.is_positive(pred)
                                        else ~Atom(f'{pred.name}_{i}_{k1}_{i}_{k2}')
                                        for pred in ext_preds
                                    )
                                )
                                clause = tmp if clause is None else clause & tmp
                                clauses = clauses & clause.simplified()
                                continue
                            if k1 > k2:
                                continue
                        for table in two_tables:
                            tmp = functools.reduce(
                                lambda x, y: x & y,
                                (
                                    (
                                        Atom(f'{pred.name}_{i}_{k1}_{j}_{k2}')
                                        if pred(a, b) in table
                                        else ~Atom(f'{pred.name}_{i}_{k1}_{j}_{k2}')
                                    ) & (
                                        Atom(f'{pred.name}_{j}_{k2}_{i}_{k1}')
                                        if pred(b, a) in table
                                        else ~Atom(f'{pred.name}_{j}_{k2}_{i}_{k1}')
                                    )
                                    for pred in ext_preds
                                )
                            )
                            clause = tmp if clause is None else clause | tmp
                        clauses = clauses & clause.simplified()

        # encode the exitensial quantifier
        for i in range(len(cells)):
            cell1 = cells[i]
            for pred in ext_preds:
                for k1 in range(delta):
                    ext_clause = None
                    for j in range(len(cells)):
                        cell2 = cells[j]
                        for k2 in range(delta):
                            if ext_clause is None:
                                ext_clause = Atom(f'c{j}_{k2}') & Atom(f'{pred.name}_{i}_{k1}_{j}_{k2}')
                            else:
                                ext_clause = ext_clause | (
                                    Atom(f'c{j}_{k2}') & Atom(f'{pred.name}_{i}_{k1}_{j}_{k2}')
                                )
                    ext_clause = (~Atom(f'c{i}_{k1}') | ext_clause)
                    ext_clause = ext_clause.simplified()
                    clauses = clauses & ext_clause

        clauses.clausify()
        return clauses


    def construct_config_cache(self, cell_graph, xpreds_with_P, domain_size: int):
        # we only consider the case that:
        # 1) there is only one element whose evidence contain '@A(X)'
        A_idx = [idx for idx, cell in enumerate(self.cells) if cell.is_positive(Pred_A)]
        # 2) there is at least one element whose evidence contain '@P(X)'
        P_idx = [idx for x_pred in xpreds_with_P
                        for idx, cell in enumerate(self.cells) if cell.is_positive(x_pred)]

        # find all possible initial configurations that satisfy above constraints
        init_configs = get_init_configs(cell_graph, self.delta,
                                        A_idx, P_idx, domain_size)
        for init_config in init_configs:
            init_holds = [True if (j in A_idx or init_config[j] == 0) else False
                            for j in range(len(self.cells))]
            # a meta config is based on the auxiliary sentence
            self.calculate_meta_config(tuple(init_config), tuple(init_holds), domain_size)

    @functools.lru_cache(maxsize=None)
    def calculate_meta_config(self, config: tuple, hold: tuple, domain_size: int):
        idx_target_element = config.index(1)
        if sum(config) > domain_size:
            return
        if any(all((a >= b and b > 0) or (a == b and b == 0) 
                        for a, b in zip(config, cur_meta_config)) 
                            for cur_meta_config in self.META_CONFIG_DICT[idx_target_element]):
            return
        if self.check_config_by_pysat(config):
            self.META_CONFIG_DICT[idx_target_element].add(config)
            return
        if sum(config) == domain_size:
            return

        hold = list(hold)
        # hold[i]==True means that we don't need to increase the i-th element
        # to get a new increamented config,
        # since it has been increased to the upper bound
        # or the new incremented config can be derived from a meta_cfg.
        # each hold[i] can only change from False to True.
        for i in range(len(config)):
            if hold[i]:
                # we don't consider to increase the i-th element if hold[i] is True
                continue
            # the four cases that hold[i] should be True
            if config[i] == self.delta:
                # 1) the i-th element has been increased to the upper bound
                hold[i] = True
                continue
            l = list(config)
            l[i] += 1
            derived_config = tuple(l)
            if derived_config in self.META_CONFIG_DICT[idx_target_element]:
                # 2) the new incremented config has been a meta_cfg
                hold[i] = True
                continue
            if any(all((a >= b and b > 0) or (a == b and b == 0) for a, b in zip(derived_config, meta_cc)) 
                   for meta_cc in self.META_CONFIG_DICT[idx_target_element]):
                # 3) the new incremented config can be derived from a meta_cfg
                hold[i] = True
                continue
            if (not hold[i]) and self.check_config_by_pysat(derived_config): # call sat function as little as possible
                self.META_CONFIG_DICT[idx_target_element].add(derived_config)
                # 4) the new incremented config is a meta_cfg
                hold[i] = True

        for i in range(len(config)):
            # TODO: only consider the 1-types that can provade possible 2-tables
            if not hold[i]:
                l = list(config)
                l[i] += 1
                self.calculate_meta_config(tuple(l), tuple(hold), domain_size)