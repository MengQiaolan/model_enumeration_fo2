from __future__ import annotations
from logzero import logger
from itertools import combinations, product
from copy import deepcopy

from wfomc.fol.syntax import *
from wfomc.problems import WFOMCProblem
from wfomc.fol.sc2 import SC2, to_sc2
from wfomc.fol.utils import new_predicate, exactly_one_qf
from wfomc.enum_utils import ENUM_R_PRED_NAME, ENUM_Z_PRED_NAME, ENUM_T_PRED_NAME, \
    ENUM_X_PRED_NAME, SKOLEM_PRED_NAME, build_two_tables, Pred_A, Pred_D, Pred_P
from wfomc.utils.third_typing import RingElement, Rational
from wfomc.cell_graph.cell_graph import CellGraph, Cell

def skolemize_one_formula(formula: QuantifiedFormula, 
                          weights: dict[Pred, tuple[Rational, Rational]]) -> tuple[QFFormula, set[Pred]]:
    quantified_formula = formula.quantified_formula
    quantifier_num = 1
    while(not isinstance(quantified_formula, QFFormula)):
        quantified_formula = quantified_formula.quantified_formula
        quantifier_num += 1

    if quantifier_num == 2:
        skolem_pred = new_predicate(1, SKOLEM_PRED_NAME)
        skolem_atom = skolem_pred(X)
    elif quantifier_num == 1:
        skolem_pred = new_predicate(0, SKOLEM_PRED_NAME)
        skolem_atom = skolem_pred()
    weights[skolem_pred] = (Rational(1, 1), Rational(-1, 1))
    return (skolem_atom | ~ quantified_formula)

def skolemize_sentence(uni_formula: QFFormula, ext_formulas: list[QuantifiedFormula], 
                       weights: dict[Pred, tuple[Rational, Rational]]) -> QFFormula:
    skolem_formula = uni_formula
    while(not isinstance(skolem_formula, QFFormula)):
        skolem_formula = skolem_formula.quantified_formula
    
    for ext_formula in ext_formulas:
        skolem_formula = skolem_formula \
                                & skolemize_one_formula(ext_formula, weights)
    return skolem_formula


class EnumContext(object):
    """
    Context for enumeration
    """
    def __init__(self, problem: WFOMCProblem):
        self.domain: set[Const] = problem.domain
        self.sentence: SC2 = problem.sentence
        self.weights: dict[Pred, tuple[Rational, Rational]] = problem.weights
        
        self.domain_: set[int] = set()
        for e in self.domain:
            self.domain_.add(int(e.name[1:]))
        self.domain = self.domain_
        
        # build self.original_uni_formula and self.original_ext_formulas
        self.original_uni_formula: QFFormula = self.sentence.uni_formula
        self.original_ext_formulas: list[QuantifiedFormula] = []
        self._scott()
        
        self._m = len(self.original_ext_formulas)
        # self.delta = max(2*self._m+1, self._m*(self._m+1)) if self._m > 0 else 1
        self.delta = self._m*(self._m+1) if self._m > 0 else 1
        
        self.original_cell_graph: CellGraph = CellGraph(self.original_uni_formula, self.get_weight)
        self.original_cells: list[Cell] = self.original_cell_graph.cells
        self.oricell_to_onetype_pred: dict[Cell, Pred] = {cell: new_predicate(1, ENUM_T_PRED_NAME) for cell in self.original_cells}
        
        
        self.original_cell_correlation: dict[tuple[Cell, Cell], int] = dict()
        for i, cell_i in enumerate(self.original_cells):
            for j, cell_j in enumerate(self.original_cells):
                if i > j:
                    continue
                self.original_cell_correlation[(cell_i, cell_j)] = len(self.original_cell_graph.two_tables[(cell_i, cell_j)].models)
                self.original_cell_correlation[(cell_j, cell_i)] = len(self.original_cell_graph.two_tables[(cell_i, cell_j)].models)
    
        
        # ===================== Skolemize and Introduce @T predicates =====================
        # each atom of @T/1 equals to a cell formula
        # neccessary to calculate template configs
        
        self.original_skolem_formula: QFFormula = skolemize_sentence(self.original_uni_formula, 
                                                                     self.original_ext_formulas, 
                                                                     self.weights)
        
        for cell, onetype_pred in self.oricell_to_onetype_pred.items():
            new_atom = onetype_pred(X)
            new_formula = top
            for atom in cell.get_evidences(X):
                new_formula = new_formula & atom
            new_formula = new_formula.equivalent(new_atom)
            self.original_skolem_formula = self.original_skolem_formula & new_formula
        self.original_skolem_cell_graph = CellGraph(self.original_skolem_formula, self.get_weight)
        
        # ===================== Introduce @D predicates  =====================
        
        self.auxiliary_uni_formula: QFFormula = self.original_uni_formula & AtomicFormula(Pred_D, (X,X), True) # TODO
        
        # ===================== Introduce @Z predicates (block type) =====================
        
        self.z_preds: list[Pred] = []
        self.zpred_to_rpred: dict[Pred, Pred] = {}
        self.rpred_to_zpred: dict[Pred, Pred] = {}
        self.auxiliary_ext_formulas: list[QuantifiedFormula] = []
        
        ext_formulas = self.original_ext_formulas
        for ext_formula in ext_formulas:
            new_pred = new_predicate(1, ENUM_Z_PRED_NAME)
            self.z_preds.append(new_pred)
            self.zpred_to_rpred[new_pred] = ext_formula.quantified_formula.quantified_formula.pred
            self.rpred_to_zpred[self.zpred_to_rpred[new_pred]] = new_pred
            new_atom = new_pred(X)
            quantified_formula = ext_formula.quantified_formula.quantified_formula
            ext_formula.quantified_formula.quantified_formula = new_atom.implies(quantified_formula & Pred_D(X,Y))
            self.auxiliary_ext_formulas.append(ext_formula)
        logger.info('The existential formulas after introducing blocks: \n%s', self.auxiliary_ext_formulas)
        logger.info('The map from tseitin predicates Zi to existential predicates: \n%s', self.zpred_to_rpred)
        
        
        # init evidence set including 1-type, block type and the predicate A
        self.init_evi_dict: dict[Cell, frozenset[AtomicFormula]] = {}
        for cell in self.original_cells:
            evi = set(cell.get_evidences(X)|{~Pred_A(X)}|{~Pred_P(X)})
            for z in self.z_preds:
                evi.add(~z(X) if cell.is_positive(self.zpred_to_rpred[z]) else z(X))
            self.init_evi_dict[cell] = frozenset(evi)
        logger.info('The initial evidence for each cell: ')
        for cell, evi in self.init_evi_dict.items():
            logger.info('%s: %s', cell, evi)
        
        # ===================== Introduce @P predicates (relation) =====================
        # ============== (neccessary to convert binary-evi to unary-evi) ===============
        
        self.relations: list[frozenset[AtomicFormula]] = None
        self.rel_dict: dict[tuple[Cell, Cell], list[frozenset[AtomicFormula]]] = {}
        self.relations, self.rel_dict = build_two_tables(self.original_uni_formula, self.original_cells)
        # self.relations is just for debug
        
        rel_formula = Implication(Pred_A(X) & Pred_P(Y), ~Pred_D(X,Y) & ~Pred_D(Y,X))
        rel_formula = rel_formula & Implication(~(Pred_A(X) & Pred_P(Y) | Pred_A(Y) & Pred_P(X)), Pred_D(X,Y) & Pred_D(Y,X))
        logger.info('The new formula for predicates A and P: %s', rel_formula)
        self.auxiliary_uni_formula = self.auxiliary_uni_formula & to_sc2(rel_formula).uni_formula
        logger.info('The uni formula after adding A&P formulas: \n%s', self.auxiliary_uni_formula)
        
        # ===================== Introduce @X predicates (evidence) =====================
        # ======================= (neccessary to evidence type) ========================
        
        self.x_preds: list[Pred] = list()
        self.evi_formulas: list[QFFormula] = []
        self.xpreds_with_P: set[Pred] = set()
        self.Evi_to_Xpred: dict[frozenset[AtomicFormula], Pred] = {}
        self.Xpred_to_Evi: dict[Pred, frozenset[AtomicFormula]] = {}
        self._introduce_evi_formulas()
        
        
        self.auxiliary_uni_formula_cell_graph: CellGraph = CellGraph(self.auxiliary_uni_formula, self.get_weight)
        self.auxiliary_formula_cells: list[Cell] = self.auxiliary_uni_formula_cell_graph.cells
        
        
        # =================== Skolemize and Introduce @T predicates ===================
        # =================== (neccessary to calculate meta configs) ==================
        
        # build balabala...
        # self.auxcell_to_onetype_pred: dict[Cell, Pred] = \
        #                             {cell: new_predicate(1, ENUM_T_PRED_NAME) 
        #                                     for cell in self.auxiliary_formula_cells}
        # self.skolem_formula_DAPZXT = self.auxiliary_uni_formula
        
        # for cell, tau in self.auxcell_to_onetype_pred.items():
        #     new_atom = tau(X)
        #     new_formula = top
        #     for atom in cell.get_evidences(X):
        #             new_formula = new_formula & atom
        #     new_formula = new_formula.equivalent(new_atom)
        #     self.skolem_formula_DAPZXT = self.skolem_formula_DAPZXT & new_formula
            
        # for ext_formula in self.auxiliary_ext_formulas:
        #     self.skolem_formula_DAPZXT = self.skolem_formula_DAPZXT \
        #                             & skolemize_one_formula(ext_formula, self.weights)
            





    def contain_existential_quantifier(self) -> bool:
        return self.sentence.contain_existential_quantifier()

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return (default, default)

    def _scott(self):
        while(not isinstance(self.original_uni_formula, QFFormula)):
            self.original_uni_formula = self.original_uni_formula.quantified_formula

        for formula in self.sentence.ext_formulas:
            quantified_formula = formula.quantified_formula.quantified_formula
            new_pred = new_predicate(2, ENUM_R_PRED_NAME)
            new_atom = new_pred(X,Y)
            formula.quantified_formula.quantified_formula = new_atom
            self.original_ext_formulas.append(formula)
            self.original_uni_formula = self.original_uni_formula.__and__(new_atom.equivalent(quantified_formula))

        logger.info('The universal formula: \n%s', self.original_uni_formula)
        logger.info('The existential formulas: \n%s', self.original_ext_formulas)
    
    def _introduce_evi_formulas(self):
        # here we need to consider all possible combinations of Zi predicates
        z_lit_combs: list[frozenset[AtomicFormula]] = []
        codes = list(product([False, True], repeat=len(self.z_preds)))
        for code in codes:
            comb = frozenset(z(X) if code[self.z_preds.index(z)] else ~z(X) for z in self.z_preds)
            z_lit_combs.append(comb)
        
        # X preds and evi formulas
        all_evi: list[set[AtomicFormula]] = []
        evi_formulas: list[QFFormula] = []
        for cell in self.original_cells:
            cell_atoms: set[AtomicFormula] = set(cell.get_evidences(X))
            # we do not need to consider all comb of tau and Z
            # some Z are not neccessary when there are some tau
            cell_z_lit_combs = []
            for z_lit_comb in z_lit_combs:
                r_lits = set(self.zpred_to_rpred[z.pred](X,X) for z in z_lit_comb if z.positive)
                if len(r_lits) != 0 and len(r_lits & cell_atoms) != 0:
                    logger.info('Impossible evidence type: %s', cell_atoms | z_lit_comb)
                else:
                    cell_z_lit_combs.append(cell_atoms | z_lit_comb)
            
            add_A = [{Pred_A(X), ~Pred_P(X)}|s for s in cell_z_lit_combs]
            # For an element e, each Pi(e) is determined by A(e) and Cell(e).
            # So we do not need to consider the case of Pi(e) when A(e) is true
            add_negA = [{~Pred_A(X)}|s for s in cell_z_lit_combs]
            # we do not need to consider all comb of Z and P
            # some Z are not neccessary when there are some P 
            add_negA_posP = [{Pred_P(X)}|s for s in add_negA]
            add_negA_negP = [{~Pred_P(X)}|s for s in add_negA]
            all_evi = all_evi + add_A + add_negA_posP + add_negA_negP
        
        
        def sort(evi):
            if Pred_A(X) in evi:
                return (1,0)
            for atom in evi:
                if atom.pred == Pred_P and atom.positive:
                    return (0,1)
            return (0,0)
        
        all_evi.sort(key=sort, reverse=True)

        for atom_set in all_evi:
            new_x_pred = new_predicate(1, ENUM_X_PRED_NAME)
            self.Xpred_to_Evi[new_x_pred] = frozenset(atom_set)
            self.Evi_to_Xpred[frozenset(atom_set)] = new_x_pred
            self.x_preds.append(new_x_pred)
            if Pred_P(X) in atom_set:
                self.xpreds_with_P.add(new_x_pred)
            new_atom = new_x_pred(X)
            
            evidence_type = top
            for atom in atom_set:
                evidence_type = evidence_type & atom
            evi_formulas.append(new_atom.implies(evidence_type))
            
        for evi_formula in evi_formulas:
            logger.info(' %s', evi_formula)
            self.auxiliary_uni_formula = self.auxiliary_uni_formula & evi_formula
        self.auxiliary_uni_formula = self.auxiliary_uni_formula & exactly_one_qf(self.x_preds)