import typing
from itertools import product, combinations_with_replacement

from loreleai.language.lp import (
    Clause,
    Procedure,
    Predicate,
    Variable,
    Type,
    c_var,
    Disjunction,
    Recursion,
    Not,
    Body,
    Constant,
    Atom
)

INPUT_ARG = 1
OUTPUT_ARG = 2
CONSTANT_ARG = 3


def new_variable(
    existing_variables: typing.Set[Variable], type: Type = None
) -> Variable:
    existing_variable_names = {x.get_name() for x in existing_variables}
    if len(existing_variables) < 27:
        potential_names = [
            chr(x)
            for x in range(ord("A"), ord("Z") + 1)
            if chr(x) not in existing_variable_names
        ]
    else:
        potential_names = [
            f"{chr(x)}{chr(y)}"
            for x in range(ord("A"), ord("Z") + 1)
            for y in range(ord("A"), ord("Z") + 1)
            if f"{chr(x)}{chr(y)}" not in existing_variable_names
        ]

    return c_var(potential_names[0], type)



"""
    Plain extensions adds literals to a clause/body with every possible combination of variables
    (no language bias, all possible combinations)
"""

def _plain_extend_clause(
        clause: typing.Union[Clause, Body],
        predicate: Predicate,
        connected_clause: bool = True
) -> typing.Sequence[typing.Union[Clause, Body]]:
    """
    Extends the clause with the predicate in every possible way (no bias)

    Arguments:
        clause: a clause to be extended
        predicate: a predicate to add to the clause
    """
    if isinstance(clause, Body) and len(clause) == 0:
        if predicate.arity < 3:
            head_variables = [chr(x) for x in range(ord("A"), ord("Z"))][: predicate.get_arity()]
            possible_heads = [
                Body(predicate(*list(x)))
                for x in combinations_with_replacement(head_variables, predicate.get_arity())
            ]

            if predicate.get_arity() == 2:
                possible_heads.append(Body(predicate(head_variables[1], head_variables[0])))
            return possible_heads
        else:
            head_variables = [chr(x) for x in range(ord("A"), ord("Z"))][: predicate.get_arity()]
            possible_heads = []
            possible_heads.append(Body(predicate(head_variables[0], head_variables[1], head_variables[2])))
            possible_heads.append(Body(predicate(head_variables[0], head_variables[2], head_variables[1])))
            possible_heads.append(Body(predicate(head_variables[1], head_variables[0], head_variables[2])))
            possible_heads.append(Body(predicate(head_variables[1], head_variables[2], head_variables[0])))
            possible_heads.append(Body(predicate(head_variables[2], head_variables[1], head_variables[0])))
            possible_heads.append(Body(predicate(head_variables[2], head_variables[0], head_variables[1])))
            return possible_heads

    clause_variables: typing.Sequence[Variable] = clause.get_variables()
    used_variables = {x for x in clause_variables}
    pred_argument_types: typing.Sequence[Type] = predicate.get_arg_types()

    argument_matches = {}
    new_variables = set()

    # create new variable for each argument of a predicate
    for arg_ind in range(len(pred_argument_types)):
        new_var = new_variable(used_variables, pred_argument_types[arg_ind])
        argument_matches[arg_ind] = [new_var]
        used_variables.add(new_var)
        new_variables.add(new_var)

    # check for potential match with other variables
    for clv_ind in range(len(clause_variables)):
        for arg_ind in range(len(pred_argument_types)):
            if clause_variables[clv_ind].get_type() == pred_argument_types[arg_ind]:
                argument_matches[arg_ind].append(clause_variables[clv_ind])

    # do cross product of matches
    base_sets = [argument_matches[x] for x in range(len(pred_argument_types))]
    candidates: typing.List[typing.Union[Clause, Body]] = []

    for arg_combo in product(*base_sets):
        new_clause = None
        if connected_clause and not all(
            [True if x in new_variables else False for x in arg_combo]
        ):
            # check that the new literal is not disconnected from the rest of the clause
            new_clause = clause + predicate(*list(arg_combo))
        elif not connected_clause:
            new_clause = clause + predicate(*list(arg_combo))

        if new_clause is not None:
            candidates.append(new_clause)

    return candidates


def _plain_extend_negation_clause(
    clause: typing.Union[Clause, Body], predicate: Predicate
) -> typing.Sequence[typing.Union[Clause, Body]]:
    """
    Extends a clause with the negation of a predicate (no new variables allowed)
    """
    if isinstance(clause, Body):
        suitable_vars = clause.get_variables()
    else:
        suitable_vars = clause.get_body_variables()
    pred_argument_types: typing.Sequence[Type] = predicate.get_arg_types()
    argument_matches = {}

    # check for potential match with other variables
    for clv_ind in range(len(suitable_vars)):
        for arg_ind in range(len(pred_argument_types)):
            if suitable_vars[clv_ind].get_type() == pred_argument_types[arg_ind]:
                if arg_ind not in argument_matches:
                    argument_matches[arg_ind] = []
                argument_matches[arg_ind].append(suitable_vars[clv_ind])

    base_sets = [argument_matches[x] for x in range(len(pred_argument_types))]
    candidates: typing.List[typing.Union[Clause, Body]] = []

    for arg_combo in product(*base_sets):
        new_clause = clause + Not(predicate(*list(arg_combo)))
        candidates.append(new_clause)

    return candidates

def _instantiate_var_clause(clause: Clause, constant: Constant):
    """
    Returns all clauses generated by substituting a `Variable` in `clause` with `constant`.
    """
    if isinstance(clause, Body):
        suitable_vars = clause.get_variables()
    else:
        suitable_vars = clause.get_body_variables()

    candidates = []
    for var in suitable_vars:
        candidates.append(clause.substitute({var:constant}))
    # if len(candidates) > 0:
    #     print("Type of each is {}", type(candidates[0]))
    return list(set(candidates))


def variable_instantiation(
    clause: typing.Union[Clause,Body,Procedure],
    constant: Constant) -> typing.Sequence[typing.Union[Clause,Body,Procedure]]:
    """
    Extends a clause by instantiation, replacing all occurrences
    of a variable with a constant
    """
    if isinstance(clause, (Clause, Body)):
        return _instantiate_var_clause(clause, constant)
    else:
        clauses = clause.get_clauses()

        # extend each clause individually
        extensions = []
        for cl_ind in range(len(clauses)):
            clause_extensions = (_instantiate_var_clause(clauses[cl_ind], constant))
            for ext_cl_ind in range(len(clause_extensions)):
                cls = [
                    clauses[x] if x != cl_ind else clause_extensions[ext_cl_ind]
                    for x in range(len(clauses))
                ]

                if isinstance(clause, Disjunction):
                    extensions.append(Disjunction(cls))
                else:
                    extensions.append(Recursion(cls))
        print("Extensions for {} are {}".format(clause,extensions))
        return extensions


def plain_extension(
    clause: typing.Union[Clause, Body, Procedure],
    predicate: Predicate,
    connected_clauses: bool = True,
    negated: bool = False,
) -> typing.Sequence[typing.Union[Clause, Body, Procedure]]:
    """
    Extends a clause or a procedure without any bias. Only checks for variable type match.
    Adds the predicate to the clause/procedure
    """
    if isinstance(clause, (Clause, Body)):
        if negated:
            return _plain_extend_negation_clause(clause, predicate)
        else:
            return _plain_extend_clause(
                clause, predicate, connected_clause=connected_clauses
            )
    else:
        clauses = clause.get_clauses()

        # extend each clause individually
        extensions = []
        for cl_ind in range(len(clauses)):
            clause_extensions = (
                _plain_extend_clause(
                    clauses[cl_ind], predicate, connected_clause=connected_clauses
                )
                if not negated
                else _plain_extend_negation_clause(clauses[cl_ind], predicate)
            )
            for ext_cl_ind in range(len(clause_extensions)):
                cls = [
                    clauses[x] if x != cl_ind else clause_extensions[ext_cl_ind]
                    for x in range(len(clauses))
                ]

                if isinstance(clause, Disjunction):
                    extensions.append(Disjunction(cls))
                else:
                    extensions.append(Recursion(cls))

        return extensions

def aleph_extension(
    clause: typing.Union[Clause, Body, Procedure],
    predicate: Predicate,
    allowed_positions_dict,
    constants: typing.Sequence[Constant],
    allowed_reflexivity = []
) -> typing.Sequence[typing.Union[Clause, Body, Procedure]]:
    """ 
    Used to generate new clauses in the Aleph learner. 
    This extension is a mixture of plain clause extension
    and smart instantiation of variables.

    First, all plain extensions are generated. Then, the least Herbrand
    model is constructed given the Task's background information. This model
    contains all grounded facts that can be derived from the background information.
    From this model we can derive position constraints, that constrain where 
    constants are allowed to appear in certain predicates, if at all.

    Next, variable instantiation is applied to all extended clauses. If a resulting
    clauses violates the position constraints, it is immediately pruned.
    """

    # Plain extension
    extensions = plain_extension(clause,predicate)
    subst_extensions = set()

    # Variable instantiation
    for ext in extensions:
        for const in constants:
            subst_extensions = subst_extensions.union(set(variable_instantiation(ext,const)))
    # print(subst_extensions)

    # Prune clauses that violate position constraints
    pruned_clauses1 = set()
    for cl in subst_extensions:
        if _valid_positions(cl,allowed_positions_dict,allowed_reflexivity=allowed_reflexivity):
            pruned_clauses1.add(cl)

    # Prune clauses that have a fully grounded atom
    pruned_clauses2 = set()
    for cl in pruned_clauses1:
        if _not_grounded(cl):
            pruned_clauses2.add(cl)
    
    # print("Pruning {} -> {} -> {}".format(len(extensions),len(subst_extensions),len(pruned_clauses1),len(pruned_clauses2)))
    return list(set(extensions).union(pruned_clauses2))

def _valid_positions(cl: Clause,allowed_positions_dict,allowed_reflexivity=[]):
    """
    Returns True iff the clause `cl` respects the allowed
    positions for constants as given by `allowed_positions_dict`, 
    and is not reflective (e.g. next(X,X)) when disallowed
    """
    for atom in cl.get_literals():
        pred = atom.get_predicate()
        args = atom.get_arguments()
        for i in range(len(args)):
            arg = args[i]
            # Constants must appear at the right places
            if isinstance(arg,Constant):
                if not i in allowed_positions_dict[arg][pred]:
                    return False
        
        # If all arguments are equal, this must be explicitly allowed
        if len(args) > 0 and all(args[i] == args[0] for i in range(len(args))) and pred not in allowed_reflexivity:
                return False
    return True

def _not_grounded(cl):
    for atom in cl.get_literals():
        if len(atom.get_variables()) == 0:
            return False
    return True

                    
        



class BottomClauseExpansion:

    def __init__(self, clause: Clause, only_connected_clauses: bool = False):
        self._clause: Clause = clause
        self._variable_literal_dependency: typing.Dict[Variable, typing.List[typing.Union[Atom, Not]]] = {}
        self._literal_order: typing.Dict[typing.Union[Atom, Not], int] = {}
        self._only_connected_clauses = only_connected_clauses

        self._to_dependency_structure()

    def _to_dependency_structure(self):
        body_lits = self._clause.get_literals()

        for lit_ind in range(len(body_lits)):
            # store order of the literal
            self._literal_order[body_lits[lit_ind]] = len(self._literal_order)

            # compute variable dependency
            vars = body_lits[lit_ind].get_variables()
            for v_ind in range(len(vars)):
                cv = vars[v_ind]

                if cv not in self._variable_literal_dependency:
                    self._variable_literal_dependency[cv] = []

                self._variable_literal_dependency[cv].append(body_lits[lit_ind])

    def _expand_clause(self, clause: typing.Union[Clause, Body]) -> typing.Sequence[typing.Union[Clause, Body]]:
        existing_vars = clause.get_variables()
        used_literals = {x for x in clause.get_literals()}
        last_literal_id = self._literal_order.get(clause.get_literals()[-1], -1) if len(clause) > 0 else -1

        expansions = []

        if self._only_connected_clauses:
            for v_ind in range(len(existing_vars)):
                v = existing_vars[v_ind]
                lits_to_add = [
                    x
                    for x in self._variable_literal_dependency.get(v, [])
                    if x not in used_literals and self._literal_order[x] > last_literal_id
                ]

                for l_ind in range(len(lits_to_add)):
                    expansions.append(clause + lits_to_add[l_ind])
        else:
            expansions = [
                clause + l
                for l in self._clause.get_literals()
                if l not in used_literals and self._literal_order[l] > last_literal_id]

        return expansions

    def expand(self, clause: typing.Union[Clause, Body, Procedure]) -> typing.Sequence[typing.Union[Body, Clause, Procedure]]:
        """
            Expands the clause/byd/procedure by adding literals from the bottom clause
            :param clause:
            :param variable_lit_dependency:
            :return:
        """
        if isinstance(clause, (Body, Clause)):
            return self._expand_clause(clause)
        else:
            clauses = clause.get_clauses()

            # extend each clause individually
            extensions = []
            for cl_ind in range(len(clauses)):
                clause_extensions = self._expand_clause(clauses[cl_ind])

                for ext_cl_ind in range(len(clause_extensions)):
                    cls = [
                        clauses[x] if x != cl_ind else clause_extensions[ext_cl_ind]
                        for x in range(len(clauses))
                    ]

                    if isinstance(clause, Disjunction):
                        extensions.append(Disjunction(cls))
                    else:
                        extensions.append(Recursion(cls))

            return extensions
