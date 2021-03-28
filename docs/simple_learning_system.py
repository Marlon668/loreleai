import typing
from abc import ABC, abstractmethod

from orderedset import OrderedSet

from loreleai.language.lp import c_var, c_pred, Clause, Procedure, Atom, Body, Not, List, c_const, c_functor, c_literal, \
    Structure
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_singleton_vars, has_duplicated_literal,has_duplicated_variable
from loreleai.learning.language_manipulation import plain_extension
from loreleai.learning.task import Task, Knowledge
from loreleai.reasoning.lp.prolog import SWIProlog, Prolog
import csv
from pylo.engines.prolog.SWIProlog import Pair

"""
This is an abstract learner class that defines a learner with the configurable options.

It follows a very simple learning principle: it iteratively
                                                     - searches for a single clause that covers most positive examples
                                                     - adds it to the program
                                                     - removes covered examples
                                                     
It is implemented as a template learner - you still need to provide the following methods:
                                                    - initialisation of the candidate pool (the data structure that keeps all candidates)
                                                    - getting a single candidate from the candidate pool
                                                    - adding candidate(s) to the pool
                                                    - evaluating a single candidate
                                                    - stopping the search for a single clause
                                                    - processing expansion/refinements of clauses
                                                    
The learner does not handle recursions correctly!
"""
class TemplateLearner(ABC):

    def __init__(self, solver_instance: Prolog):
        self._solver = solver_instance
        self._candidate_pool = []

    def _assert_knowledge(self, knowledge: Knowledge):
        """
        Assert knowledge into Prolog engine
        """
        facts = knowledge.get_atoms()
        for f_ind in range(len(facts)):
            self._solver.assertz(facts[f_ind])

        clauses = knowledge.get_clauses()
        for cl_ind in range(len(clauses)):
            self._solver.assertz(clauses[cl_ind])

    def _execute_program(self, examples: Task, clause: Clause) -> typing.Sequence[Atom]:
        """
        Evaluates a clause using the Prolog engine and background knowledge
        Returns a set of atoms that the clause covers
        """

        pos,neg = examples.get_examples()
        self._solver.assertz(clause)
        coverage = []
        for example in pos:
            if self._solver.has_solution(example):
                coverage.append(example)
        self._solver.retract(clause)
        return coverage

    @abstractmethod
    def initialise_pool(self):
        """
        Creates an empty pool of candidates
        """
        raise NotImplementedError()

    @abstractmethod
    def get_from_pool(self) -> Clause:
        """
        Gets a single clause from the pool
        """
        raise NotImplementedError()

    @abstractmethod
    def put_into_pool(self, candidates: typing.Union[Clause, Procedure, typing.Sequence]) -> None:
        """
        Inserts a clause/a set of clauses into the pool
        """
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, examples: Task, clause: Clause) -> typing.Union[int, float]:
        """
        Evaluates a clause  of a task

        Returns a number (the higher the better)
        """
        raise NotImplementedError()

    @abstractmethod
    def stop_inner_search(self, eval: typing.Union[int, float], examples: Task, clause: Clause) -> bool:
        """
        Returns true if the search for a single clause should be stopped
        """
        raise NotImplementedError()

    @abstractmethod
    def process_expansions(self, examples: Task, exps: typing.Sequence[Clause], hypothesis_space: TopDownHypothesisSpace) -> typing.Sequence[Clause]:
        """
        Processes the expansions of a clause
        It can be used to eliminate useless expansions (e.g., the one that have no solution, ...)

        Returns a filtered set of candidates
        """
        raise NotImplementedError()

    def _learn_one_clause(self, examples: Task, hypothesis_space: TopDownHypothesisSpace) -> Clause:
        """
        Learns a single clause

        Returns a clause
        """
        # reset the search space
        hypothesis_space.reset_pointer()

        # empty the pool just in case
        self.initialise_pool()

        # put initial candidates into the pool
        self.put_into_pool(hypothesis_space.get_current_candidate())
        current_cand = None
        score = -100
        self._expansionOneClause = 0
        best = None

        while current_cand is None or (len(self._candidate_pool) > 0) and self._expansionOneClause < 40000:
            # get first candidate from the pool
            current_cand = self.get_from_pool()
            print(self._expansion)

            # expand the candidate
            _ = hypothesis_space.expand(current_cand)
            # this is important: .expand() method returns candidates only the first time it is called;
            #     if the same node is expanded the second time, it returns the empty list
            #     it is safer than to use the .get_successors_of method
            exps = hypothesis_space.get_successors_of(current_cand)
            exps = [cl for cl in exps if len(cl) <= self._max_body_literals]

            new_exps = []
            # check if every clause has solutions
            for cl in exps:
                y = self.evaluate(task, cl)

                if y[0] > 0:
                    new_exps.append(cl)
                    if best == None:
                        best = (y[0],cl)
                    if (y[1]==0):
                        self._expansion += 1
                        self._expansionOneClause += 1
                        self._result.append((self._expansion, y[1]))
                        return cl
                    self._expansion += 1
                    self._expansionOneClause += 1

                    self._result.append((self._expansion, y[0]))
                    if y[0] > best[0]:
                        best = (y[0], cl)
                    else:
                        if (y[0] == best[0]) & (
                                len(cl.get_literals()) < len(best[1].get_literals())):
                            best = (y[0], cl)
                else:
                    hypothesis_space.remove(cl)

            # add into pull
            self.put_into_pool(new_exps)

        print(best)
        return best[1]

    def learn(self, examples: Task, knowledge: Knowledge, hypothesis_space: TopDownHypothesisSpace):
        """
        General learning loop
        """

        self._assert_knowledge(knowledge)
        final_program = []
        examples_to_use = examples
        pos, _ = examples_to_use.get_examples()
        self._result = []
        self._expansion = 0

        while len(final_program) == 0 or (len(pos) > 0):
            # learn na single clause
            cl = self._learn_one_clause(examples_to_use, hypothesis_space)
            final_program.append(cl)

            # update covered positive examples
            covered = self._execute_program(examples_to_use,cl)
            print('covered' , covered)

            pos, neg = examples_to_use.get_examples()
            pos = pos.difference(covered)

            examples_to_use = Task(pos, neg)
            with open('Search3.csv', 'w') as f:
                writer = csv.writer(f, delimiter=';', lineterminator='\n')
                writer.writerows(self._result)

        return final_program


"""
A simple breadth-first top-down learner: it extends the template learning by searching in a breadth-first fashion

It implements the abstract functions in the following way:
  - initialise_pool: creates an empty OrderedSet
  - put_into_pool: adds to the ordered set
  - get_from_pool: returns the first elements in the ordered set
  - evaluate: returns the number of covered positive examples and 0 if any negative example is covered
  - stop inner search: stops if the provided score of a clause is bigger than zero 
  - process expansions: removes from the hypothesis space all clauses that have no solutions

The learner does not handle recursions correctly!
"""


class SimpleBreadthFirstLearner(TemplateLearner):

    def __init__(self, solver_instance: Prolog, max_body_literals=4):
        super().__init__(solver_instance)
        self._max_body_literals = max_body_literals

    def initialise_pool(self):
        self._candidate_pool = OrderedSet()

    def put_into_pool(self, candidates: typing.Union[Clause, Procedure, typing.Sequence]) -> None:
        if isinstance(candidates, Clause):
            self._candidate_pool.add(candidates)
        else:
            self._candidate_pool |= candidates

    def get_from_pool(self) -> Clause:
        return self._candidate_pool.pop(0)

    def evaluate(self, examples: Task, clause: Clause) :
        pos, neg = examples.get_examples()
        numberofpositivecoverance = 0
        self._solver.assertz(clause)
        for example in pos:
            if self._solver.has_solution(example):
                numberofpositivecoverance += 1
        numberofnegativecoverance = 0
        for example in neg:
            if self._solver.has_solution(example):
                numberofnegativecoverance += 1
                # print(example)
        self._solver.retract(clause)
        if numberofnegativecoverance + numberofpositivecoverance == 0:
            return [0,0]
        else:
            return [numberofpositivecoverance / (numberofpositivecoverance + numberofnegativecoverance) * (
                numberofpositivecoverance) / len(pos),numberofnegativecoverance]


    def stop_inner_search(self, eval, examples: Task, clause: Clause) -> bool:
        if eval[1] > 0:
            return True
        else:
            return False

    def process_expansions(self, examples: Task, exps: typing.Sequence[Clause], hypothesis_space: TopDownHypothesisSpace) -> typing.Sequence[Clause]:
        # eliminate every clause with more body literals than allowed
        exps = [cl for cl in exps if len(cl) <= self._max_body_literals]

        new_exps = []
        # check if every clause has solutions
        for cl in exps:
            y = self.evaluate(task,cl)
            if y[0]>0:
                new_exps.append(cl)
            else:
                hypothesis_space.remove(cl)
        return new_exps

if __name__ == '__main__':
    # define the predicates
    is_empty = c_pred("is_empty", 1)
    not_empty = c_pred("not_empty", 1)
    is_space = c_pred("is_space", 1)
    not_space = c_pred("not_space", 1)
    is_uppercase_aux = c_pred("is_uppercase_aux", 1)
    is_lowercase_aux = c_pred("is_lowercase_aux", 1)
    not_uppercase = c_pred("not_uppercase", 1)
    is_lowercase = c_pred("is_lowercase", 1)
    not_lowercase = c_pred("not_lowercase", 1)
    is_letter = c_pred("is_letter", 1)
    not_letter = c_pred("not_letter", 1)
    is_number = c_pred("is_number", 1)
    not_number = c_pred("not_number", 1)
    skip1 = c_pred("skip1", 2)
    copy1 = c_pred("copy1", 2)
    write1 = c_pred("write1", 3)
    copyskip1 = c_pred("copyskip1", 2)
    mk_uppercase = c_pred("mk_uppercase", 2)
    mk_lowercase = c_pred("mk_lowercase", 2)
    convert_case = c_pred("convert_case", 2)
    is_number_aux = c_pred("is_number_aux", 1)
    s = c_pred("s", 2)
    b45 = c_pred("b45", 2)
    is_uppercase = c_pred("is_uppercase", 1)
    # define the Variables
    H = c_var("H")
    Ta = c_var("Ta")
    Tb = c_var("Tb")
    A = c_var("A")
    B = c_var("B")
    H1 = c_var("H1")
    H2 = c_var("H2")
    Z = c_var("Z")
    O = c_var("O")
    N = c_var("N")
    # create clauses
    head = Atom(not_space, [A])
    body = Atom(is_space, [A])
    clause1 = Clause(head, Body(Not(body)))
    head = Atom(is_uppercase, [Structure(s, [Pair(H, Z), O])])
    body = Atom(is_uppercase_aux, [H])
    clause2 = Clause(head, Body(body))
    head = Atom(not_uppercase, [A])
    body = Atom(is_uppercase, [A])
    clause3 = Clause(head, Body(Not(body)))
    head = Atom(is_lowercase, [Structure(s, [Pair(H, Z), O])])
    body = Atom(is_lowercase_aux, [H])
    clause4 = Clause(head, Body(body))
    head = Atom(not_lowercase, [A])
    body = Atom(is_lowercase, [A])
    clause5 = Clause(head, Body(Not(body)))
    head = Atom(is_letter, [Structure(s, [Pair(H, Z), O])])
    body1 = Atom(is_lowercase_aux, [H])
    body2 = Atom(is_uppercase, [H])
    clause6 = Clause(head, Body(body1))
    clause7 = Clause(head, Body(body2))
    head = Atom(not_letter, [A])
    body = Atom(is_letter, [A])
    clause8 = Clause(head, Body(Not(body)))
    head = Atom(is_number, [Structure(s, [Pair(H, Z), O])])
    body = Atom(is_number_aux, [H])
    clause9 = Clause(head, Body(body))
    head = Atom(not_number, [A])
    body = Atom(is_number, [A])
    clause10 = Clause(head, Body(Not(body)))
    head = Atom(mk_uppercase, [Structure(s, [Pair(H1, Ta), Pair(H2, Tb)]), Structure(s, [Ta, Tb])])
    body = Atom(convert_case, [H2, H1])
    clause11 = Clause(head, Body(body))
    head = Atom(mk_lowercase, [Structure(s, [Pair(H1, Ta), Pair(H2, Tb)]), Structure(s, [Ta, Tb])])
    body = Atom(convert_case, [H1, H2])
    clause12 = Clause(head, Body(body))
    Az = c_const("\"A\"")
    Bz = c_const("\"B\"")
    Cz = c_const("\"C\"")
    Dz = c_const("\"D\"")
    Ez = c_const("\"E\"")
    Fz = c_const("\"F\"")
    Gz = c_const("\"G\"")
    Hz = c_const("\"H\"")
    Iz = c_const("\"I\"")
    Jz = c_const("\"J\"")
    Kz = c_const("\"K\"")
    Lz = c_const("\"L\"")
    Mz = c_const("\"M\"")
    Nz = c_const("\"N\"")
    Oz = c_const("\"O\"")
    Pz = c_const("\"P\"")
    Qz = c_const("\"Q\"")
    Rz = c_const("\"R\"")
    Sz = c_const("\"S\"")
    Tz = c_const("\"T\"")
    Uz = c_const("\"U\"")
    Vz = c_const("\"V\"")
    Wz = c_const("\"W\"")
    Xz = c_const("\"X\"")
    Yz = c_const("\"Y\"")
    Zz = c_const("\"Z\"")
    A2 = c_const("\"\"A\"\"")
    B2 = c_const("\"\"B\"\"")
    C2 = c_const("\"\"C\"\"")
    D2 = c_const("\"\"D\"\"")
    E2 = c_const("\"\"E\"\"")
    F2 = c_const("\"\"F\"\"")
    G2 = c_const("\"\"G\"\"")
    H2 = c_const("\"\"H\"\"")
    I2 = c_const("\"\"I\"\"")
    J2 = c_const("\"\"J\"\"")
    K2 = c_const("\"\"K\"\"")
    L2 = c_const("\"\"L\"\"")
    M2 = c_const("\"\"M\"\"")
    N2 = c_const("\"\"N\"\"")
    O2 = c_const("\"\"O\"\"")
    P2 = c_const("\"\"P\"\"")
    Q2 = c_const("\"\"Q\"\"")
    R2 = c_const("\"\"R\"\"")
    S2 = c_const("\"\"S\"\"")
    T2 = c_const("\"\"T\"\"")
    U2 = c_const("\"\"U\"\"")
    V2 = c_const("\"\"V\"\"")
    W2 = c_const("\"\"W\"\"")
    X2 = c_const("\"\"X\"\"")
    Y2 = c_const("\"\"Y\"\"")
    Z2 = c_const("\"\"Z\"\"")
    nul = c_const("\"0\"")
    een = c_const("\"1\"")
    twee = c_const("\"2\"")
    drie = c_const("\"3\"")
    vier = c_const("\"4\"")
    vijf = c_const("\"5\"")
    zes = c_const("\"6\"")
    space = c_const("' '")
    space2 = c_const("\"' '\"")
    b45 = c_pred('b45', 1)

    # specify the background knowledge
    background = Knowledge(
        clause1,
        clause2,
        clause3,
        clause4,
        clause5,
        clause6,
        clause7,
        clause8,
        clause9,
        clause10,
        clause11,
        clause12,
        Atom(not_empty, [Structure(s, [Pair(H, Z), O])]),
        Atom(is_empty, [Structure(s, [List([]), Z])]),
        Atom(is_space, [Structure(s, [Pair(space, Z), O])]),
        Atom(skip1, [Structure(s, [Pair(Z, Ta), B]), Structure(s, [Ta, B])]),
        Atom(copy1, [Structure(s, [Pair(H, Ta), Pair(H, Tb)]), Structure(s, [Pair(H, Ta), Tb])]),
        Atom(write1, [Structure(s, [A, Pair(H, Tb)]), Structure(s, [A, Tb]), H]),
        Atom(copyskip1, [Structure(s, [Pair(H, Ta), Pair(H, Tb)]), Structure(s, [Ta, Tb])]),
        is_uppercase_aux(Az),
        is_uppercase_aux(Bz),
        is_uppercase_aux(Cz),
        is_uppercase_aux(Dz),
        is_uppercase_aux(Ez),
        is_uppercase_aux(Fz),
        is_uppercase_aux(Gz),
        is_uppercase_aux(Hz),
        is_uppercase_aux(Iz),
        is_uppercase_aux(Jz),
        is_uppercase_aux(Kz),
        is_uppercase_aux(Lz),
        is_uppercase_aux(Mz),
        is_uppercase_aux(Nz),
        is_uppercase_aux(Oz),
        is_uppercase_aux(Pz),
        is_uppercase_aux(Qz),
        is_uppercase_aux(Rz),
        is_uppercase_aux(Sz),
        is_uppercase_aux(Tz),
        is_uppercase_aux(Uz),
        is_uppercase_aux(Vz),
        is_uppercase_aux(Wz),
        is_uppercase_aux(Xz),
        is_uppercase_aux(Yz),
        is_uppercase_aux(Zz),
        is_lowercase_aux('a'),
        is_lowercase_aux('b'),
        is_lowercase_aux('c'),
        is_lowercase_aux('d'),
        is_lowercase_aux('e'),
        is_lowercase_aux('f'),
        is_lowercase_aux('g'),
        is_lowercase_aux('h'),
        is_lowercase_aux('i'),
        is_lowercase_aux('j'),
        is_lowercase_aux('k'),
        is_lowercase_aux('l'),
        is_lowercase_aux('m'),
        is_lowercase_aux('n'),
        is_lowercase_aux('o'),
        is_lowercase_aux('p'),
        is_lowercase_aux('q'),
        is_lowercase_aux('r'),
        is_lowercase_aux('s'),
        is_lowercase_aux('t'),
        is_lowercase_aux('u'),
        is_lowercase_aux('v'),
        is_lowercase_aux('w'),
        is_lowercase_aux('x'),
        is_lowercase_aux('y'),
        is_lowercase_aux('z'),
        convert_case(Az, 'a'),
        convert_case(Bz, 'b'),
        convert_case(Cz, 'c'),
        convert_case(Dz, 'd'),
        convert_case(Ez, 'e'),
        convert_case(Fz, 'f'),
        convert_case(Gz, 'g'),
        convert_case(Hz, 'h'),
        convert_case(Iz, 'i'),
        convert_case(Jz, 'j'),
        convert_case(Kz, 'k'),
        convert_case(Lz, 'l'),
        convert_case(Mz, 'm'),
        convert_case(Nz, 'n'),
        convert_case(Oz, 'o'),
        convert_case(Pz, 'p'),
        convert_case(Qz, 'q'),
        convert_case(Rz, 'r'),
        convert_case(Sz, 's'),
        convert_case(Tz, 't'),
        convert_case(Uz, 'u'),
        convert_case(Vz, 'v'),
        convert_case(Wz, 'w'),
        convert_case(Xz, 'x'),
        convert_case(Yz, 'y'),
        convert_case(Zz, 'z'),
        is_number_aux(nul),
        is_number_aux(een),
        is_number_aux(twee),
        is_number_aux(drie),
        is_number_aux(vier),
        is_number_aux(vijf),
        is_number_aux(zes)
    )

    # positive examples
    pos = {Atom(b45, [
        Structure(s, [List([M2, R2, space2, P2, A2, T2, R2, I2, C2, K2, space2, S2, T2, A2, R2, F2, I2, S2, H2]),
                      List([M2, 'r', space2, S2, 't', 'a', 'r', 'f', 'i', 's', 'h'])])]),
           Atom(b45, [Structure(s, [List(
               [P2, R2, O2, F2, E2, S2, S2, O2, R2, space2, M2, I2, N2, E2, R2, V2, A2, space2, M2, C2, G2, O2, N2, A2,
                G2, A2, L2, L2]),
               List(
                   [P2, 'r', 'o', 'f', 'e', 's', 's', 'o', 'r', space2, M2, 'c', 'g', 'o', 'n',
                    'a', 'g', 'a', 'l', 'l'])])]),
           Atom(b45, [Structure(s, [List([D2, R2, space2, R2, A2, Y2, space2, S2, T2, A2, N2, T2, Z2]),
                                    List([D2, 'r', space2, S2, 't', 'a', 'n', 't', 'z'])])]),
           Atom(b45, [
               Structure(s, [List([M2, S2, space2, H2, E2, R2, M2, I2, O2, N2, E2, space2, G2, R2, A2, N2, G2, E2, R2]),
                             List([M2, 's', space2, G2, 'r', 'a', 'n', 'g', 'e', 'r'])])]),
           Atom(b45, [Structure(s, [List([D2, R2, space2, B2, E2, R2, N2, A2, R2, D2, space2, R2, I2, E2, U2, X2]),
                                    List([D2, 'r', space2, R2, 'i', 'e', 'u', 'x'])])])}

    # negative examples
    neg = {Atom(b45, [
        Structure(s, [List([M2, R2, space2, P2, A2, T2, R2, I2, C2, K2, space2, S2, T2, A2, R2, F2, I2, S2, H2]),
                      List([M2, R2, space2, S2, 't', 'a', 'r', 'f', 'i', 's', 'h'])])]),
           Atom(b45, [Structure(s, [List(
               [P2, R2, O2, F2, E2, S2, S2, O2, R2, space2, M2, I2, N2, E2, R2, V2, A2, space2, M2, C2, G2, O2, N2, A2,
                G2, A2, L2, L2]),
               List([M2, 'i', 'n', 'e', 'r', 'v', 'a', space2, M2, 'c', 'g', 'o', 'n', 'a', 'g',
                     'a', 'l', 'l'])])]),
           Atom(b45, [Structure(s, [List([D2, R2, space2, R2, A2, Y2, space2, S2, T2, A2, N2, T2, Z2]),
                                    List([D2, 'r', space2, S2, T2, A2, N2, T2, Z2])])]),
           Atom(b45, [
               Structure(s, [List([M2, S2, space2, H2, E2, R2, M2, I2, O2, N2, E2, space2, G2, R2, A2, N2, G2, E2, R2]),
                             List(['m', 's', space2, G2, 'r', 'a', 'n', 'g', 'e', 'r'])])]),
           Atom(b45, [Structure(s, [List([D2, R2, space2, B2, E2, R2, N2, A2, R2, D2, space2, R2, I2, E2, U2, X2]),
                                    List(['d', R2, space2, R2, 'i', 'e', 'u', 'x'])])])}
    task = Task(positive_examples=pos, negative_examples=neg)

    examples = {Atom(b45, [
        Structure(s, [List([Mz, Rz, space, Pz, Az, Tz, Rz, Iz, Cz, Kz, space, Sz, Tz, Az, Rz, Fz, Iz, Sz, Hz]),
                      List([Mz, 'r', space, Sz, 't', 'a', 'r', 'f', 'i', 's', 'h'])])]),
                Atom(b45, [Structure(s, [List(
                    [Pz, Rz, Oz, Fz, Ez, Sz, Sz, Oz, Rz, space, Mz, Iz, Nz, Ez, Rz, Vz, Az, space, Mz, Cz, Gz, Oz, Nz,
                     Az,
                     Gz, Az, Lz, Lz]),
                    List(
                        [Pz, 'r', 'o', 'f', 'e', 's', 's', 'o', 'r', space, Mz, 'c', 'g', 'o', 'n',
                         'a', 'g', 'a', 'l', 'l'])])]),
                Atom(b45, [Structure(s, [List([Dz, Rz, space, Rz, Az, Yz, space, Sz, Tz, Az, Nz, Tz, Zz]),
                                         List([Dz, 'r', space, Sz, 't', 'a', 'n', 't', 'z'])])]),
                Atom(b45, [Structure(s, [
                    List([Mz, Sz, space, Hz, Ez, Rz, Mz, Iz, Oz, Nz, Ez, space, Gz, Rz, Az, Nz, Gz, Ez, Rz]),
                    List([Mz, 's', space, Gz, 'r', 'a', 'n', 'g', 'e', 'r'])])]),
                Atom(b45, [Structure(s, [List([Dz, Rz, space, Bz, Ez, Rz, Nz, Az, Rz, Dz, space, Rz, Iz, Ez, Uz, Xz]),
                                         List([Dz, 'r', space, Rz, 'i', 'e', 'u', 'x'])])]),
                Atom(b45, [Structure(s, [
                    List([Mz, Rz, space, Pz, Az, Tz, Rz, Iz, Cz, Kz, space, Sz, Tz, Az, Rz, Fz, Iz, Sz, Hz]),
                    List([Mz, Rz, space, Sz, 't', 'a', 'r', 'f', 'i', 's', 'h'])])]),
                Atom(b45, [Structure(s, [
                    List([Pz, Rz, Oz, Fz, Ez, Sz, Sz, Oz, Rz, space, Mz, Iz, Nz, Ez, Rz, Vz, Az, space, Mz,
                          Cz, Gz, Oz, Nz, Az, Gz, Az, Lz, Lz]),
                    List([Mz, 'i', 'n', 'e', 'r', 'v', 'a', space, Mz, 'c', 'g', 'o', 'n', 'a', 'g',
                          'a', 'l', 'l'])])]),
                Atom(b45, [Structure(s, [List([Dz, Rz, space, Rz, Az, Yz, space, Sz, Tz, Az, Nz, Tz, Zz]),
                                         List([Dz, 'r', space, Sz, Tz, Az, Nz, Tz, Zz])])]),
                Atom(b45, [Structure(s, [
                    List([Mz, Sz, space, Hz, Ez, Rz, Mz, Iz, Oz, Nz, Ez, space, Gz, Rz, Az, Nz, Gz, Ez, Rz]),
                    List(['m', 's', space, Gz, 'r', 'a', 'n', 'g', 'e', 'r'])])]),
                Atom(b45, [Structure(s, [List([Dz, Rz, space, Bz, Ez, Rz, Nz, Az, Rz, Dz, space, Rz, Iz, Ez, Uz, Xz]),
                                         List(['d', Rz, space, Rz, 'i', 'e', 'u', 'x'])])])

                }

    # create Prolog instance
    prolog = SWIProlog()

    learner = SimpleBreadthFirstLearner(prolog, max_body_literals=10)

    # create the hypothesis space
    hs = TopDownHypothesisSpace(primitives=[
        lambda x: plain_extension(x, not_space, connected_clauses=True),
        lambda x: plain_extension(x, mk_uppercase, connected_clauses=True),
        lambda x: plain_extension(x, mk_lowercase, connected_clauses=True),
        lambda x: plain_extension(x, is_empty, connected_clauses=True),
        lambda x: plain_extension(x, is_space, connected_clauses=True),
        lambda x: plain_extension(x, is_uppercase, connected_clauses=True),
        lambda x: plain_extension(x, not_uppercase, connected_clauses=True),
        lambda x: plain_extension(x, is_lowercase, connected_clauses=True),
        lambda x: plain_extension(x, not_lowercase, connected_clauses=True),
        lambda x: plain_extension(x, is_letter, connected_clauses=True),
        lambda x: plain_extension(x, not_letter, connected_clauses=True),
        lambda x: plain_extension(x, is_number, connected_clauses=True),
        lambda x: plain_extension(x, not_number, connected_clauses=True),
        lambda x: plain_extension(x, skip1, connected_clauses=True),
        lambda x: plain_extension(x, copy1, connected_clauses=True),
        lambda x: plain_extension(x, write1, connected_clauses=True),
        lambda x: plain_extension(x, copyskip1, connected_clauses=True)
    ],
        head_constructor=b45,
        expansion_hooks_reject=[
            lambda x, y: has_duplicated_literal(x, y),
            lambda x, y: has_duplicated_variable(x, y)],
        recursive_procedures=False)

    program = learner.learn(task, background, hs)

    print(program)



