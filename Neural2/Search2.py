import typing
import csv
from abc import ABC, abstractmethod

from orderedset import OrderedSet

from pylo.engines.prolog.SWIProlog import Pair

from Neural2 import Network
from docs.mytuple import mytuple
from loreleai.language.lp import c_var, c_pred, Clause, Procedure, Atom, Body, Not, List, c_const, c_functor, c_literal, \
    Structure
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_singleton_vars, has_duplicated_literal, has_duplicated_variable, \
    connected_clause, connected_body, max_var_occurrences
from loreleai.learning.language_manipulation import plain_extension, _plain_extend_clause
from loreleai.learning.task import Task, Knowledge
# from docs.SWIProlog2 import SWIProlog
from loreleai.reasoning.lp.prolog import SWIProlog, Prolog
import heapq as has
from Neural2 import encoding2,decoding
import numpy as np
import random
import time

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

class SimpleBreadthFirstLearner:

    def __init__(self, solver_instance: Prolog,encoder:encoding2,max_body_literals=10):
        self._solver = solver_instance
        self._candidate_pool = []
        self._encoder = encoder
        self._max_body_literals = max_body_literals

    def assert_knowledge(self, knowledge: Knowledge):
        """
        Assert knowledge into Prolog engine
        """
        facts = knowledge.get_atoms()
        for f_ind in range(len(facts)):
            self._solver.assertz(facts[f_ind])

        clauses = knowledge.get_clauses()
        for cl_ind in range(len(clauses)):
            self._solver.assertz(clauses[cl_ind])

    def initialise_pool(self):
        self._candidate_pool = []
        has.heapify(self._candidate_pool)

    def put_into_pool(self, value: float,neural:float,expansion:float, candidate):
        has.heappush(self._candidate_pool, mytuple(value,neural,expansion, candidate))

    def get_from_pool(self):
        return has.heappop(self._candidate_pool)

    def evaluate(self, examples: Task, clause: Clause):
        if self._numberOfEvaluations < 90000:
            self._numberOfEvaluations += 1
            numberofpositivecoverance = 0

            self._solver.assertz(clause)
            for example in self._positive_examples:
                if self._solver.has_solution(example):
                    numberofpositivecoverance += 1
            numberofnegativecoverance = 0
            for example in self._negative_examples:
                if self._solver.has_solution(example):
                    numberofnegativecoverance += 1
                    #print(example)
            self._solver.retract(clause)
            if numberofnegativecoverance + numberofpositivecoverance == 0:
                return [0, 0]
            else:
                return [numberofpositivecoverance / (numberofpositivecoverance + numberofnegativecoverance) * (
                    numberofpositivecoverance) / len(self._positive_examples), numberofnegativecoverance]
        else:
            self._numberOfEvaluations = 0
            self._solver.release()
            self._assert_knowledge(self._knowledge)
            self._numberOfEvaluations += 1
            numberofpositivecoverance = 0
            self._solver.assertz(clause)
            for example in self._positive_examples:
                if self._solver.has_solution(example):
                    numberofpositivecoverance += 1
            numberofnegativecoverance = 0
            for example in self._negative_examples:
                if self._solver.has_solution(example):
                    numberofnegativecoverance += 1
            self._solver.retract(clause)
            if numberofnegativecoverance + numberofpositivecoverance == 0:
                return [0,0]
            else:
                return [numberofpositivecoverance / (numberofpositivecoverance + numberofnegativecoverance) * (
                    numberofpositivecoverance) / len(self._positive_examples), numberofnegativecoverance]

    def evaluateClause(self, examples, clause: Clause):
            print(clause)
            numberofpositivecoverance = 0
            self._solver.assertz(clause)
            for example in examples:
                if self._solver.has_solution(example):
                    print(example)
                    numberofpositivecoverance += 1
            self._solver.retract(clause)
            return numberofpositivecoverance

    def encodeState(self,state):
        number = 0
        encoding = []
        for example in state:
                if(number == 0):
                    encoding = self._encodingProblems[example]
                    print(self._encodingProblems[example])
                    print(encoding)
                    number += 1
                else:
                    encoding = [a + b for a, b in zip(encoding, self._encodingProblems[example])]
                    print(encoding)
                    print(self._encodingProblems[example])
                    number +=1
        print(np.concatenate((number,np.divide(encoding,number)),axis=None))
        return np.concatenate((number,np.divide(encoding,number)),axis=None)

    def stop_inner_search(self, eval: typing.Union[int, float], examples: Task, clause: Clause) -> bool:
        if eval == 1:
            return True
        else:
            return False

    def process_expansions(self, examples: Task, exps: typing.Sequence[Clause],
                           hypothesis_space: TopDownHypothesisSpace) -> typing.Sequence[Clause]:
        # eliminate every clause with more body literals than allowed
        exps = [cl for cl in exps if len(cl) <= self._max_body_literals]

        # check if every clause has solutions
        exps = [(cl, self._solver.has_solution(*cl.get_body().get_literals())) for cl in exps]
        new_exps = []

        for ind in range(len(exps)):
            if exps[ind][1]:
                # keep it if it has solutions
                new_exps.append(exps[ind][0])
            else:
                # remove from hypothesis space if it does not
                hypothesis_space.remove(exps[ind][0])

        return new_exps


    def _learn_one_clause(self, examples: Task, hypothesis_space: TopDownHypothesisSpace,decoder,primitives) -> Clause:
        """
        Learns a single clause
        Returns a clause
        """
        # reset the search space
        hypothesis_space.reset_pointer()

        # empty the pool just in case
        self.initialise_pool()

        # put initial candidates into the pool
        head = Atom(b45, [A])
        best = None
        self._expansionOneClause = 0

        for prim in primitives:
            print(prim)
            exps = plain_extension(Clause(head,[]).get_body(),prim)
            lengte = len(exps)
            for i in range(0,lengte):
                exps[i] = Clause(head,exps[i])
                if not self.badClause(exps[i].get_head,exps[i].get_body()):
                    y = self.evaluate(examples,exps[i])
                    if y[0] >0:
                        self._expansion += 1
                        self._expansionOneClause += 1
                        self._result.append((self._expansion, y[0]))
                        self.put_into_pool(1 - y[0], 0, 0, exps[i])
                        if (best == None) or y[0]>best[0]:
                            best = (y[0],exps[i])

        current_cand = None
        found = False
        pos, _ = task.get_examples()
        state = self.encodeState(pos)



        while current_cand is None or (
                len(self._candidate_pool) > 0) and self._expansionOneClause < 10000:
            # get first candidate from the pool
            current_cand = self.get_from_pool()
            print('current : ', current_cand._clause )
            print('value : ' ,current_cand._value)
            print('expansions : ' , self._expansion)
            current_cand = current_cand._clause
            expansions = decoder.decode(self._neural1.FeedForward([*self._encoder.encode2(current_cand),*state]))
            bestk = sorted(expansions, reverse=True)[0:7]
            #print(bestk)
            for exp in bestk:
                exps = plain_extension(current_cand.get_body(),exp._clause)
                bestkClause = []
                for exp2 in exps:
                    if not self.badClause(current_cand.get_head,exp2):
                        y = self._neural2.FeedForward([*self._encoder.encode2(Clause(current_cand.get_head(),exp2)), *state])
                        bestkClause.append(mytuple(y[0],y[0],exp._value,Clause(current_cand.get_head(),exp2)))
                if len(bestkClause)>10:
                    bestkClause = sorted(bestkClause, reverse=True)[0:10]
                else:
                    sorted(bestkClause, reverse=True)
                toBeAdded = []
                for i in range(0,len(bestkClause)):
                    y = self.evaluate(examples,bestkClause[i]._clause)
                    if (y[1] == 0 )& (y[0]>0):
                        print('found')
                        return bestkClause[i]._clause
                    else:
                        if y[0]>0:
                            self._expansion += 1
                            self._expansionOneClause +=1
                            self._result.append((self._expansion, y[0]))
                            toBeAdded.append((y[0],bestkClause[i]._value,bestkClause[i]._clause))
                            if y[0] > best[0]:
                                best = (y[0],bestkClause[i]._clause)
                            else:
                                if (y[0] == best[0]) & (len(bestkClause[i]._clause.get_literals()) < len(best[1].get_literals())):
                                    best = (y[0],bestkClause[i]._clause)
                for expy in toBeAdded:
                    if len(expy[2]) < self._max_body_literals:
                        self.put_into_pool(1 - expy[0],1-expy[1],1-exp._expansion, expy[2])

        print(best)
        return best[1]

    def badClause(self,head: Atom, body: Body) -> bool:
        if has_duplicated_literal(head,body):
            return True
        if not connected_body(head, body):
            return True
        if has_duplicated_variable(head,body):
            return True
        #if max_var_occurrences(head,body,2):
        #    return True
        #if max_pred_occurrences(Atom,Body,copy1,2)
        return False

    def _learn_one_clause2(self, examples: Task, hypothesis_space: TopDownHypothesisSpace,decoder,primitives) -> Clause:
        """
        Learns a single clause
        Returns a clause
        """
        # reset the search space
        hypothesis_space.reset_pointer()

        # empty the pool just in case
        self.initialise_pool()

        # put initial candidates into the pool
        head = Atom(b45, [A])

        print('primitives : ' , primitives)
        exps = hypothesis_space.expand(Clause(head,[]))
        print(exps)
        lengte = len(exps)
        for i in range(0,lengte):
            print(exps[i])
            y = self.evaluate(examples,exps[i])
            print(y)
            if y[0] >0:
                self.put_into_pool(1 - y[0],0, exps[i])

        current_cand = None
        found = False
        pos, _ = task.get_examples()
        state = self.encodeState(pos)

        while current_cand is None or (
                len(self._candidate_pool) > 0):
            # get first candidate from the pool
            current_cand = self.get_from_pool()
            print('current : ', current_cand._clause)
            print('value : ' ,current_cand._value)
            current_cand = current_cand._clause
            expansions = decoder.decode(self._neural1.FeedForward([*self._encoder.encode2(current_cand),*state]))
            bestk = sorted(expansions, reverse=True)[0:3]
            for exp in bestk:
                exps = hypothesis_space.expandwithprimitive(current_cand,exp._clause)
                bestkClause = []
                for exp2 in exps:
                    addedLiteral = exp2.get_literals()[-1]
                    y = self._neural2.FeedForward([*self._encoder.encode2(exp2), *state])
                    bestkClause.append(mytuple(y[0],0,exp2))
                if len(bestkClause)>3:
                    bestkClause = sorted(bestkClause, reverse=True)[0:3]
                else:
                    sorted(bestkClause, reverse=True)
                toBeAdded = []
                for i in range(0,len(bestkClause)):
                    y = self.evaluate(examples,bestkClause[i]._clause)

                    if (y[1] == 0 )& (y[0]>0):
                        print('found')
                        print(bestkClause[i]._clause)
                        return bestkClause[i]._clause
                    else:
                        if y[0]>0:
                            toBeAdded.append((y[0],bestkClause[i]._value,bestkClause[i]._clause))
                for exp in toBeAdded:
                    if len(exp[2]) < self._max_body_literals:
                        self.put_into_pool(1 - exp[0],1-exp[1], exp[2])

        return current_cand

    def _execute_program(self, clause: Clause) -> typing.Sequence[Atom]:
        """
        Evaluates a clause using the Prolog engine and background knowledge
        Returns a set of atoms that the clause covers
        """
        if self._numberOfEvaluations < 90000:
            self._numberOfEvaluations += 1
            numberofpositivecoverance = 0
            self._solver.assertz(clause)
            coverage = []
            for example in self._positive_examples:
                if self._solver.has_solution(example):
                    coverage.append(example)
            self._solver.retract(clause)
            return coverage
        else:
            self._numberOfEvaluations = 0
            self._solver.release()
            self._assert_knowledge(self._knowledge)
            self._numberOfEvaluations += 1
            self._solver.assertz(clause)
            coverage = []
            for example in self._positive_examples:
                if self._solver.has_solution(example):
                    coverage.append(example)
            self._solver.retract(clause)
            return coverage

    def learn(self, examples: Task, knowledge: Knowledge, hypothesis_space: TopDownHypothesisSpace, neural1: Network,neural2, encodingExamples,decoder,primitives):
        """
        General learning loop
        """

        self._assert_knowledge(knowledge)

        final_program = []
        examples_to_use = examples
        pos, neg = examples_to_use.get_examples()
        self._positive_examples = pos
        self._negative_examples = neg
        self._neural1 = neural1
        self._neural2 = neural2
        self._encodingProblems = encodingExamples
        self._numberOfEvaluations = 0
        self._result = []
        self._expansion = 0

        while len(final_program) == 0 or len(pos) > 0:
            # learn na single clause
            cl = self._learn_one_clause(examples_to_use, hypothesis_space,decoder,primitives)
            final_program.append(cl)

            # update covered positive examples
            covered = self._execute_program(cl)
            print('covered : , ' , covered)
            print(len(covered))

            pos, neg = examples_to_use.get_examples()
            pos = pos.difference(covered)
            self._positive_examples = self._positive_examples.difference(covered)
            print(len(pos))
            print(len(self._positive_examples))
            with open('Search1.csv', 'w') as f:
                writer = csv.writer(f, delimiter=';', lineterminator='\n')
                writer.writerows(self._result)



            examples_to_use = Task(pos, neg)

        return final_program

    def getSolver(self):
        return self._solver

    def pickKBest(self,expansions,k):
        bestk = []
        best = expansions[0]._value
        last = []
        if(best==0):
            return bestk
        for clause in expansions :
            if clause._value == best:
                last.append(clause)
            else:
                if len(bestk) + len(last) <= k :
                    bestk = bestk + last
                    if (len(bestk)==k) | (clause._value<=0 ):
                        return bestk
                    last = [clause]
                    best = clause._value
                else:
                    l = k -len(bestk)
                    bestk = bestk + random.sample(last,l)
                    return bestk
        l = k-len(bestk)
        bestk = bestk + random.sample(last, l)
        return bestk

    def readEncodingsOfFile(self, file):
        encoding = np.loadtxt(file, dtype=np.object)
        encoding = [[float(y) for y in x] for x in encoding]
        return encoding


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
    b45 = c_pred("b45", 1)
    is_uppercase = c_pred("is_uppercase", 1)
    # define the Variables
    H = c_var("H")
    Ta = c_var("Ta")
    Tb = c_var("Tb")
    A = c_var("A")
    B = c_var("B")
    C = c_var("C")
    D = c_var("D")
    E = c_var("E")
    F = c_var("F")
    G = c_var("G")
    H = c_var("H")
    I = c_var("I")
    J = c_var("J")
    K = c_var("K")
    L = c_var("L")
    M = c_var("M")
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

    covered = {Atom(b45, [Structure(s, [List(
        ['z', V2, X2, 'k', F2, V2, 'p', 'e']),
        List(
            [V2, X2, F2, V2])])]),
               Atom(b45, [Structure(s, [List(
                   [N2, 'z', 'a', 'f', 'v', 'x', J2, 'o', 'z']),
                   List(
                       [N2, J2])])]),
               Atom(b45, [Structure(s, [List(
                   ['f', 's', 'g', 't', 'u']),
                   List(
                       [])])]),
               Atom(b45, [Structure(s, [List(
                   [L2, Z2, E2, 'i', 'w', 'n', G2, 'j', 'u', 't', 'a', 'f', U2, 'k']),
                   List(
                       [L2, Z2, E2, G2, U2])])]),
               Atom(b45, [Structure(s, [List(
                   ['c', U2, 's']),
                   List(
                       [U2])])])

               }

    # positive examples
    pos = {Atom(b45, [
        Structure(s, [List([F2, 'c', E2, Q2, E2, 'h', 'c', F2, C2, 'q']),
                      List([F2, E2, Q2, E2, F2, C2])])]),
           Atom(b45, [Structure(s, [List(
               ['o', N2, A2, 'z', 'g', 'f']),
               List(
                   [N2, A2])])]),
           Atom(b45, [Structure(s, [List([T2, 'r', 'g', 'y', T2, P2]),
                                    List([T2, T2, P2])])]),
           Atom(b45, [
               Structure(s, [List(['c', O2, 'o', 'j', 'j', H2, F2, M2, 'g', C2]),
                             List([O2, H2, F2, M2, C2])])]),
           Atom(b45, [Structure(s, [List(['x', 'u', 'k', 'l', 'w', 'f', Z2, L2, R2, 'h', U2, 't']),
                                    List([Z2, L2, R2, U2])])])}

    # negative examples
    neg = {Atom(b45, [
        Structure(s, [List(['e', 'd', 'i', 't', 'h']),
                      List(['e', 'i'])])]),
           Atom(b45, [Structure(s, [List(
               ['q', 'c', 'p', 'm', Z2, 'j', 'm', 'g', L2, 'y', P2, Q2, 'q']),
               List(['q', 'c', 'p', 'm', 'j', 'm', 'g', 'y', 'q'])])]),
           Atom(b45, [Structure(s, [List(['j', 'a', 'm', 'e', 's']),
                                    List(['j', 's'])])]),
           Atom(b45, [
               Structure(s, [List([E2, 'c', O2, U2, U2]),
                             List(['c'])])]),
           Atom(b45, [Structure(s, [List([Q2, 'o', 'G', 'w', 't']),
                                    List(['o', 'w', 't'])])])}
    task = Task(positive_examples=pos, negative_examples=neg)

    examples = {Atom(b45, [
        Structure(s, [List([Fz, 'c', Ez, Qz, Ez, 'h', 'c', Fz, Cz, 'q']),
                      List([Fz, Ez, Qz, Ez, Fz, Cz])])]),
                Atom(b45, [Structure(s, [List(
                    ['o', Nz, Az, 'z', 'g', 'f']),
                    List(
                        [Nz, Az])])]),
                Atom(b45, [Structure(s, [List([Tz, 'r', 'g', 'y', Tz, Pz]),
                                         List([Tz, Tz, Pz])])]),
                Atom(b45, [
                    Structure(s, [List(['c', Oz, 'o', 'j', 'j', Hz, Fz, Mz, 'g', Cz]),
                                  List([Oz, Hz, Fz, Mz, Cz])])]),
                Atom(b45, [Structure(s, [List(['x', 'u', 'k', 'l', 'w', 'f', Zz, Lz, Rz, 'h', Uz, 't']),
                                         List([Zz, Lz, Rz, Uz])])]),
                Atom(b45, [
                    Structure(s, [List(['e', 'd', 'i', 't', 'h']),
                                  List(['e', 'i'])])]),
                Atom(b45, [Structure(s, [List(
                    ['q', 'c', 'p', 'm', Zz, 'j', 'm', 'g', Lz, 'y', Pz, Qz, 'q']),
                    List(['q', 'c', 'p', 'm', 'j', 'm', 'g', 'y', 'q'])])]),
                Atom(b45, [Structure(s, [List(['j', 'a', 'm', 'e', 's']),
                                         List(['j', 's'])])]),
                Atom(b45, [
                    Structure(s, [List([Ez, 'c', Oz, Uz, Uz]),
                                  List(['c'])])]),
                Atom(b45, [Structure(s, [List([Qz, 'o', 'G', 'w', 't']),
                                         List(['o', 'w', 't'])])])

                }

    # create Prolog instance
    prolog = SWIProlog()

    #learner = SimpleBreadthFirstLearner(prolog, max_body_literals=4)

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
                                lambda x, y: has_duplicated_variable(x,y)],
        recursive_procedures=False)

    primitives2 = [b45(c_var('X')), b45(c_var('Y')), not_space(c_var('X')), not_space(c_var('Y')),
                  mk_uppercase(c_var('X'), c_var('Y')),
                  mk_uppercase(c_var('Y'), c_var('X')),
                  mk_uppercase(c_var('Y'), c_var('Z')), mk_lowercase(c_var('X'), c_var('Y')),
                  mk_lowercase(c_var('Y'), c_var('X')),
                  mk_lowercase(c_var('Y'), c_var('Z')), is_empty(c_var('X')), is_empty(c_var('Y')), is_space(c_var('X'))
        , is_space(c_var('Y')), is_uppercase(c_var('X')), is_uppercase(c_var('Y')), not_uppercase(c_var('X')),
                  not_uppercase(c_var('Y')),
                  is_lowercase(c_var('X')), is_lowercase(c_var('Y')), not_lowercase(c_var('X')),
                  not_lowercase(c_var('Y')), is_letter(c_var('X')),
                  is_letter(c_var('Y')), not_letter(c_var('X')), not_letter(c_var('Y')), is_number(c_var('X')),
                  is_number(c_var('Y')), not_number('X'), not_number('Y'), skip1(c_var('X'), c_var('Y')),
                  skip1(c_var('Y'), c_var('X')),
                  skip1(c_var('Y'), c_var('Z')), copy1(c_var('X'), c_var('Y')), copy1(c_var('Y'), c_var('X')),
                  copy1(c_var('Y'), c_var('Z')),
                  write1(c_var('X'), c_var('Y'), c_var('Z')), write1(c_var('Y'), c_var('X'), c_var('Z')),
                  write1(c_var('Y'), c_var('Z'), c_var('X')),
                  write1(c_var('Y'), c_var('Z'), c_var('U')), copyskip1(c_var('X'), c_var('Y')),
                  copyskip1(c_var('Y'), c_var('X')), copyskip1(c_var('Y'), c_var('Z'))]

    encodingExamples = {
        Atom(b45, [
            Structure(s, [List(['c', O2, 'o', 'j', 'j', H2, F2, M2, 'g', C2]),
                          List([O2, H2, F2, M2, C2])])]):
            [-3.6467e-01, 7.2022e-01, 9.9840e-01, -9.9039e-01, 8.3420e-01,
            5.5713e-01, 9.8474e-01, 8.3312e-01, -9.6535e-01, -7.9329e-01,
            9.5382e-01, 9.9510e-01, 4.0992e-01, -9.9503e-01, -9.5377e-02,
            -9.8506e-01, 9.8066e-01, -3.7422e-01, -9.9849e-01, -4.7179e-01,
            4.7208e-01, -9.9781e-01, 5.5456e-01, 8.2827e-01, 9.6785e-01,
            1.1619e-01, 9.8834e-01, 9.9754e-01, 9.2737e-01, 8.9083e-01,
            3.5186e-01, -9.8579e-01, 6.2498e-01, -9.8635e-01, 1.5097e-01,
            -4.1594e-01, -1.5747e-01, -1.0054e-01, 6.0735e-01, -5.2962e-01,
            -9.0573e-01, 1.2193e-01, -5.5054e-01, -5.0213e-01, -7.1285e-01,
            6.9840e-01, 7.0002e-01, -2.8657e-01, -3.4797e-01, 9.9937e-01,
            -9.6649e-01, 1.0000e+00, 4.8330e-01, 9.9986e-01, 9.9640e-01,
            8.4478e-01, 9.9368e-01, 5.0361e-01, 5.6546e-01, 8.7371e-01,
            9.6266e-01, -5.2744e-01, 8.8764e-01, -6.6546e-01, -6.1750e-01,
            -9.2757e-01, 1.1076e-01, 5.3724e-01, -6.0024e-01, 8.7912e-01,
            7.1418e-01, 4.1665e-01, 9.9339e-01, -9.2335e-01, 1.1751e-01,
            -9.4106e-01, 6.5125e-01, -9.9860e-01, 9.7218e-01, 9.9782e-01,
            -7.7100e-01, -9.9456e-01, 9.9022e-01, -3.1527e-01, 2.9199e-01,
            -7.1087e-01, 7.3231e-01, -9.9483e-01, 2.2366e-01, -8.6273e-01,
            -3.6894e-01, -9.6951e-01, 1.2741e-01, 3.6664e-01, 9.9757e-01,
            6.4444e-01, -1.8288e-01, 2.2932e-01, 2.3575e-01, 8.0981e-01,
            -8.1989e-01, -4.2505e-01, -7.9627e-01, 8.6019e-01, -4.7185e-01,
            -3.8494e-01, -9.6987e-01, -2.8147e-01, -4.4234e-01, 1.0784e-01,
            9.9466e-01, -9.7669e-01, -6.6159e-01, 3.0513e-01, -4.2590e-03,
            2.8770e-01, 9.7433e-01, 8.5127e-01, -3.4781e-01, 9.9746e-01,
            -1.6392e-01, 5.5473e-02, 9.9290e-01, -5.1847e-01, 7.7579e-01,
            -4.1315e-01, 8.1691e-02, 1.6544e-03, 4.9338e-01, -9.0688e-01,
            6.7730e-01, -9.8294e-01, 7.3619e-01, 9.9229e-01, -3.9875e-01,
            9.9812e-01, -9.9562e-01, 1.4758e-01, -9.9939e-01, 4.5815e-01,
            9.0275e-01, -2.7708e-01, 2.1270e-01, 7.6177e-01, 9.9479e-01,
            1.8606e-01, 1.7964e-02, 4.5942e-01, -3.7773e-01, 1.7604e-01,
            6.7649e-01, 9.3257e-01, -9.6515e-01, 9.9999e-01, -8.3262e-02,
            2.4859e-01, 7.2922e-01, -2.6777e-02, 3.7276e-01, -3.5377e-01,
            9.7556e-01, -9.9397e-01, -6.8624e-01, 6.4995e-01, 9.9400e-01,
            9.8391e-01, -6.9883e-02, 5.4316e-01, 9.9597e-01, 6.6104e-01,
            3.8143e-01, -7.2665e-01, -6.7922e-01, 4.8303e-01, 2.7916e-01,
            2.6955e-01, 9.6980e-01, 9.9274e-01, -9.8062e-01, 9.9180e-01,
            9.9970e-01, -6.5482e-01, 9.6088e-01, -6.6280e-01, -9.9457e-01,
            -9.8396e-01, -9.7528e-01, 5.2064e-01, -9.0412e-01, -4.3723e-01,
            -3.9116e-01, 9.6043e-01, -8.1902e-01, 9.3267e-01, -9.7377e-01,
            -5.4776e-02, 9.9067e-01, -2.4789e-01, 9.9619e-01, 9.1358e-01,
            -9.9529e-01, -2.5520e-01, -5.6831e-01, 9.7236e-01, -3.1447e-01,
            9.8203e-01, 2.8774e-01, 4.9507e-01, 6.2686e-01, -1.0000e+00,
            -8.6591e-01, -6.2642e-01, -7.0219e-01, 8.0485e-01, 9.8562e-01,
            5.3985e-01, -4.4396e-01, -4.6622e-02, 4.8686e-01, 9.9873e-01,
            -9.8522e-01, -5.3539e-01, 7.2772e-01, -9.8906e-01, -9.9503e-01,
            9.4938e-01, -1.6706e-01, -4.0645e-02, -5.5613e-01, -8.5635e-01,
            -1.2307e-01, 3.8780e-01, 9.8724e-01, 2.7739e-01, 8.1338e-01,
            -9.9848e-01, 9.2932e-01, -9.7434e-01, 1.5176e-01, 4.1901e-01,
            5.9797e-01, -5.1924e-01, -9.3232e-01, 6.9774e-01, 9.1404e-01,
            -5.1095e-01, 6.6777e-01, -7.2099e-01, 7.6532e-01, 7.8509e-01,
            -5.7691e-01, -4.3890e-01, -9.8610e-01, 9.9378e-01, 7.8377e-01,
            -1.7584e-01, 9.8713e-01, -9.8723e-01, -3.8431e-02, 5.5588e-01,
            -6.5306e-02, -9.9998e-01, -1.1045e-01, -8.7917e-01, 8.4667e-01,
            -4.2011e-01, 9.8855e-01, -9.5579e-01, -1.2259e-01, 3.7354e-01,
            -9.9643e-01, 9.5086e-01, -5.3302e-01, 9.8977e-01, -8.3638e-01,
            6.2319e-01, 9.9647e-01, 4.5891e-01, -9.8970e-01, -9.9885e-01,
            7.5077e-01, 9.9983e-01, -9.7343e-01, -1.7714e-01, 9.9868e-01,
            8.2817e-01, -9.6074e-01, -9.4818e-01, -9.9504e-01, -9.9420e-01,
            -6.9759e-01, -9.1964e-02, -2.4149e-01, 9.9221e-01, -3.5998e-02,
            -1.3025e-01, 9.8604e-01, 9.9989e-01, -5.0721e-01, 2.9288e-01,
            3.4391e-01, -8.9459e-01, -9.9993e-01, 7.4551e-01, 1.7648e-01,
            -9.9855e-01, 9.9524e-01, -9.9178e-01, 9.9999e-01, -8.8402e-01,
            7.3901e-01, 6.9683e-01, -3.1562e-01, 4.1003e-01, 1.6442e-01,
            9.9804e-01, 9.4683e-01, -5.8621e-01, 3.0281e-01, -3.5627e-01,
            4.8122e-01, -5.7172e-01, 8.5881e-01, 3.2883e-01, 1.7825e-02,
            -9.0477e-01, -5.4145e-01, -5.2438e-01, -9.9275e-01, -1.0088e-01,
            4.2083e-01, 9.1224e-01, -3.9628e-02, 9.2188e-01, 9.8817e-01,
            -2.3303e-01, 7.7280e-01, -4.4116e-01, -9.9642e-01, 3.5327e-01,
            2.5967e-01, -3.1377e-01, 5.9854e-01, 9.1237e-01, 9.9053e-01,
            -9.4702e-01, 9.9982e-01, -3.6196e-01, 3.7609e-01, 8.0033e-01,
            9.9904e-01, -9.9947e-01, 2.8718e-01, -2.2450e-01, 7.1495e-01,
            7.3650e-01, 9.8598e-01, -2.6949e-01, 1.7506e-01, 4.5297e-02,
            3.1123e-01, 9.2281e-01, 9.8072e-01, 2.7674e-02, -1.8662e-01,
            9.4159e-01, 2.9692e-01, 9.9112e-01, -7.1030e-01, -3.6952e-02,
            -7.4586e-01, 6.6062e-01, 9.6682e-01, -6.5103e-02, -4.3717e-01,
            -3.2153e-01, -7.1926e-02, -5.4892e-02, -5.5772e-01, 7.1477e-01,
            1.6406e-01, 2.4687e-01, -9.9093e-01, -5.3837e-01, 5.1729e-01,
            9.9437e-01, -9.9155e-01, -1.1506e-01, 9.8615e-01, -2.6445e-01,
            1.4104e-01, 7.7587e-01, -2.8414e-01, -9.3303e-01, -3.2512e-01,
            -9.9690e-01, -7.4687e-04, -6.8585e-01, 7.0480e-01, 5.6853e-01,
            1.5994e-01, 8.2832e-02, 9.1427e-02, -8.4324e-01, 9.0821e-01,
            3.7069e-01, 9.8472e-01, 8.1723e-01, -2.5179e-01, -1.1686e-01,
            4.9422e-01, 2.7198e-01, -3.8019e-01, 9.9208e-01, -9.8564e-01,
            9.9877e-01, -3.6990e-01, -9.9558e-01, 8.6555e-01, 3.9364e-01,
            -9.9790e-01, 1.1022e-01, -9.9979e-01, 9.8090e-01, -6.9518e-01,
            7.0505e-01, 7.8127e-01, -9.9978e-01, -9.9998e-01, -3.3966e-01,
            -3.1010e-01, -4.4180e-01, -8.3812e-01, 9.9334e-01, -3.5940e-02,
            6.5241e-01, -1.8249e-01, -9.3167e-01, 3.3659e-01, 7.9161e-01,
            -2.9431e-01, -9.9687e-01, 1.0690e-01, 2.2631e-01, 6.5402e-01,
            -6.6241e-01, -9.3539e-01, 9.7245e-01, -9.4735e-01, 6.8998e-01,
            9.7441e-01, 1.7277e-01, 7.1798e-01, -9.9265e-01, 9.7759e-01,
            -9.0234e-02, 3.0252e-01, -6.8593e-01, -9.9076e-01, 9.9794e-01,
            -7.5323e-01, 4.3099e-02, -3.2234e-01, -9.9012e-01, -1.5001e-01,
            2.3804e-01, -8.0636e-01, -9.7697e-01, 4.5460e-01, -9.9186e-01,
            -9.9843e-01, 4.5891e-01, -6.7833e-01, 9.9934e-01, 9.7563e-01,
            7.5283e-01, -3.6203e-01, -8.7385e-01, 2.4692e-01, -9.9918e-01,
            -8.3757e-01, -2.8049e-01, -9.8432e-01, 5.3306e-01, 9.7881e-01,
            9.8923e-01, -4.7582e-01, -5.9262e-01, -9.1677e-01, 6.8716e-01,
            9.6552e-01, 7.9985e-01, -8.0383e-01, 2.1775e-01, -3.4245e-01,
            -9.8674e-01, -8.6857e-01, 9.9401e-01, 7.2730e-01, 9.9228e-01,
            -7.0997e-01, -5.1830e-01, 2.3733e-01, -1.4184e-01, 3.1063e-01,
            -9.9998e-01, -7.3541e-01, -3.6047e-01, -9.9777e-01, 9.9836e-01,
            -9.9851e-01, -3.7026e-01, 5.5749e-01, 3.7417e-02, 9.8192e-01,
            6.2351e-01, -9.9819e-01, -9.9760e-01, -9.1619e-01, -2.8208e-01,
            9.9302e-01, -1.0444e-01, 5.2516e-01, 6.0417e-01, 3.9913e-01,
            9.9740e-01, -8.6631e-02, 5.4695e-01, 8.0354e-01, 9.9623e-01,
            2.6266e-01, -9.9267e-01, 4.3445e-01, -9.9539e-01, 1.3555e-01,
            9.7253e-01, 9.7545e-01, 9.8620e-01, 7.5062e-01, 9.9642e-01,
            -9.9910e-01, 9.9984e-01, -9.9768e-01, 7.1753e-01, 9.9820e-01,
            -9.9023e-01, 5.3449e-02, -9.9130e-01, 1.5742e-01, -5.3399e-01,
            2.9825e-01, -7.5114e-01, 9.8778e-01, -9.8921e-01, -9.9584e-01,
            7.4022e-01, 8.1353e-01, 8.5749e-01, -3.2263e-01, 1.6313e-01,
            9.8887e-01, 2.7423e-01, 9.6701e-01, -4.7427e-01, 2.5282e-01,
            9.9999e-01, -4.7095e-01, 4.5189e-01, -9.8996e-01, 9.8648e-01,
            3.3287e-01, 4.1250e-01, 9.5950e-01, -4.1976e-01, 4.3225e-01,
            8.1792e-01, -9.9305e-01, -5.2045e-01, -9.5636e-01, -3.8701e-01,
            -2.9471e-01, 5.5507e-01, 3.8675e-01, -6.7569e-02, 1.9664e-01,
            -9.9155e-01, 3.2475e-01, -9.9240e-01, 9.8300e-01, 2.4997e-01,
            2.7541e-01, -1.5502e-01, 7.4260e-01, -9.5090e-01, 9.9453e-01,
            9.9773e-01, -1.0000e+00, 4.2920e-01, 9.8291e-01, 7.2747e-01,
            9.7929e-01, -9.8909e-01, -4.2928e-01, 7.4067e-01, -4.6398e-01,
            9.8308e-01, 6.2295e-01, -4.1026e-01, 9.7675e-01, -9.8712e-01,
            -3.8594e-02, -8.7963e-01, -2.3659e-01, 8.0683e-01, -9.7641e-01,
            2.5411e-01, -4.0663e-01, 3.3587e-01, -9.9825e-01, -8.0989e-01,
            -9.9746e-01, -3.2973e-01, 9.9367e-01, 6.0861e-01, 9.9761e-01,
            7.9804e-01, -5.1881e-01, 4.8415e-02, -9.9606e-01, 5.6737e-01,
            3.5793e-01, -7.8557e-02, 5.7549e-01, -8.1908e-01, 9.1376e-02,
            -1.6113e-01, -9.9811e-01, 2.5686e-01, -2.5253e-01, 4.0164e-01,
            8.3421e-01, 8.1444e-01, -9.7899e-01, -7.2405e-01, -8.9206e-01,
            4.5792e-01, -6.7538e-01, -8.9475e-01, 9.5371e-01, -7.9438e-02,
            9.9841e-01, -9.8829e-01, -9.1986e-01, -9.3689e-01, -4.9114e-01,
            3.8935e-01, 3.9528e-01, 3.7317e-01, 6.7705e-02, -5.2909e-01,
            -5.2766e-01, 9.7323e-01, -9.8629e-01, -9.8944e-01, 9.9382e-01,
            -2.7280e-01, 1.0328e-01, -5.3651e-01, -5.3561e-01, -5.7003e-01,
            -3.7845e-01, 5.5365e-01, -9.1640e-01, -5.5639e-01, -9.9978e-01,
            -2.9526e-01, 5.6871e-01, -9.8106e-01, -8.7977e-01, -4.9732e-01,
            -9.9999e-01, 9.8781e-01, 9.2619e-01, 9.9667e-01, -9.9727e-01,
            9.2196e-01, 3.4629e-01, 9.9274e-01, -4.1276e-01, -6.3577e-01,
            4.3152e-01, 9.9514e-01, 3.9023e-01, -5.9220e-01, 2.1778e-01,
            -1.8334e-01, 1.4113e-01, -8.6677e-01, -4.7582e-01, 6.1070e-01,
            7.2937e-01, -9.7049e-01, -9.9766e-01, 9.9777e-01, -3.6022e-01,
            9.8406e-01, 5.9269e-01, 2.6457e-01, -9.6541e-01, 4.0841e-01,
            3.4999e-01, -8.9449e-01, -9.9814e-01, 7.8095e-01, -1.0000e+00,
            -9.9097e-01, 3.4224e-01, 9.7176e-01, -9.9229e-01, -9.8083e-01,
            1.7807e-01, -9.9868e-01, 3.8393e-01, 1.1461e-01, 7.1896e-01,
            -9.9238e-01, -7.2016e-01, -6.3698e-01, 3.5324e-01, 9.7574e-01,
            -9.6327e-01, 1.1743e-01, -8.2479e-01, 9.8710e-01, 3.6649e-02,
            4.7560e-01, 1.2970e-01, 3.9256e-01, 5.7329e-01, -9.5178e-01,
            3.4018e-01, -9.9095e-01, -9.4817e-01, 9.9007e-01, 9.8360e-01,
            -9.9589e-01, -9.8852e-01, -3.8393e-01, -3.1365e-01, 9.9262e-01,
            -4.8500e-01, -9.9797e-01, -9.9840e-01, 1.0328e-01, -7.0594e-01,
            9.9292e-01, -4.1498e-01, 9.9997e-01, 9.4807e-01, 5.9744e-01,
            -2.2524e-01, 6.7606e-01, 5.4553e-01, 3.9002e-01, -4.6390e-01,
            9.9857e-01, -1.8173e-01, 9.8774e-01],
        Atom(b45, [
            Structure(s, [List([F2, 'c', E2, Q2, E2, 'h', 'c', F2, C2, 'q']),
                          List([F2, E2, Q2, E2, F2, C2])])]): [-0.6333, 0.7279, 0.9994, -0.9895, 0.8537, -0.0866, 0.9803, 0.8225,
            -0.9765, -0.7353, 0.9453, 0.9985, 0.0382, -0.9962, -0.4276, -0.9761,
            0.9865, -0.1705, -0.9996, -0.7453, 0.4019, -0.9989, 0.2618, 0.8165,
            0.9655, 0.2950, 0.9886, 0.9987, 0.9353, 0.8862, 0.4954, -0.9934,
            0.2893, -0.9958, -0.1171, -0.4972, -0.6741, -0.0572, 0.6030, -0.6539,
            -0.9269, 0.4618, -0.8940, -0.4075, -0.6538, 0.7023, 0.6071, -0.0699,
            -0.3806, 0.9997, -0.9566, 1.0000, 0.2986, 0.9999, 0.9951, 0.8451,
            0.9934, 0.5333, 0.2974, 0.9008, 0.9869, -0.4758, 0.9683, -0.6659,
            -0.6507, -0.8981, 0.4827, 0.3499, -0.8343, 0.9167, 0.8895, 0.2867,
            0.9973, -0.9599, 0.0477, -0.9443, 0.8738, -0.9994, 0.9795, 0.9990,
            -0.8299, -0.9965, 0.9966, -0.2802, 0.4965, -0.9205, 0.2218, -0.9973,
            0.2483, -0.9532, -0.5708, -0.9757, -0.1172, 0.4755, 0.9987, 0.6183,
            -0.2235, 0.1892, 0.4482, 0.3771, -0.8681, -0.4894, -0.4456, 0.6220,
            -0.0588, -0.5856, -0.9672, -0.3871, -0.7516, 0.0182, 0.9939, -0.9898,
            -0.6680, 0.2027, -0.2447, -0.2309, 0.9850, 0.8566, -0.2587, 0.9985,
            -0.0539, 0.3697, 0.9958, -0.4861, 0.7117, -0.4702, 0.3973, 0.0900,
            0.4276, -0.9101, 0.6653, -0.9920, 0.5779, 0.9957, -0.2462, 0.9994,
            -0.9967, 0.3777, -0.9996, 0.3219, 0.8702, -0.1688, -0.2676, 0.8939,
            0.9953, 0.1153, 0.1076, 0.5845, -0.7580, 0.7017, 0.8270, 0.9439,
            -0.9795, 1.0000, 0.2893, 0.0855, 0.5512, -0.0164, -0.0432, -0.0345,
            0.9416, -0.9956, -0.0663, 0.1336, 0.9962, 0.9897, 0.0222, 0.3216,
            0.9983, 0.9209, 0.5645, -0.8182, -0.8237, -0.0212, 0.2155, 0.3264,
            0.9872, 0.9944, -0.9904, 0.9936, 0.9999, -0.6329, 0.9686, -0.4184,
            -0.9984, -0.9914, -0.9860, 0.5679, -0.9458, -0.1916, -0.7013, 0.9345,
            -0.5305, 0.9594, -0.9827, -0.1899, 0.9869, -0.0980, 0.9987, 0.8377,
            -0.9983, 0.0253, -0.5147, 0.9515, -0.3520, 0.9886, 0.5333, 0.5177,
            0.7953, -1.0000, -0.5870, -0.7268, -0.7715, 0.8534, 0.9912, 0.2205,
            -0.1892, -0.1214, 0.2524, 0.9993, -0.9946, -0.3249, 0.8722, -0.9953,
            -0.9976, 0.9652, -0.1106, -0.1337, -0.6633, -0.9530, -0.0321, 0.5340,
            0.9853, 0.4310, 0.8823, -0.9992, 0.7759, -0.9848, 0.0808, 0.5084,
            0.5560, -0.5334, -0.9699, 0.3301, 0.9320, 0.4519, 0.3437, -0.7301,
            0.9128, 0.4065, -0.6282, -0.1990, -0.9866, 0.9971, 0.5645, -0.2718,
            0.9860, -0.9823, 0.2386, -0.0422, -0.0878, -1.0000, -0.5450, -0.8810,
            0.7511, -0.3713, 0.9950, -0.9645, -0.1139, 0.0855, -0.9990, 0.9388,
            -0.3453, 0.9967, -0.8852, 0.7546, 0.9973, 0.6863, -0.9797, -0.9993,
            0.4843, 0.9999, -0.9877, -0.1145, 0.9993, 0.5494, -0.9670, -0.9302,
            -0.9975, -0.9967, -0.6303, 0.1531, -0.3063, 0.9921, -0.5926, -0.2948,
            0.9882, 1.0000, -0.4898, 0.0674, 0.1960, -0.9466, -1.0000, 0.5138,
            0.1183, -0.9991, 0.9978, -0.9906, 1.0000, -0.9359, 0.5039, 0.8526,
            -0.1926, 0.1514, 0.1524, 0.9993, 0.9825, -0.2883, 0.0343, -0.2814,
            0.4878, -0.6770, 0.8744, 0.6403, 0.0475, -0.9023, -0.4246, -0.1495,
            -0.9946, -0.2744, 0.4081, 0.9339, -0.5014, 0.9615, 0.9861, -0.0045,
            0.8833, -0.3730, -0.9991, -0.0031, 0.2741, -0.5375, 0.7429, 0.6994,
            0.9920, -0.9217, 0.9999, -0.3164, 0.3576, 0.3753, 0.9995, -0.9998,
            0.1707, -0.3983, 0.7091, 0.8858, 0.9927, -0.0516, 0.5302, -0.5511,
            0.3478, 0.9536, 0.9873, -0.2445, -0.2945, 0.7974, 0.9184, 0.9919,
            -0.3362, -0.0333, -0.8385, 0.4456, 0.9634, -0.2004, 0.1727, -0.4213,
            0.0912, 0.2212, 0.1788, 0.8672, 0.2742, 0.3886, -0.9929, -0.7533,
            0.5741, 0.9970, -0.9915, -0.6028, 0.9961, -0.2844, 0.4434, 0.8055,
            -0.1519, -0.9650, -0.2620, -0.9991, -0.1574, -0.8314, 0.8754, 0.7860,
            0.1553, 0.3572, -0.1323, -0.8484, 0.9399, 0.4097, 0.9869, 0.8049,
            -0.4035, -0.2132, 0.8078, 0.2467, -0.0798, 0.9942, -0.9895, 0.9995,
            -0.2729, -0.9985, 0.6671, 0.5164, -0.9988, -0.2962, -0.9998, 0.9857,
            -0.7470, 0.3072, 0.3205, -0.9999, -1.0000, -0.5803, -0.5283, -0.4425,
            -0.8534, 0.9984, 0.1928, 0.8247, -0.1732, -0.9469, 0.1960, 0.2838,
            -0.0822, -0.9993, 0.1449, 0.4451, 0.6702, -0.0930, -0.9308, 0.9150,
            -0.9688, 0.7482, 0.9829, 0.6253, 0.8868, -0.9955, 0.9901, -0.1586,
            0.2446, -0.7391, -0.9926, 0.9990, -0.1888, -0.1324, -0.4600, -0.9929,
            0.0047, -0.1449, -0.7747, -0.9915, 0.3611, -0.9939, -0.9997, 0.2527,
            -0.1264, 0.9997, 0.9787, 0.6688, -0.0422, -0.9416, 0.1765, -0.9997,
            -0.6107, 0.0562, -0.9877, 0.7177, 0.9778, 0.9872, -0.7215, -0.5458,
            -0.9349, 0.6989, 0.9585, 0.8784, -0.6758, 0.0787, -0.3734, -0.9875,
            -0.9608, 0.9964, 0.3510, 0.9929, -0.3198, 0.0878, 0.2549, -0.3198,
            0.0043, -1.0000, -0.8098, -0.2711, -0.9992, 0.9993, -0.9994, -0.5656,
            0.7157, -0.2841, 0.9861, 0.6726, -0.9993, -0.9993, -0.9195, -0.3925,
            0.9934, -0.0663, 0.2774, 0.7449, 0.6081, 0.9986, 0.6158, 0.7741,
            0.5038, 0.9984, 0.2714, -0.9950, 0.5172, -0.9980, 0.6581, 0.9677,
            0.9887, 0.9852, 0.2695, 0.9988, -0.9996, 0.9999, -0.9993, 0.2526,
            0.9989, -0.9958, 0.3622, -0.9966, -0.0648, -0.6330, 0.1697, -0.7836,
            0.9951, -0.9970, -0.9976, 0.8337, 0.7510, 0.9305, -0.0218, 0.3849,
            0.9903, 0.2278, 0.9806, -0.5852, 0.2568, 1.0000, -0.5452, 0.2736,
            -0.9914, 0.9933, 0.5734, 0.1404, 0.9651, -0.4019, 0.3614, 0.8463,
            -0.9943, -0.4409, -0.9924, 0.1883, -0.4720, 0.4984, 0.2719, -0.2676,
            0.2125, -0.9887, 0.5237, -0.9943, 0.9891, -0.2099, 0.0614, 0.0826,
            0.9356, -0.8745, 0.9958, 0.9985, -1.0000, 0.1764, 0.9836, 0.7980,
            0.9841, -0.9898, -0.5305, 0.7785, -0.7331, 0.9899, 0.7723, -0.2914,
            0.9790, -0.9921, 0.2630, -0.8972, -0.2563, 0.8332, -0.9853, 0.6205,
            0.0033, 0.2182, -0.9988, -0.7356, -0.9982, -0.4407, 0.9940, 0.6394,
            0.9992, 0.8201, -0.5692, 0.0714, -0.9980, 0.3325, 0.4306, -0.4833,
            0.7998, -0.4111, 0.3591, -0.4268, -0.9984, 0.0651, 0.3362, 0.2510,
            0.8893, 0.8994, -0.9961, -0.8027, -0.9085, 0.3651, -0.7525, -0.8891,
            0.8597, -0.0425, 0.9992, -0.9916, -0.9520, -0.9806, -0.8050, 0.5358,
            0.4208, 0.3819, -0.1099, -0.0860, -0.6879, 0.9839, -0.9902, -0.9940,
            0.9977, -0.2308, 0.3129, -0.5393, -0.4782, -0.6424, -0.2065, 0.5254,
            -0.9411, -0.3750, -0.9999, -0.9019, 0.3188, -0.9913, -0.9151, -0.5954,
            -1.0000, 0.9942, 0.9544, 0.9986, -0.9989, 0.9344, 0.4216, 0.9947,
            -0.3640, -0.5341, 0.2348, 0.9976, 0.6415, -0.4617, -0.1321, -0.1381,
            -0.0571, -0.8871, -0.1746, 0.5657, 0.8403, -0.9659, -0.9984, 0.9985,
            -0.2167, 0.9927, 0.5250, 0.1956, -0.9649, 0.5081, 0.2023, -0.9240,
            -0.9994, 0.9127, -1.0000, -0.9938, 0.5208, 0.9790, -0.9940, -0.9818,
            0.0608, -0.9993, 0.2776, -0.1532, 0.7284, -0.9946, -0.1029, -0.5638,
            0.4115, 0.9879, -0.9733, 0.3200, -0.6661, 0.9963, 0.0760, 0.4792,
            0.1072, 0.0104, 0.3572, -0.9765, -0.5619, -0.9963, -0.9436, 0.9968,
            0.9824, -0.9981, -0.9938, -0.3294, -0.6456, 0.9930, -0.7225, -0.9986,
            -0.9993, 0.2666, -0.9004, 0.9951, -0.3321, 1.0000, 0.9495, 0.8345,
            -0.5186, 0.6581, 0.6917, 0.7257, -0.3923, 0.9992, 0.1690, 0.9939],
        Atom(b45, [Structure(s, [List(
            ['o', N2, A2, 'z', 'g', 'f']),
            List(
                [N2, A2])])]): [-0.6163, 0.6654, 0.9971, -0.9868, 0.7959, -0.0906, 0.9855, 0.4276,
            -0.9411, -0.7618, 0.9687, 0.9911, 0.2056, -0.9943, -0.7627, -0.9887,
            0.9741, -0.4587, -0.9986, -0.5401, 0.4223, -0.9953, 0.5477, 0.7814,
            0.9688, 0.2820, 0.9772, 0.9957, 0.9522, 0.8762, 0.3690, -0.9904,
            0.3857, -0.9857, 0.0775, -0.6199, -0.5847, 0.1728, 0.3890, 0.1565,
            -0.9516, 0.7641, -0.5165, -0.5289, -0.4740, 0.7989, 0.4678, -0.1215,
            -0.1869, 0.9996, -0.9710, 1.0000, 0.4536, 0.9998, 0.9891, 0.8731,
            0.9832, 0.3865, 0.1871, 0.8696, 0.9790, -0.4057, 0.9199, -0.6656,
            -0.6794, -0.9215, -0.0042, 0.2894, -0.5632, 0.9130, 0.6436, 0.3322,
            0.9958, -0.9332, -0.0169, -0.9407, 0.4924, -0.9983, 0.9615, 0.9979,
            -0.6569, -0.9953, 0.9872, -0.0937, 0.4045, -0.8235, 0.6287, -0.9903,
            0.2939, -0.8910, -0.2473, -0.9726, -0.0790, 0.3085, 0.9979, 0.7044,
            -0.2941, 0.3725, 0.2900, 0.2500, -0.7592, -0.2370, -0.7020, 0.8734,
            -0.0074, 0.0473, -0.9789, -0.3036, -0.6214, 0.1416, 0.9934, -0.9867,
            -0.6056, 0.1061, -0.4527, 0.2864, 0.9785, 0.6962, -0.1116, 0.9975,
            -0.1503, -0.4941, 0.9886, -0.7391, 0.7938, -0.1513, 0.0505, -0.3817,
            0.3407, -0.9035, 0.7923, -0.9640, 0.7575, 0.9866, -0.3693, 0.9984,
            -0.9947, -0.2074, -0.9991, -0.0739, 0.9143, -0.1586, 0.3536, 0.7527,
            0.9869, 0.2816, 0.2673, 0.5132, -0.3460, 0.1400, 0.6906, 0.9067,
            -0.9561, 1.0000, 0.2485, -0.2778, 0.4427, 0.2531, 0.2719, -0.6936,
            0.9788, -0.9843, -0.8926, 0.6752, 0.9900, 0.9674, -0.1351, 0.1940,
            0.9929, 0.7009, 0.5346, -0.7187, -0.6909, 0.5951, 0.3709, 0.3873,
            0.9629, 0.9954, -0.9841, 0.9882, 0.9998, -0.6126, 0.9066, -0.5013,
            -0.9924, -0.9828, -0.9676, 0.3049, -0.8542, -0.1916, -0.5216, 0.9600,
            -0.8028, 0.9097, -0.9689, -0.2332, 0.9803, -0.3219, 0.9963, 0.9353,
            -0.9919, 0.4006, -0.5741, 0.9658, -0.3672, 0.9821, 0.3903, 0.4725,
            0.5441, -1.0000, -0.7644, -0.5718, -0.7969, 0.8246, 0.9783, 0.5508,
            -0.3960, -0.1772, 0.3736, 0.9987, -0.9836, -0.6880, 0.7286, -0.9831,
            -0.9923, 0.9670, -0.1549, 0.4989, -0.4191, -0.9480, -0.0463, -0.1613,
            0.9773, 0.4788, 0.7585, -0.9967, 0.8825, -0.9610, -0.0550, 0.4104,
            0.6465, -0.5859, -0.9662, 0.7125, 0.9031, -0.5840, 0.4146, -0.5488,
            0.7331, 0.8096, -0.2987, -0.3994, -0.9825, 0.9927, 0.8504, -0.2660,
            0.9809, -0.9793, -0.0228, 0.4527, -0.1030, -1.0000, -0.1872, -0.8587,
            0.8817, -0.2438, 0.9806, -0.9493, 0.5252, 0.4771, -0.9963, 0.9174,
            -0.4997, 0.9877, -0.8844, 0.7080, 0.9927, 0.6123, -0.9763, -0.9978,
            0.6871, 0.9999, -0.9649, -0.1699, 0.9970, 0.8182, -0.9650, -0.9293,
            -0.9951, -0.9865, -0.6908, 0.0141, 0.0556, 0.9878, 0.3334, -0.2850,
            0.9741, 0.9999, -0.3286, 0.4204, 0.1795, -0.8805, -1.0000, 0.5728,
            0.1060, -0.9984, 0.9914, -0.9876, 1.0000, -0.9072, 0.5950, 0.8631,
            0.0129, 0.1152, 0.1838, 0.9976, 0.9793, -0.4897, 0.3277, -0.4737,
            0.2512, -0.5915, 0.9165, 0.5420, -0.1152, -0.8525, 0.1329, -0.4018,
            -0.9836, 0.1820, 0.4914, 0.9447, -0.3668, 0.9062, 0.9718, -0.2939,
            0.6281, -0.3768, -0.9992, 0.2843, 0.3285, -0.5658, 0.5403, 0.8662,
            0.9857, -0.9732, 0.9999, -0.3977, 0.4110, 0.6947, 0.9989, -0.9995,
            0.3821, -0.0066, 0.6914, 0.7770, 0.9834, -0.3081, 0.1029, 0.5978,
            0.2155, 0.9610, 0.9849, 0.0508, -0.1285, 0.8702, 0.3864, 0.9848,
            -0.5989, 0.1619, -0.6894, 0.3122, 0.9701, 0.6195, -0.6687, -0.2585,
            -0.0082, 0.0675, -0.6466, 0.5512, -0.0947, 0.2862, -0.9880, -0.3163,
            0.7091, 0.9961, -0.9859, -0.1518, 0.9878, -0.2917, 0.2302, 0.7093,
            -0.1300, -0.9317, -0.3563, -0.9956, -0.2241, -0.5605, 0.5748, 0.5368,
            0.0122, -0.0427, 0.3008, -0.9214, 0.8689, 0.3133, 0.9749, 0.7520,
            -0.0838, -0.2063, 0.5360, 0.2233, -0.2904, 0.9926, -0.9898, 0.9984,
            -0.1100, -0.9945, 0.8958, 0.3218, -0.9983, 0.0707, -0.9996, 0.9661,
            -0.7662, 0.6489, 0.8158, -0.9997, -1.0000, -0.2837, -0.3629, -0.2017,
            -0.8205, 0.9966, 0.1003, 0.5563, -0.0557, -0.9372, 0.3465, 0.7748,
            -0.0933, -0.9974, 0.3479, 0.5726, 0.2574, -0.3642, -0.9210, 0.9953,
            -0.9066, 0.7170, 0.9522, 0.4505, 0.7399, -0.9964, 0.9850, 0.2477,
            0.3778, -0.6846, -0.9705, 0.9973, -0.8851, 0.2098, -0.2458, -0.9915,
            0.1186, 0.1971, -0.7048, -0.9719, 0.1829, -0.9948, -0.9973, 0.4653,
            -0.6526, 0.9993, 0.9733, 0.7677, -0.3035, -0.9385, 0.1164, -0.9992,
            -0.7989, -0.1368, -0.9815, 0.6256, 0.9821, 0.9865, -0.0901, -0.5349,
            -0.9165, 0.6988, 0.9821, 0.8500, -0.6170, 0.1793, -0.2188, -0.9821,
            -0.8908, 0.9900, 0.7248, 0.9850, -0.7114, -0.3843, -0.0567, -0.1001,
            0.2937, -1.0000, -0.8645, -0.2809, -0.9972, 0.9971, -0.9984, -0.5223,
            0.5099, -0.1099, 0.9805, 0.6160, -0.9976, -0.9964, -0.9634, -0.2634,
            0.9804, -0.1608, 0.4726, 0.6137, 0.4352, 0.9931, 0.2956, 0.6327,
            0.7809, 0.9938, 0.0765, -0.9839, 0.5076, -0.9957, -0.0938, 0.9627,
            0.8922, 0.9835, 0.7115, 0.9966, -0.9988, 0.9998, -0.9977, 0.7522,
            0.9975, -0.9664, 0.2654, -0.9808, -0.0959, -0.4220, 0.1335, -0.7658,
            0.9831, -0.9851, -0.9876, 0.8000, 0.6952, 0.7555, -0.2815, 0.2000,
            0.9897, 0.2557, 0.9668, -0.5178, 0.1688, 1.0000, -0.0842, 0.6156,
            -0.9906, 0.9830, 0.0442, 0.2600, 0.9590, -0.3401, 0.5527, 0.8224,
            -0.9778, -0.4832, -0.9768, -0.4044, -0.5619, 0.6627, 0.4119, -0.0703,
            0.2158, -0.9692, 0.1175, -0.9936, 0.9924, 0.5297, 0.1734, -0.3078,
            0.8108, -0.9227, 0.9852, 0.9929, -1.0000, 0.3223, 0.9804, 0.5845,
            0.9842, -0.9797, -0.5557, 0.5633, -0.5568, 0.9760, 0.6536, -0.3940,
            0.9746, -0.9708, 0.3789, -0.8884, -0.4311, 0.7232, -0.9583, 0.3546,
            -0.2023, 0.1657, -0.9966, -0.9180, -0.9905, -0.2643, 0.9822, 0.7552,
            0.9975, 0.7632, -0.6442, -0.0458, -0.9958, 0.3793, 0.2977, -0.1008,
            0.6771, -0.6901, 0.2132, -0.2305, -0.9978, 0.3163, -0.1587, 0.2346,
            0.8876, 0.7286, -0.9923, -0.7915, -0.8964, 0.4845, -0.6777, -0.9040,
            0.9756, -0.4402, 0.9984, -0.9805, -0.8606, -0.9425, -0.3565, -0.0402,
            0.3248, 0.2554, 0.3181, -0.3310, -0.2105, 0.9673, -0.9818, -0.9863,
            0.9948, -0.8420, 0.2132, -0.5778, -0.6257, -0.5638, -0.2313, 0.5159,
            -0.8274, -0.3944, -0.9998, -0.4473, 0.4139, -0.9856, -0.9270, -0.5495,
            -1.0000, 0.9857, 0.9176, 0.9959, -0.9965, 0.9521, 0.3865, 0.9920,
            -0.4418, -0.6233, 0.3162, 0.9948, 0.6814, -0.5154, -0.1588, -0.1764,
            -0.0711, -0.7142, -0.3921, 0.8261, 0.6662, -0.9750, -0.9971, 0.9959,
            -0.3914, 0.9784, 0.6779, 0.1206, -0.9520, 0.5628, 0.3423, -0.9306,
            -0.9986, 0.8636, -1.0000, -0.9845, 0.4218, 0.9851, -0.9953, -0.9845,
            0.1452, -0.9986, 0.0350, -0.1033, 0.6618, -0.9777, -0.7433, -0.7133,
            0.2146, 0.9779, -0.9408, 0.4513, -0.8337, 0.9937, 0.0895, 0.4235,
            -0.4274, 0.2380, 0.7785, -0.9627, -0.1862, -0.9847, -0.9524, 0.9811,
            0.9857, -0.9951, -0.9864, -0.1842, -0.3247, 0.9912, -0.6501, -0.9960,
            -0.9975, 0.1335, -0.7430, 0.9911, -0.4087, 1.0000, 0.9531, 0.7483,
            -0.2306, 0.1224, 0.5049, 0.0548, -0.4120, 0.9980, -0.0046, 0.9836],
        Atom(b45, [Structure(s, [List([T2, 'r', 'g', 'y', T2, P2]),
                                 List([T2, T2, P2])])]):
            [-0.3692, 0.7683, 0.9997, -0.9960, 0.9313, -0.5151, 0.9839, 0.8425,
            -0.9883, -0.8399, 0.9867, 0.9967, 0.4960, -0.9991, -0.5442, -0.9926,
            0.9909, -0.4797, -0.9995, -0.4182, 0.5903, -0.9997, 0.5474, 0.7378,
            0.9861, 0.3996, 0.9897, 0.9990, 0.9486, 0.7373, 0.3463, -0.9943,
            0.4904, -0.9982, 0.5211, -0.2921, 0.2226, -0.4004, 0.7226, -0.2301,
            -0.9580, 0.3579, -0.7803, -0.5726, -0.7217, 0.9207, 0.7899, -0.1723,
            -0.2815, 1.0000, -0.9724, 1.0000, 0.5432, 1.0000, 0.9946, 0.9638,
            0.9889, 0.4225, 0.5693, 0.7939, 0.9955, -0.4842, 0.9830, -0.4098,
            -0.5239, -0.9656, -0.0034, 0.6043, -0.7944, 0.9013, 0.7765, 0.4395,
            0.9983, -0.9512, -0.0266, -0.9629, 0.5700, -0.9998, 0.9892, 0.9995,
            -0.5568, -0.9980, 0.9959, -0.4730, 0.4400, -0.8117, 0.8947, -0.9971,
            0.4711, -0.9564, -0.6132, -0.9854, -0.3207, 0.5419, 0.9996, 0.7258,
            -0.5454, 0.5009, 0.2986, 0.6407, -0.8871, 0.0253, -0.9462, 0.9577,
            -0.5213, -0.7007, -0.9904, -0.0266, -0.4052, 0.3716, 0.9979, -0.9978,
            -0.2957, 0.2611, -0.5903, -0.0248, 0.9903, 0.8796, -0.2968, 0.9997,
            -0.0440, -0.5892, 0.9943, 0.0814, 0.8699, -0.5534, 0.0413, 0.0953,
            0.1374, -0.8394, 0.7037, -0.9893, 0.7348, 0.9983, -0.5512, 0.9998,
            -0.9966, -0.3892, -0.9998, 0.6092, 0.6201, -0.0177, 0.4979, 0.7511,
            0.9923, 0.4232, 0.4238, 0.6837, -0.3572, 0.2318, 0.7139, 0.9396,
            -0.9906, 1.0000, -0.3413, -0.0797, 0.3496, 0.1987, 0.5596, -0.4723,
            0.9896, -0.9965, -0.2432, 0.8538, 0.9991, 0.9911, 0.0396, 0.2851,
            0.9991, 0.6239, 0.1095, -0.6388, -0.6127, 0.7981, 0.1308, 0.5411,
            0.9908, 0.9987, -0.9945, 0.9984, 0.9999, -0.6569, 0.9776, -0.7753,
            -0.9980, -0.9968, -0.9887, 0.7134, -0.6182, 0.4012, -0.6999, 0.9714,
            -0.9445, 0.9428, -0.9826, -0.1917, 0.9944, -0.6522, 0.9993, 0.8995,
            -0.9988, -0.4933, -0.5573, 0.9877, -0.6642, 0.9929, 0.0424, 0.6676,
            0.5219, -1.0000, -0.9604, -0.5417, -0.8276, 0.8773, 0.9917, 0.8161,
            -0.5503, 0.0637, 0.3613, 0.9998, -0.9979, -0.7207, 0.6737, -0.9959,
            -0.9948, 0.9902, -0.2063, -0.1476, -0.6849, -0.9657, 0.3553, 0.4404,
            0.9902, 0.5405, 0.8352, -0.9993, 0.9768, -0.9902, 0.1265, 0.3966,
            0.8481, -0.5455, -0.9763, 0.8648, 0.9244, -0.5093, 0.5588, -0.6096,
            0.8657, 0.9572, -0.5793, 0.0412, -0.9959, 0.9981, 0.6413, -0.5607,
            0.9950, -0.9947, -0.1807, 0.8740, 0.1470, -1.0000, -0.5787, -0.7723,
            0.6906, -0.4980, 0.9930, -0.9973, 0.4264, 0.0094, -0.9994, 0.9584,
            -0.6459, 0.9959, -0.9351, 0.3009, 0.9968, 0.4276, -0.9947, -0.9997,
            0.2415, 1.0000, -0.9819, -0.4601, 0.9996, 0.9242, -0.9873, -0.9821,
            -0.9987, -0.9975, -0.5590, 0.3390, -0.1535, 0.9952, -0.2245, -0.2838,
            0.9923, 1.0000, -0.3902, 0.7160, 0.3832, -0.9536, -1.0000, 0.7715,
            0.0079, -0.9998, 0.9987, -0.9946, 1.0000, -0.9394, 0.7442, 0.9280,
            -0.1859, 0.4158, 0.3447, 0.9996, 0.9862, -0.7022, 0.4081, -0.5215,
            0.5301, -0.6456, 0.8923, 0.2786, 0.3136, -0.9235, -0.4927, -0.3506,
            -0.9974, -0.3899, 0.6654, 0.9770, 0.1171, 0.9686, 0.9947, -0.3510,
            0.6056, -0.6369, -0.9996, 0.6513, 0.6374, -0.3335, 0.7685, 0.7890,
            0.9913, -0.9908, 1.0000, -0.5799, 0.3322, 0.8928, 0.9998, -0.9999,
            0.3180, 0.3293, 0.1816, 0.7784, 0.9950, -0.2720, -0.0508, 0.5133,
            0.4411, 0.9739, 0.9921, 0.2066, -0.0388, 0.9756, 0.5813, 0.9959,
            -0.6008, -0.2820, -0.6277, 0.5841, 0.9863, -0.0196, 0.1184, -0.3232,
            0.0155, -0.1991, -0.8419, 0.7002, -0.0023, 0.5179, -0.9949, -0.5121,
            0.2784, 0.9981, -0.9867, -0.4481, 0.9942, -0.5489, 0.1920, 0.8293,
            -0.6389, -0.9856, -0.3947, -0.9993, -0.0150, -0.6875, 0.3057, 0.4269,
            0.2853, 0.0948, 0.1106, -0.9280, 0.9191, 0.4738, 0.9975, 0.8036,
            -0.6249, -0.4306, 0.3906, 0.6037, -0.4478, 0.9974, -0.9974, 0.9998,
            -0.7052, -0.9995, 0.9606, 0.4597, -0.9996, 0.4031, -1.0000, 0.9918,
            -0.8086, 0.8932, 0.9529, -1.0000, -1.0000, -0.5899, 0.1563, -0.4634,
            -0.7707, 0.9993, -0.1752, 0.7885, -0.2615, -0.9732, 0.2892, 0.9300,
            0.0527, -0.9998, 0.4145, 0.0261, 0.8044, -0.4654, -0.9491, 0.9051,
            -0.9793, 0.6685, 0.9932, 0.1588, 0.4490, -0.9985, 0.9939, -0.4866,
            0.4810, -0.7883, -0.9942, 0.9997, -0.6893, -0.0587, -0.3617, -0.9973,
            -0.0727, -0.3858, -0.3365, -0.9973, 0.5564, -0.9973, -0.9997, 0.5350,
            -0.8154, 1.0000, 0.9887, 0.7161, -0.2014, -0.9526, 0.2876, -0.9998,
            -0.6976, 0.1628, -0.9963, 0.0766, 0.9967, 0.9944, -0.1559, -0.4681,
            -0.9297, 0.7081, 0.9864, 0.9164, -0.7928, 0.3380, -0.3440, -0.9957,
            -0.9760, 0.9937, 0.9444, 0.9968, -0.8212, -0.8098, 0.2788, -0.5413,
            0.5507, -1.0000, -0.9219, -0.2625, -0.9997, 0.9998, -0.9997, -0.5996,
            0.6021, 0.3349, 0.9953, 0.8302, -0.9996, -0.9993, -0.6189, -0.2702,
            0.9966, -0.3645, 0.6586, 0.6208, 0.7261, 0.9991, -0.0032, 0.7922,
            0.8756, 0.9979, 0.6491, -0.9974, 0.1268, -0.9990, 0.4942, 0.9826,
            0.9705, 0.9880, 0.9231, 0.9994, -0.9997, 1.0000, -0.9995, 0.9415,
            0.9996, -0.9971, 0.5636, -0.9981, 0.5155, -0.5487, 0.4204, -0.7542,
            0.9943, -0.9967, -0.9957, 0.3827, 0.7635, 0.6616, -0.4622, 0.0036,
            0.9962, 0.4620, 0.9855, -0.3772, -0.0307, 1.0000, -0.1472, 0.5547,
            -0.9829, 0.9925, 0.3072, 0.6376, 0.9714, -0.6547, 0.4898, 0.9454,
            -0.9942, -0.4967, -0.9828, -0.7159, -0.5143, 0.6922, 0.5094, 0.2531,
            0.4144, -0.9954, 0.4855, -0.9986, 0.9916, 0.6160, 0.3936, -0.2553,
            0.6178, -0.9513, 0.9986, 0.9983, -1.0000, 0.3039, 0.9892, 0.4286,
            0.9946, -0.9948, -0.5701, 0.4856, -0.8257, 0.9940, 0.6761, -0.5321,
            0.9927, -0.9926, 0.2921, -0.9836, -0.2815, 0.9357, -0.9741, 0.3387,
            -0.5203, 0.4917, -0.9991, -0.8612, -0.9989, -0.6758, 0.9935, 0.1508,
            0.9989, 0.8242, -0.4914, -0.0687, -0.9981, 0.8698, 0.3758, -0.3755,
            0.2432, -0.9144, 0.2342, -0.2990, -0.9997, 0.5403, -0.3014, -0.0169,
            0.9541, 0.6982, -0.9972, -0.8381, -0.9855, 0.6712, -0.3430, -0.9434,
            0.9186, 0.1386, 0.9997, -0.9865, -0.9792, -0.9852, -0.0679, -0.1712,
            0.3865, 0.1533, 0.4160, -0.4013, -0.6194, 0.9899, -0.9933, -0.9957,
            0.9980, 0.1410, 0.4934, -0.7175, -0.7770, -0.6380, -0.5997, 0.0312,
            -0.9687, -0.5774, -1.0000, -0.0236, 0.2913, -0.9975, -0.9491, -0.5034,
            -1.0000, 0.9945, 0.9730, 0.9992, -0.9990, 0.9714, 0.5306, 0.9979,
            -0.2377, -0.6643, -0.0733, 0.9984, 0.2461, -0.3691, 0.4191, -0.4953,
            -0.0162, -0.8482, -0.6611, 0.1727, 0.5985, -0.9927, -0.9993, 0.9996,
            -0.3696, 0.9842, 0.5871, -0.0276, -0.9824, 0.7350, 0.5095, -0.9736,
            -0.9996, 0.7526, -1.0000, -0.9863, 0.4535, 0.9930, -0.9986, -0.9964,
            0.3723, -0.9996, 0.1840, -0.0489, 0.4425, -0.9950, -0.9155, -0.8944,
            0.7465, 0.9926, -0.9875, -0.1984, -0.8278, 0.9967, 0.4489, 0.5201,
            -0.0729, 0.3241, 0.7448, -0.9765, -0.4071, -0.9959, -0.9782, 0.9934,
            0.9924, -0.9984, -0.9911, -0.6526, -0.2456, 0.9942, -0.5477, -0.9995,
            -0.9993, 0.2119, -0.8161, 0.9955, -0.6450, 1.0000, 0.9698, 0.6147,
            -0.4437, 0.4978, 0.5839, 0.5699, -0.3927, 0.9999, 0.3604, 0.9880],
        Atom(b45, [Structure(s, [List(['x', 'u', 'k', 'l', 'w', 'f', Z2, L2, R2, 'h', U2, 't']),
                                 List([Z2, L2, R2, U2])])]): [-3.5642e-01, 7.6660e-01, 9.9804e-01, -9.9494e-01, 9.0499e-01,
            -7.5924e-02, 9.8771e-01, 6.7606e-01, -9.9053e-01, -8.2386e-01,
            9.7910e-01, 9.9696e-01, 1.6514e-01, -9.9762e-01, -1.5443e-01,
            -9.8710e-01, 9.7647e-01, -5.2677e-01, -9.9920e-01, -3.0467e-01,
            4.6253e-02, -9.9857e-01, 7.0327e-01, 7.9473e-01, 9.2536e-01,
            3.6475e-01, 9.8829e-01, 9.9816e-01, 9.4068e-01, 9.4131e-01,
            4.5969e-01, -9.9236e-01, 5.2615e-01, -9.9396e-01, 3.2172e-01,
            -7.5438e-01, -2.9330e-01, -1.9097e-01, 3.4246e-01, 1.2360e-02,
            -9.4904e-01, 7.3913e-01, -7.4709e-01, -6.0934e-01, -7.5999e-01,
            5.7728e-01, 7.7190e-01, -1.3718e-01, -4.1254e-01, 9.9974e-01,
            -9.6841e-01, 1.0000e+00, 3.7518e-01, 9.9993e-01, 9.9732e-01,
            7.1795e-01, 9.9587e-01, 4.8021e-01, 1.6566e-01, 8.5616e-01,
            9.7707e-01, -6.4629e-01, 9.3063e-01, -7.8192e-01, -7.7522e-01,
            -8.3453e-01, 5.4631e-01, 5.1753e-01, -6.8438e-01, 8.1773e-01,
            8.5668e-01, 6.2769e-01, 9.9788e-01, -9.5106e-01, -2.9411e-03,
            -9.4868e-01, 6.6852e-01, -9.9882e-01, 9.3677e-01, 9.9882e-01,
            -6.1839e-01, -9.9764e-01, 9.9545e-01, -3.0852e-01, 2.3155e-01,
            -9.2127e-01, 5.1421e-01, -9.9642e-01, 5.4880e-01, -9.2188e-01,
            -7.2359e-01, -9.8064e-01, 6.6980e-02, 6.0088e-01, 9.9902e-01,
            7.7835e-01, -3.0128e-01, 5.3148e-01, 5.5257e-01, 7.2668e-01,
            -9.2779e-01, -4.0721e-01, -7.2828e-01, 8.1525e-01, 1.9153e-01,
            -3.6316e-01, -9.5336e-01, -2.4462e-01, -7.6407e-01, 2.4267e-01,
            9.9506e-01, -9.7039e-01, -6.3154e-01, 3.2712e-01, -6.4711e-01,
            1.6297e-01, 9.8311e-01, 7.2734e-01, -3.8989e-01, 9.9884e-01,
            -3.7888e-01, 2.1614e-01, 9.9572e-01, -1.8777e-01, 8.1291e-01,
            -5.7708e-01, 2.5392e-01, -2.5964e-02, 6.7536e-01, -8.9752e-01,
            7.5323e-01, -9.8706e-01, 5.5676e-01, 9.9555e-01, -4.0875e-01,
            9.9901e-01, -9.9780e-01, 4.8788e-01, -9.9955e-01, 8.2109e-02,
            9.1104e-01, -9.5648e-02, -6.2856e-02, 7.2674e-01, 9.9514e-01,
            1.9828e-01, 1.9257e-01, 7.7122e-01, -7.7570e-01, 1.6956e-01,
            6.2530e-01, 9.6272e-01, -9.7374e-01, 1.0000e+00, 4.3770e-02,
            4.0874e-01, 5.1137e-01, 8.8532e-02, -2.2835e-02, -3.1978e-01,
            9.7829e-01, -9.9313e-01, -4.3599e-01, 4.2975e-01, 9.9727e-01,
            9.9034e-01, -4.1268e-01, 1.3809e-01, 9.9478e-01, 8.1737e-01,
            4.5455e-01, -7.5706e-01, -7.9213e-01, 3.2661e-01, 4.4214e-01,
            4.1534e-01, 9.6873e-01, 9.9786e-01, -9.8159e-01, 9.9678e-01,
            9.9994e-01, -7.9680e-01, 9.4329e-01, -2.2924e-01, -9.9653e-01,
            -9.9489e-01, -9.8972e-01, 6.9869e-01, -8.6204e-01, -5.1368e-01,
            -6.5210e-01, 9.7051e-01, -8.1820e-01, 8.7727e-01, -9.9133e-01,
            -2.0789e-01, 9.9138e-01, -2.7586e-01, 9.9856e-01, 6.6018e-01,
            -9.9543e-01, 1.5215e-01, -6.0283e-01, 9.8019e-01, -5.0130e-01,
            9.9246e-01, 4.3456e-01, 4.9584e-01, 8.3106e-01, -1.0000e+00,
            -7.9427e-01, -3.6073e-01, -7.7528e-01, 8.9853e-01, 9.8772e-01,
            5.2539e-01, -3.6070e-01, -5.8385e-02, 7.0146e-01, 9.9914e-01,
            -9.9236e-01, -5.5297e-01, 7.7900e-01, -9.9364e-01, -9.9724e-01,
            9.7886e-01, -3.6700e-01, 5.8317e-01, -6.4135e-01, -9.6184e-01,
            -3.7618e-02, 1.3213e-01, 9.8460e-01, 2.6646e-01, 9.1801e-01,
            -9.9915e-01, 8.2945e-01, -9.7939e-01, 2.6474e-03, 4.3428e-01,
            6.4185e-01, -7.4259e-01, -9.4198e-01, 6.6336e-01, 9.4775e-01,
            -2.8342e-01, 3.3252e-01, -6.7862e-01, 8.9031e-01, 7.7529e-01,
            -2.2198e-01, -1.0867e-01, -9.8822e-01, 9.9686e-01, 7.5361e-01,
            1.1020e-01, 9.9620e-01, -9.9523e-01, 3.0135e-01, 4.3068e-01,
            1.5686e-01, -1.0000e+00, -3.0560e-01, -9.0196e-01, 8.3643e-01,
            -4.5869e-01, 9.9221e-01, -9.5472e-01, -6.2964e-02, 1.5931e-01,
            -9.9863e-01, 9.6188e-01, -5.5338e-01, 9.9473e-01, -7.9253e-01,
            7.7925e-01, 9.9729e-01, 4.5961e-01, -9.8862e-01, -9.9931e-01,
            1.5182e-01, 9.9997e-01, -9.8668e-01, -3.9033e-01, 9.9913e-01,
            7.9198e-01, -9.3637e-01, -9.7219e-01, -9.9763e-01, -9.9559e-01,
            -7.0510e-01, -4.6162e-01, -1.1446e-01, 9.8840e-01, -3.1973e-01,
            -2.9803e-01, 9.8532e-01, 9.9998e-01, -4.6495e-01, 4.1253e-01,
            2.8706e-01, -9.5282e-01, -9.9998e-01, 4.1392e-01, 9.9388e-04,
            -9.9932e-01, 9.9634e-01, -9.8967e-01, 1.0000e+00, -7.7038e-01,
            6.6178e-01, 9.0948e-01, -2.9336e-01, -1.9188e-01, 3.1360e-01,
            9.9896e-01, 9.7895e-01, -6.9626e-01, 2.9957e-01, -4.1101e-01,
            4.6430e-01, -5.6387e-01, 8.2304e-01, 5.5155e-01, 3.4525e-01,
            -8.8261e-01, -1.3387e-01, -5.1747e-01, -9.9164e-01, 4.2790e-01,
            5.8490e-01, 9.5711e-01, 1.4041e-01, 8.8506e-01, 9.9261e-01,
            -3.8461e-01, 8.0386e-01, -5.9554e-01, -9.9844e-01, 2.9939e-01,
            5.4077e-01, -4.9458e-01, 6.5518e-01, 8.7005e-01, 9.9200e-01,
            -9.7813e-01, 9.9996e-01, -3.6967e-01, 6.2519e-01, 5.3242e-01,
            9.9953e-01, -9.9990e-01, 3.9473e-01, -4.5451e-01, 6.5825e-01,
            8.1847e-01, 9.9293e-01, 6.6598e-02, 4.3072e-01, -4.1398e-01,
            8.2090e-02, 9.6482e-01, 9.8911e-01, 1.2645e-01, -5.6353e-02,
            8.9163e-01, 6.4408e-01, 9.9002e-01, -4.6839e-01, -3.7076e-01,
            -8.1218e-01, 2.9742e-01, 9.6873e-01, 7.2706e-02, -6.7865e-03,
            -3.6139e-01, 3.5492e-01, -2.0560e-01, -5.2090e-01, 5.4854e-01,
            3.7934e-01, 3.0959e-01, -9.9418e-01, -6.5549e-01, 7.0399e-01,
            9.9768e-01, -9.9171e-01, -5.1928e-01, 9.9184e-01, -4.6645e-01,
            5.1544e-01, 8.8863e-01, 5.8067e-02, -9.5258e-01, -4.1317e-01,
            -9.9840e-01, 4.0927e-02, -6.8958e-01, 5.5002e-01, 8.6650e-01,
            1.2266e-01, -1.6570e-01, 9.0898e-02, -7.8863e-01, 7.8153e-01,
            5.7119e-01, 9.7681e-01, 8.7184e-01, -4.2890e-01, -2.0510e-01,
            6.1690e-01, 4.9376e-01, -3.9510e-01, 9.9446e-01, -9.9013e-01,
            9.9916e-01, -5.5511e-01, -9.9800e-01, 8.6015e-01, 3.5947e-01,
            -9.9882e-01, -2.0606e-01, -9.9982e-01, 9.9250e-01, -8.3251e-01,
            6.9434e-01, 6.9950e-01, -9.9991e-01, -1.0000e+00, -5.0334e-01,
            -6.4284e-01, -2.3603e-01, -8.0309e-01, 9.9534e-01, 1.5388e-01,
            7.5789e-01, -1.8876e-01, -9.5614e-01, 4.5651e-01, 6.6139e-01,
            -5.7446e-02, -9.9823e-01, 3.4970e-01, 5.9826e-01, 5.0540e-01,
            -4.5573e-01, -9.5681e-01, 9.8325e-01, -9.7443e-01, 8.0929e-01,
            9.7422e-01, 5.8732e-01, 7.7499e-01, -9.9559e-01, 9.8553e-01,
            -2.0552e-02, 5.2394e-01, -6.1216e-01, -9.8887e-01, 9.9860e-01,
            -8.3695e-01, 2.1054e-01, -3.8064e-01, -9.9403e-01, 2.5474e-01,
            1.7155e-01, -6.7165e-01, -9.8927e-01, 6.3884e-01, -9.9225e-01,
            -9.9955e-01, 6.8616e-01, -4.4052e-01, 9.9979e-01, 9.8424e-01,
            7.8972e-01, -3.8530e-01, -9.3944e-01, 1.0806e-01, -9.9939e-01,
            -6.9365e-01, 2.1264e-01, -9.8301e-01, 7.8265e-01, 9.7458e-01,
            9.9259e-01, -4.4926e-01, -6.2724e-01, -8.5727e-01, 7.1431e-01,
            9.5077e-01, 8.9012e-01, -8.1954e-01, 3.4116e-01, -2.9396e-01,
            -9.9106e-01, -8.7401e-01, 9.9687e-01, 6.6584e-01, 9.8841e-01,
            -5.5308e-01, -1.5925e-01, 2.6246e-01, -4.8569e-01, -4.2412e-03,
            -9.9999e-01, -7.1471e-01, -4.4133e-01, -9.9826e-01, 9.9817e-01,
            -9.9896e-01, -5.5848e-01, 6.2926e-01, 1.2637e-01, 9.8282e-01,
            7.7568e-01, -9.9905e-01, -9.9871e-01, -9.8457e-01, -2.1744e-01,
            9.9257e-01, -5.2820e-02, 5.2959e-01, 3.7136e-01, 8.2491e-01,
            9.9766e-01, -4.6222e-01, 5.4691e-01, 7.6726e-01, 9.9786e-01,
            6.1361e-01, -9.9624e-01, 6.2854e-01, -9.9769e-01, 3.0867e-01,
            9.6189e-01, 9.7428e-01, 9.8549e-01, 6.7416e-01, 9.9897e-01,
            -9.9939e-01, 9.9991e-01, -9.9862e-01, 6.6149e-01, 9.9911e-01,
            -9.9068e-01, 4.2313e-01, -9.9492e-01, 4.8248e-02, -8.4531e-01,
            4.6001e-01, -8.1793e-01, 9.9391e-01, -9.9374e-01, -9.9567e-01,
            4.2780e-01, 5.4645e-01, 8.3804e-01, 3.9530e-02, 5.3903e-01,
            9.9512e-01, 3.4930e-01, 9.8053e-01, -3.3840e-01, 3.8896e-01,
            1.0000e+00, -4.4561e-01, 3.1001e-01, -9.9626e-01, 9.7062e-01,
            1.3133e-01, 5.7270e-01, 9.3847e-01, -4.6251e-01, 4.9201e-01,
            8.2547e-01, -9.9315e-01, -6.1516e-01, -9.8278e-01, -2.8685e-01,
            -2.2857e-01, 6.1953e-01, 5.6592e-01, 1.9295e-01, 3.0250e-01,
            -9.9049e-01, 4.7806e-01, -9.9536e-01, 9.8773e-01, 1.9362e-01,
            3.2996e-01, -2.8732e-01, 7.9672e-01, -9.5784e-01, 9.9585e-01,
            9.9898e-01, -1.0000e+00, 4.8625e-01, 9.8812e-01, 7.1833e-01,
            9.7948e-01, -9.8858e-01, -4.2404e-01, 7.5967e-01, -5.7348e-01,
            9.8429e-01, 7.7599e-01, -5.4579e-01, 9.7485e-01, -9.9338e-01,
            1.6065e-01, -8.8388e-01, -4.8017e-01, 9.1432e-01, -9.8552e-01,
            3.6012e-01, -2.3056e-01, 4.5164e-01, -9.9822e-01, -8.1553e-01,
            -9.9837e-01, -4.6604e-01, 9.9648e-01, 5.9210e-01, 9.9905e-01,
            6.8530e-01, -6.6040e-01, -9.4691e-02, -9.9792e-01, 3.0526e-01,
            3.8308e-01, -2.8369e-01, -5.0559e-02, -6.7507e-01, 3.9323e-01,
            -6.4471e-01, -9.9907e-01, 3.8718e-01, 7.1557e-03, 2.8559e-01,
            9.0519e-01, 7.3779e-01, -9.7480e-01, -6.4177e-01, -8.9919e-01,
            7.2473e-01, -7.1246e-01, -9.7441e-01, 9.4043e-01, -2.6014e-01,
            9.9916e-01, -9.9403e-01, -7.8587e-01, -9.0498e-01, -6.7437e-01,
            -1.3853e-01, 4.6233e-01, 9.2619e-02, 4.5829e-01, -3.0873e-01,
            -6.2325e-01, 9.8342e-01, -9.9093e-01, -9.9110e-01, 9.9650e-01,
            -5.8760e-01, -9.6743e-02, -3.9513e-01, -6.8297e-01, -6.5163e-01,
            -4.8904e-01, 5.0691e-01, -9.2200e-01, -5.9916e-01, -9.9995e-01,
            -7.4408e-01, 3.5011e-01, -9.8429e-01, -9.3719e-01, -7.5029e-01,
            -1.0000e+00, 9.9319e-01, 9.6080e-01, 9.9767e-01, -9.9854e-01,
            9.4866e-01, 5.6957e-01, 9.9777e-01, -4.3564e-01, -8.0363e-01,
            4.0401e-01, 9.9736e-01, 8.1165e-01, -5.6418e-01, 1.6239e-01,
            -3.0295e-01, -2.3164e-01, -8.2747e-01, -3.4624e-01, 3.6424e-01,
            8.1976e-01, -9.7799e-01, -9.9863e-01, 9.9905e-01, -5.0400e-01,
            9.9008e-01, 7.2547e-01, 2.8944e-01, -9.5664e-01, 7.0221e-01,
            -1.0357e-01, -8.7453e-01, -9.9918e-01, 7.5942e-01, -1.0000e+00,
            -9.9382e-01, 7.4309e-01, 9.9120e-01, -9.9655e-01, -9.7025e-01,
            4.2957e-01, -9.9918e-01, -6.8676e-02, -2.0194e-01, 6.0002e-01,
            -9.9553e-01, -5.8757e-01, -7.8175e-01, 4.7940e-01, 9.8424e-01,
            -9.8006e-01, 5.0472e-01, -7.8733e-01, 9.9655e-01, 5.8023e-02,
            6.0490e-01, -5.6005e-02, 7.9968e-02, 4.1414e-01, -9.7203e-01,
            -1.8991e-01, -9.9579e-01, -9.2973e-01, 9.9308e-01, 9.7989e-01,
            -9.9732e-01, -9.9396e-01, 8.3315e-02, -2.5526e-01, 9.9527e-01,
            -6.4836e-01, -9.9898e-01, -9.9843e-01, -1.2412e-02, -7.9976e-01,
            9.9608e-01, -4.3265e-01, 9.9999e-01, 9.4814e-01, 6.2720e-01,
            -7.2097e-01, 4.3903e-01, 6.5930e-01, 5.4907e-01, -4.0927e-01,
            9.9933e-01, 1.6813e-01, 9.9457e-01]
    }


    encoding2 = encoding2.encoding(primitives2)

    primitives = [not_space, mk_uppercase, mk_lowercase, is_empty, is_space, is_uppercase, not_uppercase, is_lowercase,
                  not_lowercase, is_letter, not_letter, is_number, not_number, skip1, copy1, write1, copyskip1]
    learner = SimpleBreadthFirstLearner(prolog,encoding2)
    #primitives = [None, mk_uppercase, None,None, None, None, None, None,
    #              None, None,None,None,None,skip1, copy1, None, copyskip1]
    decoder = decoding.decoding(primitives)
    #network1 = Network.Network(3, [2619, 100, 17])
    #output = learner.readEncodingsOfFile('/home/marlon/loreleai/Neural2/[1, 1, 1, 1, 1]output500.txt')
    #input = learner.readEncodingsOfFile('/home/marlon/loreleai/Neural2/[1, 1, 1, 1, 1]input500.txt')
    #print('train')
    #tic = time.perf_counter()
    #network1.Train(input,output,1)
    network1 = Network.Network.LoadNetwork('/home/marlon/loreleai/Neural2/network(b291-1b)20.txt')
    #toc = time.perf_counter()
    #print(f"Downloaded the tutorial 2 in {toc - tic:0.10f} seconds")
    print(network1.GetNumOutputs())
    network2 = Network.Network.LoadNetwork('/home/marlon/loreleai/Neural2/network(b291-2b)20.txt')
    print(network2.GetNumOutputs())
    learner.assert_knowledge(background)
    head = Atom(b45, [A])
    body1 = Atom(not_uppercase, [A])
    body2 = Atom(not_space, [A])
    body3 = Atom(skip1, [A,C])
    body4 = Atom(not_space, [C])
    body5 = Atom(skip1, [C, D])
    body6 = Atom(not_space, [D])
    body7 = Atom(write1, [B, C,A])
    body8 = Atom(write1, [E, D, C])
    body9 = Atom(write1, [A, G, H])
    body10 = Atom(copy1, [G, I])
    clause1 = Clause(head, Body(body1,body2,body3,body4,body5,body6,body7,body8,body9,body10))

    print('coverage clause 1 : ' , learner.evaluateClause(covered,clause1))
    body1 = Atom(not_uppercase, [A])
    body2 = Atom(not_space, [A])
    body3 = Atom(skip1, [A, C])
    body4 = Atom(not_space, [C])
    body5 = Atom(skip1, [C, D])
    body6 = Atom(not_space, [D])
    body7 = Atom(write1, [B, C, A])
    body8 = Atom(not_space, [B])
    body9 = Atom(write1, [D,F,G])
    body10 = Atom(write1, [F,H,I])
    clause2 = Clause(head, Body(body1, body2, body3, body4, body5, body6, body7, body8, body9, body10))
    print('coverage clause 2 : ', learner.evaluateClause(covered, clause2))
    #print(covered)
    #print(pos.difference(covered))
    #print(len(pos.difference(covered)))
    #program = learner.learn(task, background, hs,network1,network2,encodingExamples,decoder,primitives)
    #print(program)
    #print(random.randint(0,6500))
    #print(random.randint(0, 6500))
    #print(random.randint(0, 6500))
    #print(random.randint(0, 6500))
    #print(random.randint(0, 6500))
    #print(random.randint(0, 6500))


