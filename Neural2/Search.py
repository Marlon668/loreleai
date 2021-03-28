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
        numbernegative = 0
        total = 0

        for prim in primitives:
            print(prim)
            exps = plain_extension(Clause(head,[]).get_body(),prim)
            lengte = len(exps)
            for i in range(0,lengte):
                exps[i] = Clause(head,exps[i])
                if not self.badClause(exps[i].get_head,exps[i].get_body()):
                    y = self.evaluate(examples,exps[i])
                    self._expansion += 1
                    self._expansionOneClause +=1
                    self._result.append((self._expansion,y[0]))
                    print(y)
                    if y[0] >0:
                        self.put_into_pool(1 - y[0], 0, 0, exps[i])
                        if (best == None) or y[0]>best[0]:
                            best = (y[0],exps[i])

        current_cand = None
        found = False
        pos, _ = task.get_examples()
        state = self.encodeState(pos)



        while current_cand is None or (
                len(self._candidate_pool) > 0) and self._expansionOneClause < 100:
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
                    total +=1
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
                        else:
                            numbernegative += 1
                for expy in toBeAdded:
                    if len(expy[2]) < self._max_body_literals:
                        self.put_into_pool(1 - expy[0],1-expy[1],1-exp._expansion, expy[2])
        print('negative : ' , numbernegative)
        print('total : ', total)
        print('percentage : ' , numbernegative/total)
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

    def evaluateClause(self, examples, clause: Clause):
            numberofpositivecoverance = 0
            self._solver.assertz(clause)
            for example in examples:
                if self._solver.has_solution(example):
                    print(example)
                    numberofpositivecoverance += 1
            self._solver.retract(clause)
            return numberofpositivecoverance

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
        total = 0
        numberNegative = 0

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
            bestk = sorted(expansions, reverse=True)[0:6]
            for exp in bestk:
                exps = hypothesis_space.expandwithprimitive(current_cand,exp._clause)
                bestkClause = []
                for exp2 in exps:
                    addedLiteral = exp2.get_literals()[-1]
                    y = self._neural2.FeedForward([*self._encoder.encode2(exp2), *state])
                    bestkClause.append(mytuple(y[0],0,exp2))
                if len(bestkClause)>7:
                    bestkClause = sorted(bestkClause, reverse=True)[0:7]
                else:
                    sorted(bestkClause, reverse=True)
                toBeAdded = []
                for i in range(0,len(bestkClause)):
                    y = self.evaluate(examples,bestkClause[i]._clause)
                    total +=1
                    if (y[1] == 0 )& (y[0]>0):
                        print('found')
                        print(bestkClause[i]._clause)
                        return bestkClause[i]._clause
                    else:
                        if y[0]>0:
                            toBeAdded.append((y[0],bestkClause[i]._value,bestkClause[i]._clause))
                        else:
                            numberNegative +=1
                for exp in toBeAdded:
                    if len(exp[2]) < self._max_body_literals:
                        self.put_into_pool(1 - exp[0],1-exp[1], exp[2])
        print
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
    b45 = c_pred("b45", 2)
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

    #learner = SimpleBreadthFirstLearner(prolog, max_body_literals=4)

    # create the hypothesis space
    hs = TopDownHypothesisSpace(primitives=[
        lambda x: plain_extension(x, not_space, connected_body=True),
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
            Structure(s, [List([M2, S2, space2, H2, E2, R2, M2, I2, O2, N2, E2, space2, G2, R2, A2, N2, G2, E2, R2]),
                          List([M2, 's', space2, G2, 'r', 'a', 'n', 'g', 'e', 'r'])])])
        : [-0.2781, 0.6987, 0.9962, -0.9904, 0.7330, 0.2802, 0.9755, 0.0176,
            -0.9587, -0.4745, 0.9579, 0.9959, -0.6831, -0.9916, -0.3643, -0.9754,
            0.9877, -0.6198, -0.9988, -0.7973, -0.4209, -0.9973, 0.5917, 0.9586,
            0.9527, 0.1145, 0.9913, 0.9971, 0.8988, 0.5626, 0.4200, -0.9923,
            0.8336, -0.9929, 0.2988, -0.3810, 0.2731, 0.0120, 0.7613, -0.6406,
            -0.7205, -0.1814, -0.0781, -0.6823, -0.0223, 0.3609, 0.3988, 0.0380,
            -0.2528, 0.9936, -0.9118, 0.9996, -0.5469, 0.9986, 0.9955, 0.8055,
            0.9934, 0.2085, -0.6068, 0.7159, 0.9731, -0.4350, 0.8816, -0.2034,
            -0.0580, -0.8204, -0.4178, 0.3743, -0.3146, 0.8256, 0.4890, 0.4463,
            0.9939, -0.8793, 0.0597, -0.9365, -0.2496, -0.9988, 0.9541, 0.9970,
            0.4883, -0.9942, 0.9953, -0.3014, -0.5808, 0.0098, -0.5253, -0.9915,
            0.4872, -0.8130, 0.2347, -0.9877, 0.4543, 0.1116, 0.9973, 0.1652,
            -0.2541, 0.2323, 0.8216, -0.2831, -0.6363, 0.6435, 0.5321, 0.1773,
            0.4930, 0.3521, -0.9593, -0.8121, -0.1362, 0.1838, 0.9915, -0.9605,
            -0.8833, -0.0807, 0.4454, -0.8256, 0.9936, 0.9557, -0.3940, 0.9975,
            -0.3648, 0.2157, 0.9951, 0.0555, 0.0523, -0.3994, -0.4632, 0.6140,
            -0.2623, -0.8320, 0.7865, -0.9883, -0.0628, 0.9961, -0.2538, 0.9981,
            -0.9961, 0.7105, -0.9990, -0.5205, 0.5975, -0.0633, -0.1061, 0.2487,
            0.9942, 0.1678, -0.5378, -0.3891, 0.0377, -0.4941, 0.3577, 0.8838,
            -0.9749, 0.9974, 0.7729, 0.6993, 0.7969, 0.0897, -0.2588, 0.1072,
            0.9842, -0.9883, -0.5121, -0.0358, 0.9946, 0.9809, 0.8408, -0.7821,
            0.9982, 0.0183, 0.1013, -0.3947, -0.3817, -0.1865, 0.4161, 0.3009,
            0.9409, 0.9852, -0.9928, 0.9729, 0.9761, -0.5153, 0.9287, 0.6487,
            -0.9954, -0.9921, -0.9806, 0.0382, 0.0674, 0.4170, 0.0542, 0.9450,
            0.0300, 0.7437, -0.9596, -0.1254, 0.9695, -0.3458, 0.9968, 0.4978,
            -0.9973, -0.3804, 0.0184, 0.9597, -0.3482, 0.9661, -0.5344, 0.1220,
            0.9332, -0.9993, -0.0708, -0.4270, -0.3806, 0.8004, 0.9922, -0.2296,
            -0.4835, -0.2191, 0.2361, 0.9981, -0.9930, -0.6651, 0.6283, -0.9913,
            -0.9945, 0.9827, -0.2923, -0.4224, -0.5068, -0.6449, -0.0609, 0.7713,
            0.9825, -0.4038, -0.0940, -0.9990, 0.3718, -0.9327, -0.7313, 0.1548,
            0.5767, -0.5833, -0.9795, -0.2420, 0.9033, 0.7153, -0.3380, -0.5115,
            0.2157, -0.1442, 0.3030, -0.6225, -0.9752, 0.9956, 0.2578, 0.4027,
            0.9771, -0.9843, 0.7393, -0.4487, 0.0735, -0.9991, 0.2083, -0.0400,
            0.1298, -0.3265, 0.9907, -0.9679, -0.3685, 0.8571, -0.9968, 0.8763,
            -0.4689, 0.9893, -0.5652, -0.3202, 0.9962, 0.9036, -0.9864, -0.9986,
            0.9594, 0.9653, -0.9841, 0.0993, 0.9989, -0.0847, -0.9242, -0.9708,
            -0.9966, -0.9971, -0.0443, -0.6744, -0.3039, 0.9925, 0.5765, -0.1519,
            0.9881, 0.9995, -0.1293, -0.2027, -0.0024, -0.9283, -0.9967, 0.9736,
            0.1539, -0.9985, 0.9966, -0.9902, 0.9990, -0.7632, -0.5763, 0.5855,
            -0.0673, -0.4635, 0.4786, 0.9978, 0.9774, -0.6012, 0.1512, 0.3360,
            0.1563, 0.0742, 0.4392, -0.3845, -0.0447, -0.7797, 0.6511, 0.6592,
            -0.9911, 0.7155, 0.5286, 0.8979, -0.8220, 0.9518, 0.9954, 0.0190,
            0.2120, -0.0138, -0.9486, -0.6520, 0.4896, -0.9541, -0.0377, 0.8616,
            0.9832, -0.9788, 0.9986, -0.2905, 0.8476, -0.2907, 0.9978, -0.9990,
            0.2161, 0.4128, -0.2029, 0.2304, 0.9896, 0.6930, 0.8897, 0.5727,
            -0.3176, 0.8148, 0.9779, -0.3760, -0.0215, -0.0840, -0.3977, 0.9805,
            0.1852, -0.2360, -0.0902, -0.6839, 0.9711, -0.4181, -0.7518, -0.3234,
            -0.5526, 0.7308, 0.2621, -0.4944, 0.7029, 0.1680, -0.9903, 0.4751,
            0.9562, 0.9951, -0.9907, -0.2057, 0.9842, -0.2183, -0.4972, 0.7599,
            0.7875, -0.9739, -0.3326, -0.9975, -0.2146, -0.1533, 0.2832, -0.3546,
            0.2892, -0.5724, 0.8030, -0.7095, 0.8960, 0.2042, 0.9789, 0.4348,
            -0.1946, -0.2793, 0.2340, 0.5401, -0.3132, 0.9936, -0.9724, 0.9975,
            0.0622, -0.9961, 0.3406, -0.4864, -0.9956, 0.7876, -0.9981, 0.9857,
            -0.7173, -0.3995, -0.2368, -0.9988, -0.9992, 0.8559, 0.5006, -0.2887,
            -0.5385, 0.9055, -0.0027, 0.0294, 0.0190, -0.9459, 0.1841, -0.0684,
            0.5423, -0.9983, -0.6873, 0.9111, -0.5560, -0.8110, -0.9440, 0.7545,
            -0.8083, 0.5871, 0.9744, -0.3869, -0.4660, -0.9825, 0.9818, -0.6874,
            0.1631, -0.9363, -0.9837, 0.9981, -0.2439, 0.1083, -0.2183, -0.9755,
            0.5086, -0.7434, -0.9249, -0.9827, 0.3609, -0.9911, -0.9977, 0.3467,
            -0.0494, 0.9970, 0.9599, 0.5431, -0.5174, -0.9392, 0.2893, -0.9987,
            -0.3513, 0.7956, -0.9818, -0.4427, 0.9886, 0.9916, -0.7989, -0.7743,
            -0.6554, 0.4418, 0.9750, -0.1399, -0.6134, 0.3995, -0.3079, -0.9898,
            -0.8954, 0.9965, -0.4051, 0.9911, -0.0959, 0.4664, -0.1478, 0.1620,
            -0.6100, -0.9988, -0.7869, -0.0079, -0.9982, 0.9972, -0.9981, -0.1570,
            -0.1068, 0.7447, 0.9758, 0.6029, -0.9981, -0.9985, -0.8253, 0.3506,
            0.9943, -0.0594, 0.4133, 0.3685, 0.1790, 0.9959, -0.0449, -0.2685,
            -0.1980, 0.9972, 0.0837, -0.9951, 0.9070, -0.9937, 0.9366, 0.9864,
            0.9422, 0.9203, -0.3975, 0.9975, -0.9990, 0.9996, -0.9961, -0.2524,
            0.9979, -0.9892, -0.6673, -0.9878, -0.8042, 0.0670, 0.4735, -0.6767,
            0.9942, -0.9904, -0.9960, 0.3672, 0.3555, 0.2256, 0.7322, -0.4311,
            0.9945, 0.2602, 0.9783, 0.0782, 0.8867, 0.9999, -0.8429, 0.2217,
            -0.9918, 0.9916, -0.4721, 0.4325, 0.9488, -0.3135, 0.1276, 0.7153,
            -0.9911, -0.2777, -0.8948, 0.6778, 0.0265, 0.5677, 0.2503, -0.2161,
            0.0161, -0.9970, 0.3165, -0.9963, 0.9934, -0.2876, 0.5566, -0.1211,
            0.3323, -0.9044, 0.9946, 0.9967, -0.9997, 0.3209, 0.9879, -0.2477,
            0.9861, -0.9878, -0.4238, 0.9663, -0.2601, 0.9629, 0.4240, -0.2325,
            0.9936, -0.9884, -0.5349, -0.7930, -0.0621, 0.6793, -0.9545, 0.0183,
            0.6223, 0.1337, -0.9972, -0.7112, -0.9966, -0.2161, 0.9935, -0.3448,
            0.9970, 0.1712, -0.3533, -0.0987, -0.9954, -0.6812, -0.0013, 0.1822,
            -0.5481, 0.2009, 0.0021, 0.6650, -0.9937, 0.1874, 0.7503, 0.2760,
            0.7073, -0.2129, -0.9866, -0.9599, -0.7350, 0.4788, 0.2539, -0.8703,
            0.8364, -0.7567, 0.9988, -0.9831, -0.9385, -0.9755, 0.3039, 0.6223,
            0.4314, 0.2273, -0.5834, 0.6081, -0.7870, 0.9838, -0.9919, -0.9953,
            0.9978, -0.1641, -0.7473, 0.3772, -0.6043, -0.1532, -0.3328, 0.8597,
            -0.8804, -0.5220, -0.9996, 0.3692, -0.5992, -0.9921, -0.9140, 0.1098,
            -0.9985, 0.9954, 0.9488, 0.9966, -0.9960, 0.9247, 0.5731, 0.9968,
            -0.3134, -0.7066, 0.5335, 0.9955, -0.5496, 0.1704, 0.0049, -0.2797,
            0.6293, -0.4152, 0.5417, 0.0667, 0.2936, -0.9858, -0.9971, 0.9961,
            -0.2477, 0.9874, 0.4632, 0.7761, -0.9345, 0.9561, -0.1890, -0.9296,
            -0.9986, 0.4133, -0.9996, -0.9944, 0.2342, 0.9947, -0.9963, -0.9854,
            -0.0753, -0.9979, 0.8009, -0.6535, -0.4362, -0.9921, 0.3049, -0.5027,
            -0.7795, 0.9744, -0.9706, 0.7393, -0.6422, 0.8402, 0.0548, 0.4172,
            -0.5947, -0.8404, 0.1459, -0.9763, 0.3811, -0.9873, -0.9575, 0.9924,
            0.9776, -0.9958, -0.9790, 0.4823, 0.4268, 0.9916, -0.6530, -0.9977,
            -0.9986, 0.0306, -0.5929, 0.9932, -0.3658, 0.9980, 0.9142, 0.0983,
            0.5506, -0.2584, 0.1526, -0.1592, -0.2938, 0.9982, -0.8250, 0.9892],
        Atom(b45, [Structure(s, [List(
            [P2, R2, O2, F2, E2, S2, S2, O2, R2, space2, M2, I2, N2, E2, R2, V2, A2, space2, M2, C2, G2, O2, N2, A2,
             G2, A2, L2, L2]),
            List(
                [P2, 'r', 'o', 'f', 'e', 's', 's', 'o', 'r', space2, M2, 'c', 'g', 'o', 'n',
                 'a', 'g', 'a', 'l', 'l'])])])
        : [-6.0647e-01, 7.0167e-01, 9.9876e-01, -9.9364e-01, 8.8290e-01,
            2.5290e-01, 9.8172e-01, 4.1366e-01, -9.8121e-01, -6.4425e-01,
            9.5639e-01, 9.9801e-01, -3.0718e-01, -9.9850e-01, -5.2722e-01,
            -9.8008e-01, 9.8443e-01, -6.3704e-01, -9.9967e-01, -5.4850e-01,
            3.1209e-01, -9.9933e-01, 5.9739e-01, 7.9737e-01, 9.7425e-01,
            3.6413e-02, 9.9011e-01, 9.9934e-01, 8.4953e-01, 9.0886e-01,
            3.9349e-01, -9.9417e-01, 2.3505e-01, -9.9610e-01, 2.4344e-01,
            -4.9034e-01, -4.0463e-01, -4.8388e-02, 3.2546e-01, -6.0633e-01,
            -8.2086e-01, 5.5035e-01, -5.5307e-01, -6.3841e-01, -7.2883e-01,
            6.3231e-01, 6.2479e-01, 1.2351e-01, -5.9944e-01, 9.9929e-01,
            -9.6132e-01, 1.0000e+00, -1.6921e-01, 9.9991e-01, 9.9758e-01,
            8.3069e-01, 9.9501e-01, 2.8568e-01, -1.6720e-01, 9.1905e-01,
            9.7910e-01, -4.8775e-01, 9.4478e-01, -7.0960e-01, -4.7010e-01,
            -8.6118e-01, 2.3795e-01, 4.6285e-01, -4.7812e-01, 8.8731e-01,
            7.6561e-01, 5.0175e-01, 9.9433e-01, -9.0793e-01, -1.4292e-01,
            -9.3706e-01, 4.5983e-01, -9.9953e-01, 9.6247e-01, 9.9933e-01,
            -1.7554e-01, -9.9866e-01, 9.9620e-01, -4.2421e-01, 6.0706e-02,
            -8.1314e-01, 1.3104e-01, -9.9634e-01, 4.5339e-01, -9.0839e-01,
            1.3095e-01, -9.8826e-01, 3.0422e-01, 5.8701e-01, 9.9967e-01,
            3.9241e-01, -3.5849e-01, 4.8737e-01, 7.7804e-01, -2.4586e-02,
            -7.8911e-01, 2.7665e-01, -1.9385e-01, 6.6433e-01, 2.3741e-01,
            -1.4694e-01, -9.7426e-01, -3.7927e-01, -6.7469e-01, 2.3062e-01,
            9.8926e-01, -9.8756e-01, -6.9928e-01, 3.1632e-01, 4.7653e-01,
            -5.4219e-01, 9.8960e-01, 5.4498e-01, -4.3669e-01, 9.9950e-01,
            -3.5857e-01, -2.0933e-01, 9.9696e-01, -4.1834e-01, 5.5848e-01,
            -4.7441e-01, 2.3498e-02, 9.8734e-02, 2.9829e-01, -8.8818e-01,
            8.2422e-01, -9.9178e-01, 2.4226e-01, 9.9789e-01, -4.6571e-01,
            9.9964e-01, -9.9860e-01, 6.5031e-01, -9.9976e-01, -4.9367e-01,
            7.7125e-01, 1.9602e-01, -1.6196e-01, 6.4835e-01, 9.9805e-01,
            2.5965e-01, -1.6477e-01, -1.0991e-01, -5.7672e-01, -1.5251e-01,
            6.8301e-01, 9.3031e-01, -9.8353e-01, 9.9998e-01, 3.0991e-01,
            7.3882e-01, 6.4896e-01, 1.6200e-01, -1.3606e-01, -8.6408e-02,
            9.8977e-01, -9.9569e-01, -7.3319e-01, 2.4794e-01, 9.9856e-01,
            9.7319e-01, 3.9729e-01, -2.0107e-01, 9.9919e-01, 5.8797e-01,
            5.1469e-01, -7.1371e-01, -4.4857e-01, 1.4994e-01, 4.5426e-01,
            3.7638e-01, 9.6804e-01, 9.9769e-01, -9.9231e-01, 9.9424e-01,
            9.9962e-01, -4.0112e-01, 9.4652e-01, -4.0532e-03, -9.9756e-01,
            -9.9517e-01, -9.8979e-01, 2.5916e-01, -7.5931e-01, -8.6652e-02,
            -2.2474e-01, 9.6820e-01, -5.6862e-01, 9.1283e-01, -9.8909e-01,
            -1.7017e-01, 9.8286e-01, -2.9433e-01, 9.9938e-01, 8.1089e-01,
            -9.9799e-01, 3.7120e-01, -2.5407e-01, 8.7349e-01, -5.4398e-01,
            9.5349e-01, 9.0745e-02, 3.2742e-01, 8.1768e-01, -1.0000e+00,
            -6.2645e-01, -5.8880e-01, -5.8715e-01, 9.1304e-01, 9.9352e-01,
            -2.2220e-01, -4.0870e-01, 1.3931e-01, 4.7769e-01, 9.9955e-01,
            -9.9661e-01, -7.5133e-01, 7.0273e-01, -9.9081e-01, -9.9669e-01,
            9.8608e-01, -4.6299e-01, -1.0107e-01, -6.5142e-01, -9.1923e-01,
            1.8661e-01, 3.7980e-01, 9.8956e-01, 2.8759e-02, 6.8602e-01,
            -9.9960e-01, 8.1337e-01, -9.5459e-01, -4.2262e-01, 5.4794e-01,
            5.7638e-01, -6.0728e-01, -9.7509e-01, 1.3422e-01, 9.3868e-01,
            4.6035e-01, 1.7448e-01, -8.0482e-01, 6.3124e-01, 4.8005e-01,
            -5.8219e-01, -3.5106e-01, -9.9171e-01, 9.9864e-01, 6.0655e-01,
            1.4871e-01, 9.8766e-01, -9.9326e-01, 2.5296e-01, 2.9669e-01,
            1.5646e-01, -9.9998e-01, -3.3307e-01, -8.3246e-01, 6.8745e-01,
            -3.3702e-01, 9.9295e-01, -9.6268e-01, -4.9541e-01, 5.5990e-01,
            -9.9912e-01, 9.3051e-01, -4.3999e-01, 9.9723e-01, -8.4053e-01,
            5.9287e-01, 9.9710e-01, 7.2907e-01, -9.9129e-01, -9.9949e-01,
            7.0710e-01, 9.9983e-01, -9.9013e-01, -2.7764e-01, 9.9967e-01,
            4.7339e-01, -9.5310e-01, -9.7674e-01, -9.9889e-01, -9.9808e-01,
            -4.3637e-01, -6.0990e-01, -1.8012e-01, 9.9063e-01, -6.9201e-02,
            -2.3625e-01, 9.9563e-01, 9.9996e-01, -2.8364e-01, 3.3689e-01,
            2.5468e-01, -9.6878e-01, -9.9996e-01, 6.1103e-01, 2.9320e-01,
            -9.9944e-01, 9.9908e-01, -9.9558e-01, 1.0000e+00, -8.0887e-01,
            4.0286e-01, 6.9022e-01, -2.4503e-01, -2.8627e-01, 4.4176e-01,
            9.9936e-01, 9.8320e-01, -6.2913e-01, 2.9875e-01, 1.7606e-01,
            1.9604e-01, -3.4047e-01, 7.7687e-01, 2.6691e-01, 3.3180e-01,
            -8.7733e-01, 2.5906e-01, 8.6864e-02, -9.9303e-01, 2.8172e-01,
            4.3580e-01, 8.6426e-01, -6.4285e-01, 9.5792e-01, 9.9370e-01,
            -1.1679e-02, 7.5774e-01, -3.2534e-01, -9.9746e-01, -3.5203e-01,
            5.3669e-01, -7.5752e-01, 3.5787e-01, 6.3122e-01, 9.8550e-01,
            -9.5990e-01, 9.9992e-01, -5.5203e-01, 6.5714e-01, 2.8118e-01,
            9.9968e-01, -9.9966e-01, 2.9301e-01, -8.1365e-02, 5.7121e-01,
            5.3579e-01, 9.9201e-01, 3.2754e-01, 6.9232e-01, 4.7695e-01,
            3.3016e-01, 8.9652e-01, 9.8821e-01, -1.8712e-01, 1.4506e-01,
            6.6431e-01, 3.2145e-01, 9.8952e-01, -4.1639e-01, -3.1645e-01,
            -7.5161e-01, 3.2325e-02, 9.7384e-01, 7.7870e-05, -4.4248e-01,
            -3.3279e-01, -1.5032e-01, 5.2725e-01, -2.9825e-01, 4.7620e-01,
            4.3734e-01, 2.3896e-01, -9.9546e-01, -2.2462e-01, 8.2656e-01,
            9.9876e-01, -9.8743e-01, -5.3963e-01, 9.8960e-01, -3.0005e-01,
            -4.1886e-02, 7.1281e-01, 3.2044e-01, -9.4621e-01, -4.9347e-01,
            -9.9864e-01, 1.5301e-01, -5.7658e-01, 6.7608e-01, 2.2222e-01,
            3.6677e-01, -2.8093e-01, 7.4703e-01, -8.1331e-01, 9.3653e-01,
            3.2275e-01, 9.8299e-01, 7.8289e-01, -2.4359e-01, -1.9535e-01,
            6.0490e-01, 6.3353e-01, -4.8652e-01, 9.9633e-01, -9.7814e-01,
            9.9957e-01, -3.1451e-01, -9.9916e-01, 6.1767e-01, -5.2510e-03,
            -9.9897e-01, 8.2203e-02, -9.9976e-01, 9.8981e-01, -5.4607e-01,
            3.5195e-01, 1.3845e-01, -9.9978e-01, -9.9999e-01, 3.8446e-01,
            -2.4739e-01, -3.8743e-01, -7.8938e-01, 9.9436e-01, 2.3871e-01,
            5.9972e-01, -2.8519e-01, -9.6293e-01, 3.3654e-01, 5.4497e-01,
            1.1429e-01, -9.9955e-01, 5.0310e-03, 6.2773e-01, 1.6201e-01,
            -4.4749e-01, -9.4616e-01, 8.7845e-01, -9.2780e-01, 7.3211e-01,
            9.8519e-01, 2.8577e-01, 5.8792e-01, -9.9753e-01, 9.8874e-01,
            4.6612e-03, 2.1994e-01, -8.1337e-01, -9.8598e-01, 9.9925e-01,
            -1.9447e-01, 2.4264e-01, -3.1200e-01, -9.9375e-01, 3.6094e-01,
            -9.6690e-02, -8.2541e-01, -9.8516e-01, 5.3434e-01, -9.8909e-01,
            -9.9942e-01, 1.9903e-01, -2.0277e-01, 9.9955e-01, 9.7682e-01,
            6.3715e-01, -5.0944e-01, -9.6188e-01, 3.0751e-01, -9.9960e-01,
            -2.9076e-01, 3.2408e-01, -9.5987e-01, 2.0439e-01, 9.9216e-01,
            9.9097e-01, -5.8438e-01, -4.5484e-01, -3.1633e-01, 5.8105e-01,
            9.7802e-01, 5.6386e-01, -7.6407e-01, 4.0148e-01, -4.1778e-01,
            -9.9226e-01, -8.8093e-01, 9.9775e-01, 1.2174e-01, 9.9184e-01,
            -2.2789e-01, 3.7021e-02, 4.4453e-01, -1.2769e-01, -2.1056e-01,
            -9.9998e-01, -7.7478e-01, -1.2767e-01, -9.9911e-01, 9.9941e-01,
            -9.9945e-01, -3.9202e-01, 4.3603e-01, 4.5810e-01, 9.8346e-01,
            7.6426e-01, -9.9905e-01, -9.9938e-01, -9.5960e-01, -2.7036e-01,
            9.9589e-01, -1.8451e-01, 6.3319e-01, 5.4442e-01, 6.0993e-01,
            9.9911e-01, 4.2377e-01, 5.2362e-01, 4.9579e-01, 9.9920e-01,
            5.5720e-01, -9.9776e-01, 7.2790e-01, -9.9693e-01, 7.5605e-01,
            9.7792e-01, 9.4907e-01, 9.7664e-01, 3.1474e-01, 9.9960e-01,
            -9.9951e-01, 9.9986e-01, -9.9935e-01, 3.4643e-01, 9.9926e-01,
            -9.9313e-01, -1.8611e-01, -9.9704e-01, -5.4671e-01, -5.2513e-01,
            4.3026e-01, -7.3001e-01, 9.9739e-01, -9.9536e-01, -9.9756e-01,
            7.4659e-01, 8.3773e-01, 6.6907e-01, 3.7297e-01, 3.4991e-01,
            9.9334e-01, 2.1016e-01, 9.7567e-01, -1.9020e-01, 7.1076e-01,
            1.0000e+00, -6.5501e-01, 1.1019e-01, -9.9517e-01, 9.9029e-01,
            -4.5814e-02, 5.7919e-01, 9.5112e-01, -3.0836e-01, 4.2786e-01,
            6.7596e-01, -9.9638e-01, -2.5711e-01, -9.8824e-01, 1.2719e-01,
            -3.0762e-01, 7.4099e-01, 4.1874e-01, 3.5530e-02, 1.8262e-01,
            -9.9599e-01, 4.6723e-01, -9.9846e-01, 9.9468e-01, -1.7359e-01,
            5.9897e-01, -1.4572e-01, 7.9477e-01, -8.9397e-01, 9.9747e-01,
            9.9734e-01, -1.0000e+00, 5.3243e-01, 9.8578e-01, 3.6480e-01,
            9.8025e-01, -9.8723e-01, -4.8943e-01, 8.8175e-01, -5.2533e-01,
            9.7219e-01, 6.0832e-01, -4.6090e-01, 9.8811e-01, -9.9446e-01,
            -1.9173e-01, -8.5892e-01, -1.8452e-01, 7.2990e-01, -9.8348e-01,
            4.4456e-01, 1.0476e-01, 1.6662e-01, -9.9907e-01, -7.7772e-01,
            -9.9854e-01, -1.6796e-01, 9.9528e-01, 2.2867e-01, 9.9924e-01,
            5.6700e-02, -3.0241e-01, -1.7064e-01, -9.9854e-01, -3.4242e-01,
            2.6246e-01, -1.0848e-01, -7.4343e-03, -3.5167e-01, 4.5704e-01,
            -3.1024e-01, -9.9895e-01, 3.0770e-01, 2.5033e-01, 4.8296e-01,
            7.7193e-01, 4.9552e-01, -9.8790e-01, -8.5971e-01, -9.1356e-01,
            5.3379e-01, -2.3800e-01, -8.1558e-01, 8.2397e-01, -4.5566e-01,
            9.9957e-01, -9.9299e-01, -9.1944e-01, -9.7580e-01, -5.9542e-01,
            4.4665e-01, 3.7309e-01, 1.4469e-02, -3.0672e-01, -1.1431e-01,
            -4.9567e-01, 9.9139e-01, -9.9307e-01, -9.9305e-01, 9.9935e-01,
            -4.8900e-01, -9.8950e-02, -2.0385e-01, -6.2810e-01, -3.1681e-01,
            -3.4125e-01, 5.0720e-01, -5.3799e-01, -4.6424e-01, -9.9994e-01,
            -7.8039e-01, -3.3464e-01, -9.9232e-01, -9.1637e-01, -2.9409e-01,
            -1.0000e+00, 9.9613e-01, 9.7324e-01, 9.9912e-01, -9.9866e-01,
            9.3416e-01, 5.1047e-01, 9.9859e-01, -4.5964e-01, -7.4957e-01,
            2.0644e-01, 9.9842e-01, 1.3321e-01, -2.5123e-01, 1.6538e-01,
            -3.8355e-01, 2.6932e-02, -8.0084e-01, -2.8008e-02, 4.9842e-01,
            6.8707e-01, -9.6341e-01, -9.9863e-01, 9.9912e-01, -5.1821e-01,
            9.8883e-01, 6.1874e-01, 2.9949e-01, -9.5069e-01, 7.7472e-01,
            -3.0964e-01, -9.6495e-01, -9.9962e-01, 5.7864e-01, -1.0000e+00,
            -9.9525e-01, 2.0768e-01, 9.9216e-01, -9.9875e-01, -9.8886e-01,
            1.1145e-01, -9.9961e-01, -3.6389e-02, -3.1772e-01, 4.4770e-01,
            -9.8923e-01, -1.4895e-01, -6.5690e-01, 3.8580e-01, 9.8402e-01,
            -9.8123e-01, 4.7146e-01, -7.7393e-01, 9.8984e-01, 1.6881e-01,
            3.5740e-01, -3.9912e-01, -4.1888e-01, 2.1562e-01, -9.7836e-01,
            -5.0246e-01, -9.9240e-01, -7.7799e-01, 9.9393e-01, 9.8729e-01,
            -9.9765e-01, -9.9235e-01, 1.8600e-01, -1.6214e-01, 9.9338e-01,
            -7.0086e-01, -9.9877e-01, -9.9940e-01, 2.4942e-01, -7.2072e-01,
            9.9652e-01, -5.1564e-01, 9.9998e-01, 9.0094e-01, 5.1567e-01,
            1.7005e-01, 3.5974e-01, 6.2212e-01, 5.1522e-01, -4.2492e-01,
            9.9946e-01, -2.4030e-01, 9.9246e-01],
        Atom(b45, [Structure(s, [List([D2, R2, space2, R2, A2, Y2, space2, S2, T2, A2, N2, T2, Z2]),
                                 List([D2, 'r', space2, S2, 't', 'a', 'n', 't', 'z'])])])
        : [-2.0942e-01, 6.6699e-01, 9.9740e-01, -9.8807e-01, 7.2662e-01,
            -5.2140e-01, 9.6109e-01, 6.3515e-01, -9.6225e-01, -6.2207e-01,
            9.6144e-01, 9.9049e-01, 2.2708e-01, -9.9356e-01, -4.9933e-01,
            -9.6986e-01, 9.5746e-01, -6.5449e-01, -9.9770e-01, -4.0531e-01,
            1.0734e-01, -9.9666e-01, 6.7439e-01, 5.8421e-01, 9.4866e-01,
            2.0535e-01, 9.7151e-01, 9.9633e-01, 8.1164e-01, 9.2236e-01,
            4.6380e-01, -9.8662e-01, 4.8836e-01, -9.9239e-01, 5.4430e-01,
            -5.5173e-01, 5.2909e-02, -7.3077e-02, 4.6339e-01, 8.2434e-02,
            -8.1221e-01, 6.7245e-01, -4.8128e-01, -6.0840e-01, -6.8410e-01,
            3.6772e-02, 7.2865e-01, -5.1182e-02, -3.1730e-01, 9.9947e-01,
            -9.0122e-01, 1.0000e+00, 4.4671e-01, 9.9975e-01, 9.8341e-01,
            6.9239e-01, 9.7335e-01, 3.5371e-01, -5.2153e-04, 6.4170e-01,
            9.6266e-01, -4.1558e-01, 8.4721e-01, -4.8597e-01, -8.5614e-01,
            -3.8473e-01, 2.1519e-01, 5.7436e-01, -5.5880e-01, 7.9682e-01,
            7.3476e-01, 3.1242e-01, 9.9654e-01, -8.9865e-01, -1.2079e-01,
            -8.9660e-01, 8.3435e-01, -9.9759e-01, 9.1863e-01, 9.9746e-01,
            -2.6000e-01, -9.9084e-01, 9.8500e-01, -2.9125e-01, 4.8521e-01,
            -8.3200e-01, 5.8208e-01, -9.8987e-01, 3.0071e-01, -8.4156e-01,
            -3.9662e-01, -9.7648e-01, 1.3757e-01, 6.7275e-01, 9.9825e-01,
            5.3512e-01, -2.5758e-01, 3.1329e-01, 2.9280e-01, 1.0728e-01,
            -8.1466e-01, -3.4724e-03, -5.7783e-01, 6.7002e-01, 6.3286e-03,
            -3.1078e-01, -8.8311e-01, -4.0579e-02, -8.3537e-01, 2.1260e-01,
            9.8632e-01, -9.4636e-01, -6.2350e-01, 2.0781e-01, -3.8555e-01,
            1.5092e-01, 9.6868e-01, 4.2390e-01, -1.1876e-01, 9.9667e-01,
            -2.8140e-01, -2.7215e-01, 9.8683e-01, -4.5476e-01, 8.3919e-01,
            -4.0663e-01, -3.5042e-02, -1.5834e-01, 3.7111e-01, -7.0819e-01,
            5.5071e-01, -9.8108e-01, 6.3058e-01, 9.9399e-01, -2.0176e-01,
            9.9832e-01, -9.9287e-01, 2.4957e-01, -9.9799e-01, -5.6815e-01,
            5.6434e-01, -1.0917e-01, -5.9209e-02, 7.6607e-01, 9.6133e-01,
            2.6616e-01, 2.8838e-01, 3.9112e-01, -3.7874e-01, -6.3053e-02,
            2.0891e-01, 9.3344e-01, -9.5419e-01, 9.9998e-01, -2.9591e-01,
            1.5910e-01, 1.8165e-01, 3.1905e-01, 2.5999e-01, -2.1229e-01,
            9.4449e-01, -9.8343e-01, -1.3357e-01, 7.0272e-01, 9.9247e-01,
            9.8375e-01, 3.6997e-01, 1.0963e-01, 9.9519e-01, 7.9992e-01,
            3.1207e-01, -8.0969e-01, -5.1783e-01, 3.1159e-01, 3.5778e-01,
            5.7730e-01, 9.0842e-01, 9.9727e-01, -9.8030e-01, 9.9651e-01,
            9.9970e-01, -4.4180e-01, 8.8279e-01, -7.9906e-02, -9.9304e-01,
            -9.8886e-01, -9.6717e-01, 5.7587e-01, -6.8216e-01, 2.9792e-01,
            -5.5241e-01, 9.6005e-01, -6.8017e-01, 6.2087e-01, -9.9211e-01,
            -1.6772e-01, 9.6065e-01, -4.1759e-01, 9.9670e-01, 8.2661e-01,
            -9.9415e-01, -1.7549e-01, -3.7734e-01, 9.2305e-01, -4.9972e-01,
            9.5467e-01, 7.7546e-02, 3.1025e-01, 6.4294e-01, -9.9999e-01,
            -5.4155e-01, -7.3985e-01, -8.1234e-01, 8.0617e-01, 9.8934e-01,
            5.6116e-01, -4.5485e-01, -7.5584e-02, 6.1621e-01, 9.9718e-01,
            -9.8564e-01, -7.0817e-01, 6.4534e-01, -9.8382e-01, -9.8572e-01,
            9.5335e-01, -3.2616e-01, -1.6265e-01, -3.7232e-01, -9.5065e-01,
            7.0781e-02, -3.8941e-04, 9.4486e-01, 2.2401e-01, 7.3233e-01,
            -9.9545e-01, 7.8556e-01, -8.7089e-01, 1.7578e-01, 7.1209e-02,
            7.4211e-01, -6.6163e-01, -9.1038e-01, 4.1796e-01, 9.3353e-01,
            -1.4302e-01, -3.2966e-01, -3.7802e-01, 9.1217e-01, 6.4777e-01,
            -3.6934e-01, -5.7230e-02, -9.8237e-01, 9.8969e-01, 5.1470e-01,
            -3.9557e-01, 9.8939e-01, -9.8270e-01, 1.2100e-01, 3.8593e-01,
            -4.2357e-02, -9.9999e-01, -2.4788e-01, -8.1020e-01, 4.9007e-01,
            -1.5985e-01, 9.8035e-01, -9.3639e-01, -7.2748e-02, -1.3308e-01,
            -9.9658e-01, 9.5365e-01, -4.7428e-01, 9.8652e-01, -8.9740e-01,
            9.4116e-01, 9.9213e-01, 5.2853e-01, -9.8010e-01, -9.9709e-01,
            1.3826e-01, 9.9978e-01, -9.6780e-01, -1.4088e-01, 9.9821e-01,
            7.3982e-01, -7.3846e-01, -9.7694e-01, -9.9464e-01, -9.9091e-01,
            -6.2459e-01, -1.9879e-01, 9.6358e-02, 9.8830e-01, 2.8104e-01,
            -1.2485e-01, 9.7155e-01, 9.9991e-01, -2.2789e-01, 4.7399e-01,
            4.0417e-01, -8.8147e-01, -9.9997e-01, 3.2393e-01, 1.9186e-02,
            -9.9780e-01, 9.9555e-01, -9.7136e-01, 9.9999e-01, -7.0781e-01,
            5.1079e-01, 8.6549e-01, -2.5449e-01, 1.7092e-01, 4.4618e-01,
            9.9693e-01, 9.7079e-01, -3.7661e-01, 1.3375e-01, -1.8204e-01,
            2.3871e-01, -5.8770e-01, 8.0582e-01, 3.2942e-01, 1.9289e-01,
            -7.9334e-01, -5.2091e-02, -4.2048e-01, -9.8416e-01, -1.0382e-01,
            5.5321e-01, 8.1395e-01, -5.0970e-02, 7.9010e-01, 9.7987e-01,
            -1.9142e-01, 6.8578e-01, -1.8952e-01, -9.9858e-01, 4.6173e-01,
            5.4375e-01, -4.7645e-01, 7.7011e-01, 2.0033e-01, 9.8921e-01,
            -9.4912e-01, 9.9990e-01, -1.9840e-01, 2.7187e-01, 4.6664e-01,
            9.9824e-01, -9.9968e-01, 3.0067e-01, -1.1839e-01, 2.8660e-01,
            8.1291e-01, 9.8451e-01, 1.5942e-02, 6.0365e-01, 6.5000e-01,
            2.5520e-01, 8.8446e-01, 9.7533e-01, 5.8938e-02, -3.1762e-01,
            8.1577e-01, 7.5108e-01, 9.8754e-01, -4.7648e-01, -4.4851e-01,
            -8.4216e-01, 2.1845e-01, 9.0741e-01, 8.2814e-02, -1.8661e-02,
            -3.2276e-01, -1.7224e-01, 3.1728e-01, -5.1760e-01, 6.9092e-01,
            3.8690e-01, 2.2132e-01, -9.8663e-01, -1.6821e-01, 7.1022e-01,
            9.9493e-01, -9.6370e-01, -1.9108e-01, 9.7667e-01, -4.2588e-01,
            -1.6718e-01, 7.3914e-01, -2.9472e-01, -9.1023e-01, -2.7224e-01,
            -9.9616e-01, -2.6742e-01, -8.3985e-01, 6.7644e-01, 4.4565e-01,
            2.2601e-01, 2.5230e-01, 1.9739e-02, -5.1342e-01, 3.5844e-01,
            -3.2819e-01, 9.3916e-01, 2.2302e-01, -4.3474e-01, -5.4655e-01,
            6.8610e-01, 4.9803e-01, -3.2886e-01, 9.8631e-01, -9.7073e-01,
            9.9892e-01, 1.1778e-01, -9.9409e-01, 8.2092e-01, 2.1541e-01,
            -9.9875e-01, 1.1275e-01, -9.9922e-01, 9.8391e-01, -7.6766e-01,
            6.7652e-01, 7.5888e-01, -9.9963e-01, -9.9998e-01, -1.4020e-01,
            -4.5651e-01, -5.9474e-02, -9.3616e-01, 9.9534e-01, -3.0824e-02,
            3.7342e-01, -2.6866e-01, -8.7376e-01, -2.1981e-01, 6.1835e-01,
            2.1046e-01, -9.9638e-01, 3.5155e-02, 3.9936e-01, 5.8502e-01,
            -1.0746e-01, -9.0856e-01, 8.6830e-01, -9.3436e-01, 6.7476e-01,
            9.7614e-01, 7.8571e-01, 6.5756e-01, -9.8989e-01, 9.7708e-01,
            -4.4654e-02, 1.6780e-01, -6.0453e-01, -9.7334e-01, 9.9711e-01,
            -3.8377e-01, 2.7500e-01, -4.1436e-01, -9.7977e-01, 7.7881e-02,
            -3.9060e-01, -5.3626e-01, -9.8425e-01, 4.6132e-01, -9.3672e-01,
            -9.9860e-01, 5.9372e-01, -5.6619e-01, 9.9939e-01, 9.6431e-01,
            5.9338e-01, -3.1812e-01, -9.5393e-01, 3.2804e-01, -9.9796e-01,
            -3.2698e-02, -1.2278e-01, -9.6480e-01, 7.6666e-01, 9.4290e-01,
            9.8554e-01, 2.4839e-01, -3.2710e-01, -6.9057e-01, 7.6658e-01,
            9.0784e-01, 8.2764e-01, -6.6817e-01, 3.1619e-01, -1.4912e-01,
            -9.7406e-01, -9.0639e-01, 9.8548e-01, 6.3651e-01, 9.7571e-01,
            -5.7931e-01, -1.4440e-01, 2.9412e-01, -6.6646e-01, 2.5593e-01,
            -9.9995e-01, -7.6561e-01, -1.9387e-01, -9.9763e-01, 9.9786e-01,
            -9.9821e-01, -9.7990e-02, 7.5303e-01, 3.2270e-01, 9.6601e-01,
            7.0793e-01, -9.9715e-01, -9.9539e-01, -9.7932e-01, -2.3240e-03,
            9.7249e-01, -7.5725e-02, 2.0594e-01, 6.6617e-01, -1.5959e-02,
            9.8867e-01, 1.6564e-01, 8.7461e-01, 6.9483e-01, 9.9354e-01,
            2.6914e-01, -9.9129e-01, 4.7168e-01, -9.9484e-01, 2.6813e-01,
            9.5946e-01, 9.3299e-01, 9.6648e-01, 4.9852e-01, 9.9752e-01,
            -9.9894e-01, 9.9977e-01, -9.9660e-01, 5.1922e-01, 9.9699e-01,
            -9.7960e-01, 3.3510e-01, -9.8881e-01, 3.7055e-01, -8.3045e-01,
            4.1504e-01, -7.9847e-01, 9.8292e-01, -9.9056e-01, -9.8900e-01,
            8.4564e-01, 5.9163e-01, 6.8982e-01, -1.2365e-01, 2.5741e-01,
            9.9006e-01, 3.7025e-01, 9.7785e-01, -5.1819e-01, 3.8053e-01,
            9.9999e-01, -2.1737e-01, 4.2293e-01, -9.8251e-01, 9.5499e-01,
            3.6034e-01, 2.9889e-01, 8.7643e-01, -2.6580e-01, 4.8429e-01,
            8.0702e-01, -9.7785e-01, -3.2934e-01, -9.3024e-01, 1.0929e-01,
            -3.0473e-01, 5.9015e-01, 3.5354e-01, 1.1022e-01, 3.4459e-01,
            -9.8013e-01, 2.7408e-01, -9.9142e-01, 9.7209e-01, 3.6795e-01,
            4.4164e-01, -1.6262e-01, 8.3543e-01, -9.0227e-01, 9.8498e-01,
            9.9510e-01, -1.0000e+00, 9.9935e-02, 9.8194e-01, 6.8123e-01,
            9.8489e-01, -9.5927e-01, -5.1264e-01, 7.1342e-01, -6.3159e-01,
            9.6713e-01, 7.6422e-01, -4.4568e-01, 9.7760e-01, -9.7343e-01,
            2.4430e-01, -4.7684e-01, -4.0806e-01, 6.6506e-01, -9.6326e-01,
            2.2823e-01, -2.9132e-01, 4.0384e-01, -9.9633e-01, -7.6076e-01,
            -9.9280e-01, -4.1550e-01, 9.9125e-01, 2.2671e-01, 9.9793e-01,
            4.7856e-01, -5.5322e-01, -3.9033e-02, -9.9258e-01, 4.4145e-01,
            2.7163e-01, -2.1724e-01, -3.6243e-01, -6.0242e-01, 5.4481e-01,
            -8.2925e-01, -9.9738e-01, 3.2462e-01, -1.1234e-01, -9.9533e-02,
            5.1910e-01, 6.0728e-01, -9.4267e-01, -7.0601e-01, -7.5313e-01,
            5.2742e-01, -3.5680e-01, -9.3701e-01, 7.7634e-01, -3.3152e-01,
            9.9795e-01, -9.7539e-01, -8.7816e-01, -9.5421e-01, -4.8376e-01,
            -1.6280e-02, 3.2544e-01, 1.8569e-01, 3.5865e-01, -3.3756e-01,
            -3.5749e-01, 9.6683e-01, -9.8387e-01, -9.8279e-01, 9.9249e-01,
            -6.1549e-01, 1.2671e-01, -6.8164e-01, -5.3660e-01, -4.4973e-01,
            -5.0631e-01, 2.7944e-01, -8.1334e-01, -6.1199e-01, -9.9984e-01,
            -3.1761e-01, -1.3656e-01, -9.7584e-01, -8.7396e-01, -5.8373e-01,
            -9.9999e-01, 9.8523e-01, 9.4264e-01, 9.9707e-01, -9.9585e-01,
            8.2310e-01, 5.0408e-01, 9.9145e-01, -2.1545e-01, -6.2055e-01,
            -5.8085e-02, 9.9387e-01, 7.2350e-01, -6.0457e-01, 5.6077e-02,
            -3.0484e-01, -4.1405e-02, -9.2928e-01, -3.8053e-01, 4.4131e-01,
            5.8740e-01, -9.3722e-01, -9.9693e-01, 9.9757e-01, -1.5664e-01,
            9.8364e-01, 6.4468e-01, 4.4041e-01, -8.3508e-01, 6.9855e-01,
            1.3128e-01, -8.0604e-01, -9.9857e-01, 2.2262e-01, -1.0000e+00,
            -9.8779e-01, 2.3786e-01, 9.6787e-01, -9.8996e-01, -9.6933e-01,
            2.6618e-01, -9.9805e-01, -2.6687e-01, 3.7359e-02, 3.2840e-01,
            -9.8253e-01, -4.2811e-01, -4.9723e-01, 6.5876e-01, 9.6562e-01,
            -9.7786e-01, 2.7280e-01, -6.4230e-01, 9.9135e-01, 1.4232e-01,
            1.9284e-01, -4.2615e-01, -2.0228e-01, 4.1499e-01, -9.7790e-01,
            -7.1110e-01, -9.9102e-01, -6.0403e-01, 9.7912e-01, 9.7891e-01,
            -9.9631e-01, -9.8112e-01, -1.8543e-01, -8.9752e-01, 9.8971e-01,
            -3.7192e-01, -9.9705e-01, -9.9620e-01, -1.0467e-01, -6.5039e-01,
            9.8558e-01, -4.8505e-01, 9.9999e-01, 8.7286e-01, 5.5062e-01,
            -7.0260e-01, 2.6718e-01, 6.3794e-01, 3.1736e-01, -3.6458e-01,
            9.9834e-01, -3.5935e-01, 9.7944e-01],
        Atom(b45, [Structure(s, [List([D2, R2, space2, B2, E2, R2, N2, A2, R2, D2, space2, R2, I2, E2, U2, X2]),
                                 List([D2, 'r', space2, R2, 'i', 'e', 'u', 'x'])])])
        : [-0.3844, 0.7598, 0.9974, -0.9927, 0.7489, -0.1418, 0.9849, 0.6287,
            -0.9794, -0.7103, 0.9606, 0.9924, 0.5133, -0.9944, -0.7021, -0.9751,
            0.9771, -0.2263, -0.9984, -0.3891, 0.0646, -0.9965, 0.6009, 0.5822,
            0.9661, 0.1581, 0.9835, 0.9962, 0.9240, 0.9132, 0.2590, -0.9891,
            0.5327, -0.9929, 0.3870, -0.4543, -0.7488, 0.0154, 0.4009, 0.3301,
            -0.9223, 0.9007, -0.7794, -0.5083, -0.8318, 0.4491, 0.6598, -0.0395,
            -0.3910, 0.9992, -0.9678, 1.0000, 0.7372, 0.9999, 0.9892, 0.8641,
            0.9768, 0.3740, 0.5993, 0.8339, 0.9405, -0.4901, 0.9329, -0.8018,
            -0.6308, -0.9086, 0.3425, 0.5298, -0.4592, 0.8339, 0.8287, 0.3205,
            0.9982, -0.9379, 0.0734, -0.9545, 0.7694, -0.9990, 0.9707, 0.9977,
            -0.5071, -0.9915, 0.9908, -0.3069, 0.6272, -0.9329, 0.8038, -0.9903,
            0.3306, -0.8902, -0.5071, -0.9807, 0.2676, 0.8839, 0.9976, 0.6617,
            -0.1947, 0.2696, 0.6349, -0.0336, -0.8031, -0.1912, -0.7460, 0.8568,
            -0.4920, -0.1950, -0.9512, -0.3912, -0.6186, -0.1107, 0.9937, -0.9675,
            -0.7079, -0.1341, 0.0148, 0.1026, 0.9815, 0.1427, -0.2687, 0.9957,
            -0.0327, -0.4350, 0.9922, -0.7567, 0.8065, -0.5099, 0.2703, -0.0604,
            0.2658, -0.9009, 0.6483, -0.9849, 0.8255, 0.9931, -0.2356, 0.9980,
            -0.9945, 0.2290, -0.9986, -0.3118, 0.7286, -0.2763, 0.2849, 0.9049,
            0.9909, 0.2226, 0.2157, 0.4241, -0.3875, 0.3645, 0.4791, 0.9551,
            -0.9683, 1.0000, -0.2601, 0.0895, 0.3070, 0.3075, 0.5659, -0.6502,
            0.9710, -0.9859, -0.5036, 0.7070, 0.9944, 0.9910, -0.2546, 0.5372,
            0.9932, 0.8140, 0.7697, -0.9064, -0.5785, 0.6913, 0.2384, 0.3730,
            0.9645, 0.9928, -0.9714, 0.9942, 1.0000, -0.5646, 0.9587, -0.6770,
            -0.9940, -0.9941, -0.9811, 0.4623, -0.7484, -0.2543, -0.5298, 0.9675,
            -0.8614, 0.8581, -0.9770, 0.1705, 0.9674, -0.4185, 0.9946, 0.8927,
            -0.9908, 0.6391, -0.8894, 0.9465, -0.2790, 0.9879, 0.3577, 0.4151,
            0.7382, -1.0000, -0.8455, -0.5245, -0.8458, 0.6979, 0.9907, 0.6597,
            -0.4347, -0.0224, 0.5317, 0.9977, -0.9844, -0.4839, 0.6922, -0.9915,
            -0.9945, 0.9702, -0.2328, 0.2343, -0.3780, -0.9616, 0.0270, 0.0329,
            0.9676, 0.3693, 0.8113, -0.9967, 0.9541, -0.9762, 0.4955, 0.0881,
            0.5198, -0.6812, -0.9784, 0.7620, 0.9314, -0.2298, 0.0710, -0.6153,
            0.9351, 0.8062, -0.7495, -0.0085, -0.9721, 0.9938, 0.8716, -0.6947,
            0.9913, -0.9708, 0.1537, 0.5724, -0.0281, -1.0000, -0.3072, -0.9004,
            0.8362, -0.1727, 0.9857, -0.9742, 0.0868, 0.1087, -0.9973, 0.8913,
            -0.3312, 0.9869, -0.9667, 0.8728, 0.9960, 0.7291, -0.9720, -0.9981,
            0.6040, 0.9999, -0.9709, -0.1735, 0.9986, 0.8854, -0.8941, -0.9720,
            -0.9975, -0.9903, -0.7353, 0.3124, -0.0172, 0.9909, 0.1367, 0.0134,
            0.9809, 1.0000, -0.3678, 0.0532, 0.2278, -0.7800, -1.0000, 0.4471,
            0.1035, -0.9982, 0.9946, -0.9678, 1.0000, -0.9589, 0.8390, 0.8653,
            -0.1692, 0.2483, 0.3037, 0.9974, 0.9854, -0.4317, 0.1337, -0.3415,
            0.4246, -0.5906, 0.8219, 0.6922, 0.0188, -0.7606, -0.6266, -0.3906,
            -0.9852, -0.5671, 0.4199, 0.9047, -0.2169, 0.9596, 0.9743, 0.0067,
            0.6321, -0.4188, -0.9997, 0.3084, 0.5923, -0.3583, 0.6116, -0.1893,
            0.9874, -0.9722, 0.9999, -0.1980, 0.5226, 0.7666, 0.9981, -0.9997,
            0.2348, -0.0718, 0.8528, 0.8292, 0.9826, -0.1734, 0.5221, 0.4843,
            0.0163, 0.8976, 0.9804, 0.6256, -0.1692, 0.9497, 0.8143, 0.9806,
            -0.8455, -0.1496, -0.8692, 0.7026, 0.9264, 0.1128, -0.5251, -0.2941,
            0.0731, -0.0132, -0.6764, 0.7811, 0.3649, 0.1690, -0.9914, 0.2461,
            0.8492, 0.9941, -0.9840, -0.2755, 0.9864, -0.2539, 0.2399, 0.7391,
            -0.5926, -0.9385, -0.2124, -0.9963, -0.3909, -0.7670, 0.8340, 0.5545,
            -0.0179, 0.4229, 0.0454, -0.6966, 0.8934, 0.4577, 0.9825, 0.3269,
            -0.6687, -0.3606, 0.7393, 0.3474, -0.3586, 0.9935, -0.9862, 0.9992,
            0.5014, -0.9957, 0.9167, 0.3748, -0.9982, -0.2084, -0.9995, 0.9816,
            -0.8925, 0.8466, 0.8401, -0.9998, -1.0000, 0.1839, -0.6170, -0.0631,
            -0.8790, 0.9991, -0.1016, 0.4901, -0.2246, -0.9219, 0.0440, 0.8897,
            -0.3960, -0.9962, 0.5394, -0.1135, 0.7941, 0.4836, -0.9519, 0.8833,
            -0.9324, 0.7418, 0.9703, 0.8622, 0.6055, -0.9911, 0.9877, 0.4411,
            0.0781, -0.7867, -0.9840, 0.9978, -0.7519, 0.2649, -0.3597, -0.9865,
            -0.4433, -0.0422, -0.7812, -0.9881, 0.3642, -0.9935, -0.9990, 0.5861,
            -0.8205, 0.9997, 0.9759, 0.5210, 0.0134, -0.9487, 0.1935, -0.9980,
            -0.0542, 0.0760, -0.9714, 0.8797, 0.8930, 0.9910, -0.1215, 0.2746,
            -0.8622, 0.6973, 0.9471, 0.8521, -0.6135, 0.1129, -0.2841, -0.9802,
            -0.9529, 0.9903, 0.7742, 0.9759, -0.8327, -0.4785, 0.1978, -0.4587,
            0.4982, -1.0000, -0.8447, -0.1459, -0.9976, 0.9969, -0.9982, -0.3875,
            0.5291, -0.1038, 0.9704, 0.7354, -0.9975, -0.9974, -0.9969, -0.3190,
            0.9838, -0.0331, 0.3930, 0.7526, 0.1691, 0.9895, 0.7271, 0.6544,
            0.8568, 0.9934, 0.0347, -0.9908, 0.1037, -0.9966, 0.3778, 0.9309,
            0.9703, 0.9739, 0.7101, 0.9982, -0.9993, 0.9998, -0.9963, 0.7564,
            0.9974, -0.9783, -0.0713, -0.9806, 0.4846, -0.7233, 0.4484, -0.7741,
            0.9905, -0.9909, -0.9899, 0.8866, 0.8719, 0.5758, -0.5113, 0.5472,
            0.9907, 0.2030, 0.9698, -0.5555, 0.1185, 1.0000, -0.0671, 0.3357,
            -0.9923, 0.9813, 0.5045, 0.3502, 0.9335, -0.2826, 0.5445, 0.7910,
            -0.9822, -0.5865, -0.9872, 0.4099, -0.8314, 0.4022, 0.3560, -0.0448,
            0.4206, -0.9835, 0.1743, -0.9932, 0.9878, 0.4682, 0.2869, 0.3760,
            0.9095, -0.8866, 0.9906, 0.9963, -1.0000, 0.1432, 0.9838, 0.6051,
            0.9946, -0.9630, -0.3922, 0.8818, -0.6255, 0.9595, 0.7818, -0.4306,
            0.9891, -0.9827, 0.6382, -0.7266, -0.3941, 0.8959, -0.9626, 0.2490,
            -0.4270, 0.3156, -0.9971, -0.8991, -0.9947, -0.2625, 0.9928, 0.8687,
            0.9972, 0.6575, -0.6324, 0.1730, -0.9948, 0.6796, 0.0988, -0.1987,
            0.4550, -0.8225, 0.3564, -0.8699, -0.9978, 0.1169, -0.5131, 0.0664,
            0.8464, 0.5472, -0.9766, -0.7735, -0.7378, 0.5023, -0.6156, -0.9571,
            0.9424, -0.4820, 0.9985, -0.9861, -0.9095, -0.9679, -0.5723, 0.2151,
            0.3284, 0.3640, 0.3299, -0.5595, -0.3997, 0.9636, -0.9819, -0.9811,
            0.9940, -0.7224, 0.2600, -0.2032, -0.5046, -0.4351, -0.3248, 0.6190,
            -0.8816, -0.6824, -0.9999, -0.7596, 0.2054, -0.9675, -0.9567, -0.5705,
            -1.0000, 0.9895, 0.9541, 0.9958, -0.9970, 0.9042, 0.2633, 0.9918,
            -0.4319, -0.5505, 0.1249, 0.9931, 0.8138, -0.9093, -0.0127, -0.2758,
            0.2548, -0.9072, -0.7672, 0.7482, 0.7510, -0.9602, -0.9977, 0.9971,
            -0.0765, 0.9885, 0.6439, 0.5971, -0.9633, 0.5753, 0.5471, -0.8815,
            -0.9983, 0.5609, -1.0000, -0.9948, 0.5486, 0.9779, -0.9899, -0.9597,
            0.2072, -0.9986, -0.5357, 0.6122, 0.4247, -0.9892, -0.7679, -0.4385,
            0.6610, 0.9731, -0.9873, 0.1171, -0.9427, 0.9975, -0.0202, 0.3340,
            -0.2203, 0.4057, 0.6489, -0.9754, -0.9151, -0.9923, -0.9344, 0.9847,
            0.9810, -0.9966, -0.9803, -0.5344, -0.7663, 0.9895, -0.3810, -0.9981,
            -0.9980, -0.0135, -0.7315, 0.9865, -0.5233, 1.0000, 0.9394, 0.6901,
            -0.5603, 0.1885, 0.6749, 0.4359, -0.3039, 0.9980, -0.1982, 0.9906],
        Atom(b45, [
            Structure(s, [List([M2, R2, space2, P2, A2, T2, R2, I2, C2, K2, space2, S2, T2, A2, R2, F2, I2, S2, H2]),
                          List([M2, 'r', space2, S2, 't', 'a', 'r', 'f', 'i', 's', 'h'])])])
        : [-0.3852, 0.7345, 0.9975, -0.9941, 0.8059, 0.1372, 0.9769, 0.3380,
            -0.9710, -0.7639, 0.9537, 0.9944, -0.6378, -0.9972, 0.1381, -0.9642,
            0.9826, -0.6583, -0.9990, -0.8471, -0.0242, -0.9978, 0.5266, 0.9145,
            0.9733, -0.0463, 0.9903, 0.9978, 0.9608, 0.8392, 0.3897, -0.9939,
            0.7493, -0.9895, 0.1275, -0.6422, 0.1913, -0.1657, 0.4672, -0.4753,
            -0.8613, 0.5287, -0.3918, -0.5905, -0.5743, 0.5197, 0.4387, 0.0882,
            -0.2982, 0.9991, -0.9532, 1.0000, -0.4085, 0.9998, 0.9948, 0.8237,
            0.9902, 0.2680, -0.6296, 0.7766, 0.9731, -0.3563, 0.9560, -0.6181,
            -0.5123, -0.7991, -0.2703, 0.3500, -0.6042, 0.8757, 0.7169, 0.4822,
            0.9962, -0.8862, -0.0171, -0.9352, 0.5910, -0.9992, 0.9729, 0.9981,
            0.0999, -0.9965, 0.9961, -0.4205, -0.0909, -0.5181, -0.2293, -0.9933,
            0.4338, -0.8908, -0.3783, -0.9848, 0.2543, 0.5419, 0.9990, 0.5529,
            -0.3872, 0.4308, 0.6484, -0.0919, -0.7589, 0.2609, 0.1259, 0.3663,
            0.6789, -0.4556, -0.9450, -0.7046, -0.4196, 0.0428, 0.9966, -0.9836,
            -0.6301, 0.2507, 0.4674, -0.4736, 0.9889, 0.9493, -0.1827, 0.9984,
            -0.4153, 0.4323, 0.9962, 0.1001, 0.5383, -0.2455, -0.1985, 0.4913,
            0.1638, -0.8653, 0.7844, -0.9897, 0.0204, 0.9955, -0.3433, 0.9991,
            -0.9983, 0.7345, -0.9997, -0.5251, 0.8592, -0.1724, -0.5439, 0.8175,
            0.9949, 0.3356, -0.2149, 0.2726, -0.5845, -0.5103, 0.1170, 0.9395,
            -0.9835, 0.9999, 0.6233, 0.6268, 0.5288, 0.1497, -0.4299, -0.3925,
            0.9518, -0.9908, -0.3655, 0.0607, 0.9964, 0.9784, 0.7254, -0.4924,
            0.9981, 0.6327, 0.1694, -0.7728, -0.6771, -0.2628, 0.2479, 0.4258,
            0.9799, 0.9955, -0.9928, 0.9917, 0.9983, -0.5409, 0.9199, 0.5329,
            -0.9931, -0.9937, -0.9849, 0.2594, -0.7298, 0.3730, -0.3142, 0.9751,
            -0.0240, 0.8544, -0.9889, -0.1050, 0.9589, -0.2545, 0.9985, 0.6252,
            -0.9975, -0.3108, -0.0131, 0.9476, -0.6496, 0.9274, -0.2822, 0.0052,
            0.8804, -1.0000, -0.0156, -0.6480, -0.5767, 0.8310, 0.9932, -0.4859,
            -0.4324, -0.1334, 0.3659, 0.9986, -0.9929, -0.5160, 0.5494, -0.9919,
            -0.9918, 0.9777, -0.3362, -0.7890, -0.6091, -0.9072, 0.1227, 0.4054,
            0.9800, -0.2131, 0.1595, -0.9982, 0.4893, -0.9794, -0.6006, 0.3867,
            0.4503, -0.5934, -0.9497, -0.1525, 0.9319, 0.4307, -0.3206, -0.5420,
            0.6359, 0.2368, -0.2453, -0.5570, -0.9791, 0.9957, 0.4796, 0.2508,
            0.9937, -0.9883, 0.5527, -0.3445, -0.0376, -1.0000, 0.0140, -0.5303,
            0.3818, -0.2331, 0.9916, -0.9456, 0.0088, 0.5758, -0.9971, 0.9247,
            -0.2938, 0.9947, -0.8102, 0.9015, 0.9967, 0.7359, -0.9878, -0.9986,
            0.7122, 0.9992, -0.9892, -0.1406, 0.9988, -0.1096, -0.9361, -0.9530,
            -0.9969, -0.9963, -0.3721, -0.6823, 0.3140, 0.9852, 0.2963, -0.2001,
            0.9703, 0.9999, 0.0507, 0.2957, 0.0079, -0.9191, -0.9996, 0.7627,
            0.1882, -0.9990, 0.9969, -0.9881, 1.0000, -0.5503, 0.0145, 0.7404,
            -0.1242, -0.3625, 0.1737, 0.9985, 0.9891, -0.5261, 0.0040, 0.0058,
            -0.0701, 0.1499, 0.7313, 0.0668, 0.1995, -0.7474, 0.5998, -0.2848,
            -0.9901, 0.6951, 0.3880, 0.8765, -0.6177, 0.9441, 0.9809, 0.2549,
            0.8276, -0.0631, -0.9907, -0.1131, 0.4568, -0.8409, 0.3996, 0.9514,
            0.9775, -0.9392, 0.9998, -0.4386, 0.5328, -0.1037, 0.9985, -0.9995,
            0.1791, 0.3184, 0.2907, 0.6855, 0.9960, 0.6461, 0.8343, 0.6132,
            0.0707, 0.8761, 0.9812, -0.4295, -0.2146, 0.3613, 0.4318, 0.9925,
            0.2477, -0.2118, -0.4657, -0.4815, 0.9771, -0.2570, -0.0540, -0.1607,
            -0.4816, 0.6151, 0.2879, -0.0758, 0.5907, 0.1042, -0.9895, -0.1400,
            0.9141, 0.9974, -0.9827, -0.2399, 0.9869, -0.2773, -0.3477, 0.7438,
            0.6536, -0.9655, -0.3127, -0.9979, -0.1210, -0.5882, 0.6174, 0.1984,
            0.3459, -0.3406, 0.6218, -0.4702, 0.8656, -0.2468, 0.9864, 0.3165,
            0.0039, -0.3949, 0.3852, 0.5815, -0.4438, 0.9912, -0.9824, 0.9989,
            0.4603, -0.9972, 0.0372, 0.3459, -0.9981, 0.6860, -0.9993, 0.9717,
            -0.5288, -0.3365, -0.2376, -0.9997, -1.0000, 0.2861, 0.3779, -0.3145,
            -0.8685, 0.9840, 0.0616, 0.2655, -0.2133, -0.9380, 0.3112, 0.1968,
            0.3359, -0.9982, -0.3040, 0.8059, -0.2713, -0.8022, -0.9462, 0.9563,
            -0.8568, 0.4049, 0.9788, 0.2497, 0.6084, -0.9937, 0.9873, -0.4304,
            0.1925, -0.9253, -0.9732, 0.9984, -0.1558, 0.2254, -0.2934, -0.9850,
            0.5940, -0.6573, -0.8620, -0.9816, 0.5400, -0.9929, -0.9989, 0.2561,
            0.3204, 0.9996, 0.9692, 0.5016, -0.4319, -0.9447, 0.4284, -0.9988,
            -0.1240, 0.5304, -0.9910, 0.3945, 0.9671, 0.9937, -0.7938, -0.8671,
            -0.7942, 0.7279, 0.9523, 0.6246, -0.4579, 0.2611, -0.2162, -0.9836,
            -0.9132, 0.9972, -0.2350, 0.9942, 0.1800, 0.4150, 0.2842, -0.1855,
            -0.6758, -0.9999, -0.8770, -0.2012, -0.9985, 0.9986, -0.9986, -0.3612,
            0.7681, 0.4189, 0.9611, 0.7079, -0.9986, -0.9989, -0.7225, 0.1241,
            0.9861, -0.0723, 0.2671, 0.4228, 0.1654, 0.9974, -0.5130, 0.6585,
            -0.1069, 0.9983, 0.3707, -0.9956, 0.8987, -0.9961, 0.7641, 0.9654,
            0.9755, 0.9806, -0.3402, 0.9990, -0.9994, 0.9999, -0.9984, -0.1470,
            0.9976, -0.9844, 0.4191, -0.9958, -0.7167, 0.0522, 0.3565, -0.7588,
            0.9941, -0.9938, -0.9940, 0.6411, 0.5685, 0.3377, 0.5661, -0.1372,
            0.9925, 0.0089, 0.9642, -0.3252, 0.8777, 1.0000, -0.5970, 0.0450,
            -0.9934, 0.9914, 0.0745, 0.3735, 0.9483, -0.2509, 0.6404, 0.7305,
            -0.9918, -0.4714, -0.9594, 0.3555, -0.3206, 0.5273, 0.0886, -0.2442,
            0.1888, -0.9924, 0.4264, -0.9970, 0.9950, -0.2296, 0.4891, -0.3531,
            0.8298, -0.9362, 0.9898, 0.9954, -1.0000, 0.2641, 0.9756, -0.0347,
            0.9815, -0.9704, -0.2848, 0.8885, -0.7025, 0.9771, 0.4239, -0.4619,
            0.9674, -0.9800, -0.2157, -0.7636, -0.1618, 0.7059, -0.9845, 0.3201,
            0.3618, 0.2071, -0.9976, -0.5275, -0.9964, -0.2254, 0.9904, -0.4919,
            0.9987, 0.3791, -0.5217, 0.0764, -0.9955, -0.5732, 0.0985, -0.0564,
            -0.7296, 0.2732, 0.1567, 0.1312, -0.9970, 0.4564, 0.6875, 0.1908,
            0.8108, 0.5675, -0.9917, -0.9397, -0.8316, 0.3401, -0.3697, -0.8310,
            0.8345, -0.2541, 0.9989, -0.9803, -0.9423, -0.9830, -0.3402, 0.2689,
            0.4802, -0.0058, -0.1093, -0.0909, -0.8000, 0.9834, -0.9838, -0.9821,
            0.9985, -0.2094, -0.6600, -0.2724, -0.5412, -0.5338, -0.3967, 0.4447,
            -0.7538, -0.4502, -0.9999, 0.2972, -0.1653, -0.9902, -0.9012, -0.4459,
            -1.0000, 0.9928, 0.9506, 0.9982, -0.9966, 0.9370, 0.5172, 0.9967,
            -0.4265, -0.7110, 0.4767, 0.9941, 0.2522, -0.3958, -0.0453, -0.2692,
            0.1593, -0.8081, 0.4141, -0.0015, 0.2456, -0.9670, -0.9965, 0.9980,
            -0.2458, 0.9768, 0.6371, 0.6954, -0.9648, 0.7882, -0.0871, -0.9062,
            -0.9990, 0.2383, -1.0000, -0.9946, 0.1865, 0.9808, -0.9971, -0.9833,
            -0.0320, -0.9981, 0.5306, -0.6038, 0.3752, -0.9869, 0.3325, -0.5508,
            -0.3786, 0.9820, -0.9682, 0.6421, -0.4328, 0.9552, 0.0431, 0.3549,
            -0.7455, -0.4980, 0.4953, -0.9796, 0.3671, -0.9885, -0.8749, 0.9914,
            0.9846, -0.9985, -0.9867, 0.5563, -0.4794, 0.9963, -0.6235, -0.9966,
            -0.9987, 0.1526, -0.5981, 0.9942, -0.4524, 0.9999, 0.9179, 0.0297,
            -0.0145, 0.0737, 0.5049, 0.0341, -0.1828, 0.9989, -0.7702, 0.9887]
    }
    primitives3 = [
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
        lambda x: plain_extension(x, copyskip1, connected_clauses=True)]
    print(encodingExamples.get(Atom(b45, [Structure(s, [List([D2, R2, space2, R2, A2, Y2, space2, S2, T2, A2, N2, T2, Z2]),
                                 List([D2, 'r', space2, S2, 't', 'a', 'n', 't', 'z'])])])))
    encoding2 = encoding2.encoding(primitives2)

    primitives = [not_space, mk_uppercase, mk_lowercase, is_empty, is_space, is_uppercase, not_uppercase, is_lowercase,
                  not_lowercase, is_letter, not_letter, is_number, not_number, skip1, copy1, write1, copyskip1]
    learner = SimpleBreadthFirstLearner(prolog,encoding2)
    #primitives = [None, mk_uppercase, None,None, None, None, None, None,
    #              None, None,None,None,None,skip1, copy1, None, copyskip1]
    decoder = decoding.decoding(primitives)
    #primitives = [mk_uppercase, copy1, copyskip1]
    #network1 = Network.Network(3, [2619, 100, 17])
    #output = learner.readEncodingsOfFile('/home/marlon/loreleai/Neural2/[1, 1, 1, 1, 1]output500.txt')
    #input = learner.readEncodingsOfFile('/home/marlon/loreleai/Neural2/[1, 1, 1, 1, 1]input500.txt')
    #print('train')
    #tic = time.perf_counter()
    #network1.Train(input,output,1)
    network1 = Network.Network.LoadNetwork('/home/marlon/loreleai/Neural2/network18.txt')
    #toc = time.perf_counter()
    #print(f"Downloaded the tutorial 2 in {toc - tic:0.10f} seconds")
    print(network1.GetNumOutputs())
    network2 = Network.Network.LoadNetwork('/home/marlon/loreleai/Neural2/network19.txt')
    print(network1.GetNumOutputs())
    covered = [Atom(b45, [Structure(s, [List(
        [D2,R2,space2,M2,O2,N2,T2,G2,O2,M2,E2,R2,Y2,space2,M2,O2,N2,T2,G2,O2,M2,E2,R2,Y2]),
        List(
            [D2,'r',space2,M2,'o','n','t','g','o','m','e','r','y'])])]),
       Atom(b45, [Structure(s, [List(
           [P2,R2,O2,F2,E2,S2,S2,O2,R2,space2,S2,E2,V2,E2,R2,U2,S2,space2,S2,N2,A2,P2,E2]),
           List(
               [P2,'r','o','f','e','s','s','o','r',space2,S2,'n','a','p','e'])])]),
       Atom(b45, [Structure(s, [List(
           [M2,R2,space2,S2,P2,O2,N2,G2,E2,B2,O2,B2,space2,S2,Q2,U2,A2,R2,E2,P2,A2,N2,T2,S2]),
           List(
               [M2,'r',space2,S2,'q','u','a','r','e','p','a','n','t','s'])])]),
       Atom(b45, [Structure(s, [List(
           [M2, R2, space2, H2,A2,R2,R2,Y2,space2,P2,O2,T2,T2,E2,R2]),
           List(
               [M2, 'r', space2, P2,'o','t','t','e','r'])])]),
       Atom(b45, [Structure(s, [List(
           [M2, S2, space2, D2,A2,E2,N2,E2,R2,Y2,S2,space2,T2,A2,R2,G2,A2,R2,Y2,E2,N2]),
           List(
               [M2, 's',space2,T2,'a','r','g','a','r','y','e','n'])])])

               ]

    learner.assert_knowledge(background)
    head = Atom(b45, [A])
    body1 = Atom(copyskip1, [A,B])
    body2 = Atom(skip1, [B,D])
    body3 = Atom(skip1, [C, D])
    body4 = Atom(mk_lowercase, [C,F])
    clause1 = Clause(head, Body(body1, body2, body3, body4))

    print('coverage clause 1 : ', learner.evaluateClause(covered, clause1))
    body1 = Atom(copyskip1, [A,B])
    body2 = Atom(skip1, [B,D])
    body3 = Atom(skip1, [C, D])
    body4 = Atom(mk_lowercase, [C,F])
    body5 = Atom(skip1, [C, G])
    body6 = Atom(not_uppercase, [G])
    body7 = Atom(skip1, [C,H])
    body8 = Atom(mk_lowercase, [C,I])
    body9 = Atom(skip1, [E,F])
    body10 = Atom(mk_lowercase, [E,K])
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


