import typing
import csv
from abc import ABC, abstractmethod

from orderedset import OrderedSet

from pylo.engines.prolog.SWIProlog import Pair

from docs.mytuple import mytuple
from loreleai.language.lp import c_var, c_pred, Clause, Procedure, Atom, Body, Not, List, c_const, c_functor, c_literal, \
    Structure
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_singleton_vars, has_duplicated_literal, has_duplicated_variable, \
    connected_clause, connected_body, max_var_occurrences
from loreleai.learning.language_manipulation import plain_extension, _plain_extend_clause
from loreleai.learning.task import Task, Knowledge
from loreleai.reasoning.lp.prolog import SWIProlog, Prolog
import heapq as has
from MyModel import encoding2,decoding,Network
import numpy as np
import random
import time


class MyOwnLearner:

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

    def evaluateClause(self, examples, clause: Clause):
        numberofpositivecoverance = 0
        self._solver.assertz(clause)
        for example in examples:
            if self._solver.has_solution(example):
                print(example)
                numberofpositivecoverance += 1
        self._solver.retract(clause)
        return numberofpositivecoverance

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
                len(self._candidate_pool) > 0) and self._expansionOneClause < 10000:
            # get first candidate from the pool
            current_cand = self.get_from_pool()
            print('current : ', current_cand._clause )
            print('value : ' ,1-current_cand._value)#because of heapq (ordered from minimum to maximum)
            print('expansions : ' , self._expansion)
            current_cand = current_cand._clause
            expansions = decoder.decode(self._neural1.FeedForward([*self._encoder.encode2(current_cand),*state]))
            bestk = sorted(expansions, reverse=True)[5:0:-1]
            #print(bestk)
            for exp in bestk:
                exps = plain_extension(current_cand.get_body(),exp._clause)
                bestkClause = []
                for exp2 in exps:
                    if not self.badClause(current_cand.get_head,exp2):
                        y = self._neural2.FeedForward([*self._encoder.encode2(Clause(current_cand.get_head(),exp2)), *state])
                        bestkClause.append(mytuple(y[0],y[0],exp._value,Clause(current_cand.get_head(),exp2)))
                if len(bestkClause)>10:
                    bestkClause = sorted(bestkClause, reverse=True)[10:0:-1]
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

    def _execute_program(self, clause: Clause) -> typing.Sequence[Atom]:
        """
        Evaluates a clause using the Prolog engine and background knowledge
        Returns a set of atoms that the clause covers
        """
        if self._numberOfEvaluations < 90000:
            self._numberOfEvaluations += 1
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

        self.assert_knowledge(knowledge)

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
            with open('Search15.csv', 'w') as f:
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

if __name__ == '__main__':
    print("Search for b99")
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
    b291 = c_pred("b291", 1)
    b99 = c_pred("b99", 1)
    b94 = c_pred("b94", 1)
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
    zeroz = c_const("\"0\"")
    onez = c_const("\"1\"")
    twoz= c_const("\"2\"")
    threez = c_const("\"3\"")
    fourz = c_const("\"4\"")
    fivez = c_const("\"5\"")
    sexz = c_const("\"6\"")
    sevenz = c_const("\"7\"")
    eightz = c_const("\"8\"")
    ninez = c_const("\"9\"")
    tinez = c_const("\"10\"")
    hook1z = c_const("\"(\"")
    hook2z = c_const("\")\"")
    puntz = c_const("\".\"")
    kommaz= c_const("\",\"")
    streepz = c_const("\"-\"")
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
    one2 = c_const("\"\"1\"\"")
    two2 = c_const("\"\"2\"\"")
    three2 = c_const("\"\"3\"\"")
    four2 = c_const("\"\"4\"\"")
    five2 = c_const("\"\"5\"\"")
    sex2 = c_const("\"\"6\"\"")
    seven2 = c_const("\"\"7\"\"")
    eight2 = c_const("\"\"7\"\"")
    nine2 = c_const("\"\"9\"\"")
    zero2 = c_const("\"\"0\"\"")
    hook1a= c_const("\"\"(\"\"")
    hook2a = c_const("\"\")\"\"")
    punt2 = c_const("\"\".\"\"")
    komma2 = c_const("\"\",\"\"")
    streep2 = c_const("\"\"-\"\"")
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
    zero = c_const("\"0\"")
    one = c_const("\"1\"")
    two = c_const("\"2\"")
    three = c_const("\"3\"")
    four = c_const("\"4\"")
    five = c_const("\"5\"")
    six = c_const("\"6\"")
    seven = c_const("\"7\"")
    eight = c_const("\"8\"")
    nine = c_const("\"9\"")
    space = c_const("' '")
    space2 = c_const("\"' '\"")
    b45 = c_pred('b45', 1)
    b291 = c_pred("b291", 1)

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
        is_number_aux(zero),
        is_number_aux(one),
        is_number_aux(two),
        is_number_aux(three),
        is_number_aux(four),
        is_number_aux(five),
        is_number_aux(six),
        is_number_aux(seven),
        is_number_aux(eight),
        is_number_aux(nine),
    )

    # positive examples
    pos = {
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, one, zero, hook2a]),
                          List([one, zero])])]),
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, one, hook2a]),
                          List([one])])]),

        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, three, hook2a]),
                          List([three])])]),
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, five, zero, hook2a]),
                          List([five, zero])])]),
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, five, hook2a]),
                          List([five])])])}

    # negative examples
    neg = {Atom(b45, [
        Structure(s, [List(['l', 'e', 't', 't', 'e', 'r', hook1a, 'd', hook2a, punt2]),
                      List(['d'])])]),
            Atom(b45, [Structure(s, [List(
                [three, two, punt2, six, space2, hook1z, nine, zero, punt2, seven, hook2a]),
                List([nine, zero, punt2, seven])])]),
            Atom(b45, [Structure(s, [List(
                [four, space2, S2, 'c', 'i', 'e', 'n', 'c', 'e', space2, nine, komma2, three, six, nine, space2, one,
                 komma2, one, two, five, komma2, zero, two, two]),
                                     List([four, space2, S2, 'c', 'i', 'e', 'n', 'c', 'e'])])]),
            Atom(b45, [
                Structure(s, [List(
                    [eight, space2, J2, 'o', 'h', 'n', space2, M2, 'a', 'j', 'o', 'r', space2, one, nine, nine, one,
                     streep2, one, nine, nine, seven, space2, C2, 'o', 'n', 's', 'e', 'r', 'v', 'a', 't', 'i', 'v',
                     'e']),
                              List([J2, 'o', 'h', 'n', space2, M2, 'a', 'j', 'o', 'r'])])]),
            Atom(b45, [Structure(s, [List(
                [three, space2, four, nine, six, space2, 'k', 'm', space2, hook1a, two, komma2, one, seven, two, space2,
                 'm', 'i', hook2z, space2, eight, seven, 'h', space2, three, four, space2, four, seven]),
                                     List([three, komma2, four, nine, six, komma2, space2, two, komma2, one, seven,
                                           two])])])}
    task = Task(positive_examples=pos, negative_examples=neg)

    examples = {
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1z, onez, zeroz, hook2z]),
                          List([onez, zeroz])])]),
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1z, onez, hook2z]),
                          List([onez])])]),

        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1z, threez, hook2z]),
                          List([threez])])]),
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1z, fivez, zeroz, hook2z]),
                          List([fivez, zeroz])])]),
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1z, fivez, hook2z]),
                          List([fivez])])]),
        Atom(b45, [
            Structure(s, [List(['l', 'e', 't', 't', 'e', 'r', hook1z, 'd', hook2z, puntz]),
                          List(['d'])])]),
        Atom(b45, [Structure(s, [List(
            [threez, twoz, puntz, sexz, space, hook1z, ninez, zeroz, puntz, sevenz, hook2a]),
            List([ninez, zeroz, puntz, sevenz])])]),
        Atom(b45, [Structure(s, [List(
            [fourz, space, Sz, 'c', 'i', 'e', 'n', 'c', 'e', space, ninez, kommaz, threez, sexz, ninez, space, onez,
             kommaz, onez, twoz, fivez, kommaz, zeroz, twoz, twoz]),
            List([fourz, space, Sz, 'c', 'i', 'e', 'n', 'c', 'e'])])]),
        Atom(b45, [
            Structure(s, [List(
                [eightz, space, Jz, 'o', 'h', 'n', space, Mz, 'a', 'j', 'o', 'r', space, onez, ninez, ninez, onez,
                 streepz, onez, ninez, ninez, sevenz, space, Cz, 'o', 'n', 's', 'e', 'r', 'v', 'a', 't', 'i', 'v',
                 'e']),
                List([Jz, 'o', 'h', 'n', space, Mz, 'a', 'j', 'o', 'r'])])]),
        Atom(b45, [Structure(s, [List(
            [threez, space, fourz, ninez, sexz, space, 'k', 'm', space, hook1z, twoz, kommaz, onez, sevenz, twoz,
             space, 'm', 'i', hook2z, space, eightz, sevenz, 'h', space, threez, fourz, space, fourz, sevenz]),
            List([threez, kommaz, fourz, ninez, sexz, kommaz, space, twoz, kommaz, onez, sevenz,
                  twoz])])])

    }

    # create Prolog instance
    prolog = SWIProlog()

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
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, one, zero, hook2a]),
                          List([one, zero])])])
        : [-0.4290,  0.5826,  0.9949, -0.9743,  0.7096, -0.4042,  0.9145,  0.1713,
         -0.8593, -0.8956,  0.9048,  0.9880, -0.4734, -0.9929, -0.5674, -0.9426,
          0.9410, -0.4888, -0.9978,  0.5214, -0.2408, -0.9981,  0.5766,  0.8594,
          0.8975,  0.1904,  0.9652,  0.9985,  0.8058,  0.9587,  0.2213, -0.9664,
          0.4913, -0.9860, -0.0549, -0.6582, -0.1837, -0.2232,  0.0046, -0.5133,
         -0.9001,  0.5968, -0.6766, -0.6660,  0.3984,  0.2971,  0.5707, -0.0627,
         -0.0387,  0.9981, -0.8811,  1.0000, -0.5152,  0.9979,  0.9864,  0.7909,
          0.9786,  0.1799, -0.6591,  0.8287,  0.9642, -0.0719,  0.8055, -0.4996,
         -0.7769, -0.4128,  0.2894,  0.0525, -0.2845,  0.7376,  0.3891,  0.2531,
          0.9796, -0.6752,  0.1911, -0.8243,  0.5392, -0.9978,  0.8816,  0.9967,
         -0.8348, -0.9953,  0.9854, -0.4671,  0.2883, -0.8875, -0.8777, -0.9926,
          0.3607, -0.7041, -0.4780, -0.9353,  0.4165,  0.5279,  0.9982,  0.5413,
         -0.1903,  0.4288,  0.6014, -0.2928, -0.8192,  0.7249,  0.5629, -0.2875,
          0.7816, -0.4057, -0.9328, -0.3474, -0.9042, -0.0642,  0.9841, -0.9674,
         -0.8110,  0.3085,  0.8165, -0.5086,  0.9012,  0.2404, -0.0704,  0.9985,
         -0.1770,  0.8330,  0.9821,  0.1617,  0.4055, -0.4715,  0.2975,  0.3069,
          0.7086, -0.4856,  0.6143, -0.9873, -0.6819,  0.9937, -0.1451,  0.9986,
         -0.9934,  0.5610, -0.9991, -0.0023,  0.7679, -0.2291, -0.7024,  0.8038,
          0.9570,  0.2529, -0.4588,  0.5119, -0.5603,  0.4324,  0.6906,  0.6666,
         -0.9644,  0.9999,  0.4537,  0.6630,  0.8605, -0.0177, -0.6269, -0.1775,
          0.9481, -0.9865,  0.4802, -0.4580,  0.9959,  0.9200, -0.0327, -0.3679,
          0.9957,  0.7841,  0.3862, -0.5755, -0.7882, -0.6007,  0.3207,  0.2779,
          0.7732,  0.9964, -0.9779,  0.9919,  0.9994, -0.6600,  0.9049,  0.4165,
         -0.9804, -0.9078, -0.9688,  0.7860, -0.8627, -0.4092, -0.4850,  0.9088,
          0.3845,  0.7584, -0.9880,  0.0928,  0.9403, -0.3639,  0.9979,  0.1384,
         -0.9950, -0.1782, -0.0345,  0.8757, -0.3857,  0.8672,  0.0997,  0.4994,
          0.9565, -1.0000,  0.5282,  0.1587, -0.7165,  0.7897,  0.9899, -0.3987,
         -0.2220, -0.1036,  0.3454,  0.9977, -0.9951, -0.1655,  0.2228, -0.9803,
         -0.9906,  0.9794, -0.3334,  0.6466, -0.3461, -0.9618, -0.1402, -0.1964,
          0.9747,  0.3893,  0.9136, -0.9961, -0.1264, -0.8819, -0.4372,  0.0866,
          0.6003, -0.3450, -0.6239, -0.5924,  0.8286, -0.5123,  0.3934, -0.6088,
          0.9197, -0.6352, -0.6408,  0.3574, -0.9871,  0.9912,  0.5664,  0.6081,
          0.9744, -0.9827,  0.5893, -0.8055, -0.1210, -0.9999, -0.6267, -0.8953,
          0.7322, -0.3364,  0.9781, -0.9047, -0.7668, -0.0374, -0.9973,  0.8367,
         -0.4589,  0.9828, -0.6585,  0.7171,  0.9841,  0.6317, -0.9789, -0.9978,
          0.3688,  0.9997, -0.9653, -0.3939,  0.9977, -0.3471, -0.8156, -0.9444,
         -0.9954, -0.9888, -0.5021, -0.0153, -0.2905,  0.9553, -0.4425, -0.1228,
          0.9800,  0.9996, -0.4148,  0.2340,  0.1314, -0.8799, -0.9999, -0.4164,
         -0.3188, -0.9987,  0.9961, -0.9880,  0.9999,  0.2022, -0.4462,  0.8123,
         -0.2613, -0.2323,  0.0098,  0.9983,  0.8289, -0.2892,  0.2180,  0.1411,
          0.3193, -0.5865,  0.8622,  0.3707,  0.2725, -0.8425,  0.5218, -0.0701,
         -0.9897,  0.3379,  0.1474,  0.8044,  0.3513,  0.6639,  0.9546, -0.2792,
          0.7706, -0.3437, -0.9971, -0.2458,  0.2050, -0.4782,  0.6467,  0.9035,
          0.9856, -0.8418,  0.9995, -0.2744,  0.5834, -0.7705,  0.9985, -0.9982,
          0.2477, -0.4633,  0.2264,  0.7053,  0.9678,  0.3824,  0.8152, -0.1687,
         -0.3000,  0.7819,  0.9540, -0.1670, -0.3554, -0.4845,  0.7285,  0.9624,
          0.1137, -0.1142, -0.6700, -0.4473,  0.7722, -0.4009,  0.7703, -0.0986,
          0.0922,  0.3470,  0.7198,  0.4043,  0.2090,  0.3339, -0.9745, -0.5189,
          0.7297,  0.9947, -0.9291, -0.5597,  0.9665, -0.5180, -0.1387,  0.6092,
          0.6045, -0.9119, -0.0820, -0.9934, -0.1967, -0.6486,  0.5717,  0.8889,
          0.0810,  0.0627,  0.1580, -0.7638,  0.8813,  0.0137,  0.9349,  0.9023,
         -0.2268, -0.3342,  0.3202,  0.3394, -0.2678,  0.9843, -0.9568,  0.9982,
         -0.2039, -0.9970, -0.4601,  0.6247, -0.9989, -0.4052, -0.9982,  0.9595,
         -0.0910, -0.4545, -0.5656, -0.9993, -0.9998, -0.4814, -0.2711,  0.0536,
         -0.3622,  0.9836, -0.0176,  0.5277, -0.1189, -0.9417,  0.2947, -0.5532,
          0.2826, -0.9983, -0.2444,  0.8923, -0.2125, -0.5807, -0.9137,  0.8897,
         -0.8925,  0.4392,  0.9418,  0.6157,  0.8763, -0.9913,  0.9751, -0.5581,
          0.4414, -0.8734, -0.9666,  0.9975, -0.7707,  0.1718, -0.3373, -0.9930,
          0.6088, -0.6357, -0.4649, -0.9679,  0.4482, -0.8983, -0.9985,  0.1466,
          0.6069,  0.9974,  0.9533,  0.7333, -0.3858, -0.8320,  0.0225, -0.9987,
          0.0721,  0.2395, -0.9654,  0.8238,  0.9838,  0.9796, -0.7641, -0.7607,
         -0.2181,  0.5655,  0.7167,  0.4556, -0.7529,  0.1236, -0.1441, -0.9731,
         -0.6800,  0.9873, -0.7875,  0.9457,  0.7804,  0.7114,  0.5791, -0.1683,
         -0.4783, -0.9996, -0.9053, -0.3861, -0.9974,  0.9968, -0.9980, -0.4271,
          0.6942,  0.0733,  0.9707,  0.2714, -0.9984, -0.9978, -0.8501, -0.3570,
          0.9762,  0.1209,  0.1978,  0.6990,  0.5741,  0.9944, -0.6839,  0.5671,
         -0.4194,  0.9927,  0.6079, -0.9869,  0.7377, -0.9969,  0.8970,  0.9196,
          0.9344,  0.8965, -0.7894,  0.9974, -0.9975,  0.9990, -0.9982, -0.5880,
          0.9982, -0.9670,  0.6926, -0.9944, -0.4931, -0.7706,  0.1774, -0.6642,
          0.9863, -0.9875, -0.9925,  0.3096,  0.1863,  0.4019,  0.8125,  0.4131,
          0.9929,  0.3486,  0.8698, -0.4335,  0.7491,  0.9999, -0.3987,  0.2674,
         -0.9773,  0.9155,  0.5508,  0.3614,  0.8164, -0.3843,  0.5149,  0.7491,
         -0.9912, -0.1101, -0.9716,  0.4915, -0.6656,  0.5653,  0.1278,  0.5024,
          0.0479, -0.9863,  0.3791, -0.9932,  0.9011, -0.4177,  0.3012, -0.2266,
          0.7977, -0.7168,  0.9919,  0.9951, -1.0000,  0.2941,  0.9707,  0.8688,
          0.9439, -0.9674, -0.4354,  0.8791, -0.6003,  0.9482,  0.7643, -0.0540,
          0.9081, -0.9840,  0.0456, -0.7063, -0.1136,  0.8312, -0.9462,  0.2039,
          0.5499,  0.2941, -0.9964, -0.5639, -0.9958, -0.2037,  0.9860,  0.8860,
          0.9975,  0.7895, -0.4812, -0.0140, -0.9938, -0.7486,  0.4155, -0.3034,
          0.4422,  0.6012,  0.4952, -0.6278, -0.9985,  0.0866,  0.7185, -0.0169,
          0.9016,  0.7420, -0.8242, -0.7372, -0.8199,  0.2983, -0.2135, -0.7495,
          0.5452, -0.4929,  0.9975, -0.9912, -0.7318, -0.8276, -0.8360,  0.2803,
          0.2441, -0.0555,  0.4131,  0.3951, -0.6477,  0.9723, -0.9837, -0.9885,
          0.9929, -0.5351, -0.5415, -0.3514, -0.5280, -0.3334, -0.0413,  0.6337,
         -0.8589, -0.1506, -0.9992, -0.7288,  0.0693, -0.9654, -0.8164, -0.4800,
         -1.0000,  0.9880,  0.8060,  0.9970, -0.9972,  0.7975,  0.4289,  0.9942,
          0.0092, -0.6671,  0.2741,  0.9971,  0.4241, -0.2259, -0.1309, -0.2640,
          0.1185, -0.8836,  0.4255, -0.0844,  0.8892, -0.9775, -0.9981,  0.9979,
         -0.0605,  0.9662,  0.3311,  0.7785, -0.8548,  0.8728, -0.3019, -0.7669,
         -0.9981,  0.7786, -1.0000, -0.9780,  0.2986,  0.9828, -0.9915, -0.9202,
          0.0579, -0.9979,  0.2323, -0.7690,  0.3650, -0.9696,  0.6022, -0.7063,
          0.8593,  0.9662, -0.9158,  0.4641, -0.0557,  0.9885,  0.0109,  0.5648,
          0.1455, -0.7009, -0.1106, -0.7822,  0.1686, -0.9814, -0.7772,  0.9668,
          0.9474, -0.9963, -0.9813,  0.6957, -0.4042,  0.9755, -0.8069, -0.9963,
         -0.9984, -0.1712, -0.7027,  0.9892, -0.1497,  0.9999,  0.7520,  0.2078,
         -0.7987,  0.3703,  0.1979,  0.7477, -0.3540,  0.9992,  0.6691,  0.9786],
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, one, hook2a]),
                          List([one])])])
        : [-2.1451e-01,  5.6787e-01,  9.9562e-01, -9.7916e-01,  7.5928e-01,
         -4.8031e-01,  7.4318e-01,  1.8425e-01, -8.4487e-01, -9.4355e-01,
          9.0157e-01,  9.9269e-01, -6.8578e-01, -9.9340e-01, -6.4869e-01,
         -9.4570e-01,  9.2831e-01, -3.0517e-01, -9.9870e-01,  2.9559e-01,
          2.9331e-01, -9.9768e-01,  3.8114e-01,  8.6011e-01,  8.8517e-01,
          3.4167e-01,  9.6720e-01,  9.9889e-01,  7.7632e-01,  8.9224e-01,
          4.7610e-02, -9.6585e-01,  1.2689e-01, -9.9419e-01, -3.5128e-02,
         -4.7663e-01, -3.1027e-01, -1.0455e-01, -2.0928e-01, -4.8485e-01,
         -9.3768e-01,  7.7585e-01, -5.9740e-01, -6.5700e-01,  5.5275e-01,
          2.5154e-01,  5.1355e-01,  7.1683e-03,  2.5394e-01,  9.9928e-01,
         -8.8359e-01,  9.9998e-01, -4.7833e-01,  9.9625e-01,  9.8034e-01,
          8.4465e-01,  9.8003e-01,  1.6796e-01, -7.0433e-01,  9.3027e-01,
          9.8459e-01,  5.3007e-02,  8.8496e-01, -7.9314e-01, -7.1178e-01,
         -6.5593e-01,  4.1004e-01, -2.6967e-02, -3.3386e-01,  8.3128e-01,
          3.5401e-01,  2.0629e-01,  9.5122e-01, -8.9329e-01,  2.0328e-01,
         -9.1920e-01,  6.7895e-01, -9.9842e-01,  9.1663e-01,  9.9817e-01,
         -8.4310e-01, -9.9568e-01,  9.6626e-01, -3.2164e-01,  7.8766e-01,
         -9.1291e-01, -9.3446e-01, -9.9416e-01,  3.0210e-01, -8.4984e-01,
         -5.6352e-01, -9.6898e-01,  1.1574e-01,  3.0161e-01,  9.9834e-01,
          4.8824e-02, -4.1578e-02,  4.8005e-01,  7.3397e-01,  1.9937e-01,
         -9.2599e-01,  4.2440e-01,  6.5718e-01, -5.7298e-01,  8.8233e-01,
         -7.4233e-01, -9.5762e-01, -1.9664e-01, -7.8568e-01, -1.1821e-01,
          9.7884e-01, -9.4399e-01, -7.2727e-01,  5.4198e-02,  1.9404e-01,
         -4.9110e-01,  8.2735e-01, -8.0067e-02,  2.0968e-02,  9.9901e-01,
         -5.9589e-02,  8.4158e-01,  9.8750e-01,  1.3850e-01,  7.2535e-01,
         -4.0384e-01,  5.9743e-01,  5.7414e-01,  4.9631e-01, -8.9373e-01,
          6.9633e-01, -9.8237e-01, -8.8744e-01,  9.9513e-01, -1.0098e-01,
          9.9899e-01, -9.9563e-01,  7.7057e-01, -9.9931e-01,  3.2824e-01,
          4.3158e-01, -3.3865e-01, -7.4130e-01,  9.1138e-01,  9.6383e-01,
          1.9928e-01, -2.2375e-01,  7.0260e-01, -6.2423e-01,  1.9780e-01,
          8.5282e-01,  8.7318e-01, -9.6014e-01,  9.9993e-01,  6.2282e-01,
          6.3219e-01,  8.2410e-01, -1.4354e-01, -8.5298e-01, -2.7889e-01,
          9.5207e-01, -9.8743e-01,  6.7414e-01, -6.0003e-01,  9.9592e-01,
          9.3330e-01, -5.6973e-01, -3.4241e-01,  9.9557e-01,  7.7559e-01,
          6.5826e-01, -7.1723e-01, -6.8991e-01, -7.7092e-01,  1.1845e-01,
          1.3604e-01,  9.3610e-01,  9.9729e-01, -9.6913e-01,  9.9326e-01,
          9.9938e-01, -7.1012e-01,  9.1073e-01,  4.3510e-01, -9.8523e-01,
         -9.2912e-01, -9.3998e-01,  8.5816e-01, -8.4729e-01, -4.6397e-01,
         -6.3103e-01,  8.2389e-01,  7.4649e-01,  8.2860e-01, -9.8537e-01,
          2.5515e-01,  8.9841e-01, -1.1844e-01,  9.9836e-01,  2.3407e-01,
         -9.9667e-01, -2.8858e-02, -2.1615e-01,  6.0337e-01, -4.1957e-01,
          8.0264e-01,  2.7569e-01,  7.1465e-01,  8.5542e-01, -9.9998e-01,
          6.8369e-01,  1.6808e-01, -7.1864e-01,  8.9088e-01,  9.5724e-01,
         -9.8966e-02, -2.5187e-01, -9.6987e-02, -6.8404e-03,  9.9814e-01,
         -9.9630e-01, -1.9335e-01,  6.1945e-01, -9.8355e-01, -9.8961e-01,
          9.7582e-01, -3.1255e-01,  2.2473e-01, -2.2230e-01, -9.7649e-01,
         -1.4824e-01, -3.5875e-01,  9.8497e-01,  5.3239e-01,  8.8178e-01,
         -9.9730e-01, -4.4284e-01, -9.6103e-01, -4.8817e-01,  1.4802e-01,
          6.1585e-01, -4.6991e-01, -9.2799e-01, -8.1915e-01,  7.7016e-01,
         -5.0174e-01,  3.9597e-01, -1.2818e-01,  9.4493e-01, -7.9512e-01,
         -7.8585e-01,  1.5614e-01, -9.7629e-01,  9.9422e-01,  8.6851e-01,
          7.9069e-01,  9.3161e-01, -9.8404e-01,  6.6977e-01, -8.8025e-01,
         -4.5013e-02, -9.9991e-01, -4.8750e-01, -8.0045e-01,  8.4330e-01,
         -3.4572e-01,  9.7199e-01, -9.5050e-01, -5.7389e-01, -1.8202e-01,
         -9.9788e-01,  8.9967e-01, -2.7397e-01,  9.8221e-01, -4.5857e-01,
          6.7087e-01,  9.6725e-01,  4.6989e-01, -9.8196e-01, -9.9843e-01,
          4.6818e-01,  9.9980e-01, -9.4615e-01, -3.5661e-01,  9.9729e-01,
         -8.2695e-01, -9.5595e-01, -8.9882e-01, -9.9511e-01, -9.9280e-01,
         -5.1073e-01,  6.3383e-01, -3.8175e-01,  9.1867e-01, -2.0110e-01,
         -1.3123e-01,  9.6495e-01,  9.9962e-01, -1.2248e-01,  4.7975e-01,
          6.6455e-03, -9.1898e-01, -9.9990e-01, -4.3956e-01, -3.0551e-01,
         -9.9926e-01,  9.9502e-01, -9.7997e-01,  9.9990e-01, -1.0232e-01,
         -6.0008e-01,  8.5638e-01,  1.9596e-01, -4.4011e-01,  4.9920e-02,
          9.9825e-01,  8.3025e-01, -3.9361e-01,  1.4551e-01, -1.0702e-01,
          1.8258e-01, -6.4565e-01,  8.8498e-01,  6.7782e-01,  6.5983e-03,
         -7.3039e-01,  5.5005e-01, -2.0029e-01, -9.8165e-01,  7.2040e-01,
         -3.0614e-03,  9.2245e-01,  1.8529e-01,  8.5744e-01,  8.9460e-01,
         -1.3539e-01,  9.2388e-01, -1.8151e-01, -9.9645e-01, -3.0411e-01,
          2.2579e-01, -6.6895e-01,  5.7391e-01,  8.3677e-01,  9.5258e-01,
         -6.3147e-01,  9.9935e-01, -2.9874e-01,  5.4113e-01, -8.3108e-01,
          9.9919e-01, -9.9628e-01,  3.1210e-01, -3.4364e-01, -2.6626e-01,
          7.8434e-01,  9.5832e-01, -8.4009e-02,  5.4610e-01, -8.7244e-01,
         -2.3571e-01,  6.6305e-01,  9.3699e-01, -3.9064e-01, -3.0298e-01,
         -6.4040e-01,  9.3978e-01,  9.8570e-01,  2.2312e-01, -3.3801e-01,
         -6.6308e-01, -7.4925e-01,  8.4893e-01, -5.0883e-01,  6.7457e-01,
         -2.6977e-01,  2.4521e-01, -9.1442e-05,  9.1263e-01,  6.0908e-01,
          8.2846e-02,  2.1685e-01, -9.4737e-01, -3.4065e-01,  6.5740e-01,
          9.9668e-01, -9.5208e-01, -5.5126e-01,  9.6338e-01, -3.4305e-01,
          2.4484e-01,  7.3913e-01,  7.8805e-01, -8.9833e-01, -1.7404e-01,
         -9.9657e-01, -1.5906e-01, -3.3435e-01,  3.4015e-01,  9.0850e-01,
          1.0230e-02,  2.0201e-01,  4.3318e-01, -9.6923e-01,  9.1216e-01,
          2.7533e-01,  9.6451e-01,  8.7790e-01, -7.9498e-02, -1.5370e-01,
          4.6434e-01,  3.5428e-01, -3.1950e-01,  9.6091e-01, -9.0452e-01,
          9.9899e-01, -3.2348e-01, -9.9824e-01, -6.8585e-01,  8.4794e-01,
         -9.9891e-01, -5.7393e-01, -9.9771e-01,  9.5344e-01, -1.8158e-01,
         -7.9982e-01, -7.3221e-01, -9.9847e-01, -9.9982e-01, -4.8556e-01,
         -4.0170e-01, -2.3429e-01, -2.0992e-01,  9.8811e-01, -1.4578e-01,
          6.4324e-01,  1.3404e-01, -9.3664e-01,  5.4135e-01, -7.4273e-01,
         -1.5878e-01, -9.9909e-01,  3.1261e-01,  8.8962e-01, -3.7441e-01,
         -4.2346e-01, -9.1223e-01,  8.1511e-01, -9.0731e-01,  6.2915e-01,
          9.6429e-01,  7.8230e-01,  8.2704e-01, -9.9054e-01,  9.5241e-01,
         -4.8401e-01,  3.3649e-01, -8.8031e-01, -9.6511e-01,  9.9813e-01,
         -3.9378e-01, -1.8263e-01, -3.3503e-01, -9.9461e-01,  7.1867e-01,
         -4.3959e-01, -2.3022e-01, -9.5571e-01,  3.9417e-01, -8.8943e-01,
         -9.9803e-01,  3.4004e-01,  7.9589e-01,  9.9613e-01,  9.3591e-01,
          9.0179e-01, -2.0220e-01, -7.6146e-01,  5.4621e-02, -9.9915e-01,
         -1.2446e-01,  1.6197e-01, -9.6547e-01,  8.0155e-01,  9.8064e-01,
          9.7242e-01, -1.7942e-01, -8.8789e-01, -3.6468e-01,  2.7316e-01,
          8.3758e-01,  8.3901e-01, -6.9883e-01,  4.2700e-02,  1.8337e-02,
         -9.6542e-01, -9.0104e-01,  9.8900e-01, -9.3347e-01,  9.4337e-01,
          9.0828e-01,  8.7582e-01,  3.4694e-01, -3.9620e-01, -3.8469e-01,
         -9.9980e-01, -9.2446e-01, -1.8374e-01, -9.9851e-01,  9.9605e-01,
         -9.9873e-01, -5.5518e-01,  5.9460e-01,  2.4370e-01,  9.6660e-01,
          1.1288e-01, -9.9874e-01, -9.9768e-01, -7.9306e-01, -6.2431e-01,
          9.7396e-01,  3.0210e-01,  1.4691e-01,  8.2036e-01,  7.6171e-01,
          9.9412e-01, -6.0124e-01,  5.7753e-01, -7.5033e-01,  9.9492e-01,
          4.5075e-01, -9.8646e-01,  7.9964e-01, -9.9706e-01,  7.0638e-01,
          7.5888e-01,  9.4407e-01,  9.4101e-01, -9.2524e-01,  9.9845e-01,
         -9.9846e-01,  9.9841e-01, -9.9941e-01, -8.7869e-01,  9.9847e-01,
         -9.7701e-01,  8.4340e-01, -9.9157e-01, -6.5512e-01, -8.2674e-01,
          2.5582e-01, -7.8063e-01,  9.7210e-01, -9.7216e-01, -9.9474e-01,
          2.7511e-01,  2.8544e-01,  4.1619e-01,  8.7502e-01,  6.0008e-01,
          9.6868e-01,  4.8907e-01,  9.1979e-01, -1.9830e-01,  5.9957e-01,
          9.9987e-01, -3.2001e-01, -9.6526e-02, -9.7239e-01,  9.5947e-01,
          6.5982e-01,  3.8878e-01,  8.6373e-01, -4.4506e-01,  5.4288e-01,
          7.4797e-01, -9.8234e-01, -3.4124e-02, -9.9063e-01,  4.9642e-01,
         -5.1254e-01,  4.9145e-01,  2.1373e-01,  3.2248e-01, -4.0851e-02,
         -9.7424e-01,  2.4766e-01, -9.9303e-01,  9.5320e-01, -7.1113e-01,
          1.2842e-01, -2.8780e-01,  9.0068e-01, -5.4970e-01,  9.9366e-01,
          9.9647e-01, -9.9998e-01,  4.3145e-01,  9.7191e-01,  8.0509e-01,
          9.2468e-01, -9.7540e-01, -3.1610e-01,  8.5833e-01, -6.9730e-01,
          9.5603e-01,  7.0491e-01, -9.5361e-02,  9.1543e-01, -9.6698e-01,
          1.8062e-01, -9.1130e-01,  1.0774e-01,  8.9507e-01, -9.3751e-01,
          2.9478e-01,  6.3783e-01,  3.1565e-01, -9.9807e-01, -3.3766e-01,
         -9.9504e-01, -4.3560e-01,  9.9048e-01,  9.6262e-01,  9.9739e-01,
          7.2891e-01, -2.3598e-01, -7.6804e-02, -9.9535e-01, -8.5958e-01,
          2.6892e-01, -2.9059e-01,  7.1036e-01,  8.6598e-01,  4.0921e-01,
         -5.6847e-01, -9.9798e-01,  5.2852e-02,  8.2901e-01, -3.6962e-01,
          9.3911e-01,  7.9441e-01, -8.9709e-01, -7.8058e-01, -9.1496e-01,
          3.4024e-01, -5.4400e-01, -4.8460e-01,  5.0350e-01, -4.6387e-01,
          9.9886e-01, -9.9019e-01, -6.8870e-01, -8.3693e-01, -7.7146e-01,
          1.3762e-01,  2.1174e-01, -5.5411e-02,  6.0308e-01,  3.2018e-01,
         -8.9683e-01,  9.7571e-01, -9.7656e-01, -9.7959e-01,  9.9463e-01,
         -5.5416e-01, -5.3156e-01, -2.5316e-01, -7.0400e-01, -3.7305e-01,
         -5.9944e-02,  5.0992e-01, -5.0298e-01, -2.9130e-01, -9.9881e-01,
         -7.4625e-01,  3.7078e-01, -9.6503e-01, -9.1905e-01, -5.0869e-01,
         -9.9999e-01,  9.9284e-01,  8.1545e-01,  9.9828e-01, -9.9729e-01,
          9.3996e-01,  4.4080e-01,  9.9509e-01, -1.3041e-01, -6.0139e-01,
          4.5045e-01,  9.9753e-01,  7.4395e-01, -7.2094e-01, -2.1877e-01,
         -3.4587e-01, -2.7355e-01, -8.1835e-01,  6.4384e-01, -2.5727e-01,
          8.4031e-01, -9.5566e-01, -9.9773e-01,  9.9749e-01, -3.4037e-02,
          9.6736e-01,  2.5490e-01,  4.3039e-02, -9.1157e-01,  7.7833e-01,
         -4.3414e-01, -8.4685e-01, -9.9902e-01,  9.0856e-01, -9.9998e-01,
         -9.5288e-01,  5.6237e-01,  9.8130e-01, -9.9269e-01, -8.9115e-01,
         -8.5638e-02, -9.9889e-01, -2.2623e-01, -8.5059e-01,  3.4062e-01,
         -9.7387e-01,  8.2079e-01, -7.9265e-01,  8.7170e-01,  9.5354e-01,
         -8.5100e-01,  3.6714e-01,  6.3911e-02,  9.9534e-01,  2.4536e-02,
          7.2902e-01,  5.8514e-01, -8.2716e-01, -4.5634e-01, -8.3606e-01,
         -2.3288e-01, -9.6367e-01, -7.9507e-01,  9.6957e-01,  9.2124e-01,
         -9.9666e-01, -9.8081e-01,  8.3492e-01, -2.6442e-01,  9.7069e-01,
         -8.5257e-01, -9.9761e-01, -9.9847e-01, -2.4608e-01, -6.5073e-01,
          9.8551e-01, -2.7737e-01,  9.9994e-01,  8.6573e-01,  4.8107e-01,
         -7.8910e-01,  2.2018e-01,  3.6846e-01,  6.4627e-01, -3.5463e-01,
          9.9930e-01,  7.0647e-01,  9.6527e-01],
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, three, hook2a]),
                          List([three])])])
        : [-0.1696,  0.6456,  0.9928, -0.9531,  0.7475, -0.4578,  0.6849,  0.0665,
         -0.7997, -0.7235,  0.8001,  0.9864, -0.4445, -0.9806, -0.7003, -0.8626,
          0.8837, -0.4345, -0.9983,  0.2347,  0.3011, -0.9952,  0.4498,  0.8031,
          0.6283,  0.2921,  0.9422,  0.9988,  0.5656,  0.9510,  0.0750, -0.9201,
          0.3151, -0.9796, -0.1581, -0.1934, -0.3280, -0.0661, -0.5642, -0.1747,
         -0.8417,  0.6016, -0.6177, -0.4716, -0.2113,  0.2419,  0.6644,  0.0553,
          0.0370,  0.9980, -0.7507,  0.9999, -0.4476,  0.9913,  0.9673,  0.6224,
          0.9647,  0.0580, -0.3290,  0.8433,  0.9579, -0.1274,  0.7818, -0.6997,
         -0.7708, -0.6569,  0.5580,  0.3302,  0.0183,  0.7356,  0.3368,  0.4534,
          0.8859, -0.7554,  0.3285, -0.8433,  0.7058, -0.9981,  0.8767,  0.9971,
         -0.7158, -0.9948,  0.9544, -0.3143,  0.6422, -0.8565, -0.8764, -0.9881,
          0.1601, -0.8384, -0.8389, -0.9173,  0.2155,  0.4971,  0.9977,  0.6004,
         -0.1644,  0.3808,  0.3808,  0.0705, -0.7509,  0.0919,  0.4775, -0.2150,
          0.6883, -0.6744, -0.8913,  0.1543, -0.8586,  0.0247,  0.9135, -0.9076,
         -0.6857,  0.0737,  0.1931, -0.5917,  0.7443,  0.4937, -0.0213,  0.9985,
          0.0898,  0.5792,  0.9694, -0.0056,  0.5303, -0.3329,  0.4662,  0.5251,
          0.5527, -0.7338,  0.7220, -0.9457, -0.7647,  0.9891, -0.1582,  0.9979,
         -0.9947,  0.5280, -0.9988, -0.1208,  0.7102, -0.1982, -0.5705,  0.8788,
          0.9645,  0.3679,  0.4944,  0.6960, -0.1331,  0.2809,  0.6999,  0.6421,
         -0.8753,  0.9998,  0.4528,  0.5294,  0.5690,  0.0216, -0.5791, -0.3982,
          0.9564, -0.9868,  0.4480, -0.1618,  0.9915,  0.9160, -0.5326, -0.1380,
          0.9919,  0.7516,  0.2977, -0.7061, -0.7195, -0.5985,  0.2465,  0.3056,
          0.8451,  0.9892, -0.9184,  0.9857,  0.9987, -0.4281,  0.8401, -0.0455,
         -0.9725, -0.7870, -0.9232,  0.8234, -0.8715, -0.4866, -0.0247,  0.7353,
          0.6693,  0.6916, -0.9765,  0.1461,  0.8902, -0.0924,  0.9976,  0.5690,
         -0.9937, -0.0840, -0.5313,  0.6484, -0.4066,  0.6677,  0.3503,  0.4483,
          0.8629, -1.0000,  0.5088, -0.0496, -0.8258,  0.7296,  0.9599,  0.1476,
         -0.4205, -0.0418,  0.1847,  0.9972, -0.9919, -0.2549,  0.5046, -0.9662,
         -0.9814,  0.8881, -0.3584,  0.4280, -0.2816, -0.9622, -0.1426,  0.0094,
          0.9525,  0.5838,  0.9343, -0.9957,  0.1484, -0.9135, -0.3554,  0.2365,
          0.5320, -0.3806, -0.8909, -0.5883,  0.4295, -0.2347,  0.4592, -0.0618,
          0.8819, -0.6391, -0.8678,  0.3838, -0.9586,  0.9831,  0.7763,  0.4308,
          0.8518, -0.9728,  0.4780, -0.5048, -0.1710, -0.9996, -0.4617, -0.9524,
          0.8662, -0.4720,  0.9526, -0.8782, -0.5053, -0.2939, -0.9949,  0.7777,
         -0.3348,  0.9878, -0.5387,  0.5637,  0.9472,  0.4979, -0.9681, -0.9972,
          0.5491,  0.9995, -0.9230, -0.3431,  0.9969, -0.4359, -0.8081, -0.6777,
         -0.9888, -0.9889, -0.3436,  0.5485, -0.3159,  0.8543, -0.5830, -0.2327,
          0.9590,  0.9991, -0.5365,  0.4837,  0.2299, -0.8093, -0.9998, -0.5013,
         -0.2875, -0.9988,  0.9934, -0.9717,  0.9998, -0.5849, -0.4520,  0.6880,
          0.2007, -0.0213,  0.2167,  0.9977,  0.6813, -0.2476,  0.2523, -0.0842,
          0.3841, -0.4250,  0.5608,  0.7287,  0.2084, -0.8460, -0.0996, -0.3480,
         -0.9675,  0.1056,  0.1259,  0.7041,  0.0780,  0.6237,  0.8460, -0.1775,
          0.8948, -0.4557, -0.9926, -0.2613,  0.3686, -0.4869,  0.6803,  0.5974,
          0.9240, -0.6574,  0.9982, -0.2079,  0.5065, -0.7335,  0.9988, -0.9923,
          0.3872, -0.2078, -0.3317,  0.8314,  0.8506,  0.2244,  0.6701, -0.4831,
         -0.1354,  0.5958,  0.9111, -0.4008, -0.3457, -0.2631,  0.8130,  0.8802,
          0.0668, -0.3278, -0.7652, -0.3226,  0.6993, -0.4887,  0.8428, -0.3323,
          0.1115,  0.2722,  0.8147,  0.5228,  0.1422,  0.1884, -0.8437, -0.4727,
          0.6176,  0.9966, -0.9132, -0.2622,  0.9205, -0.3009, -0.1393,  0.5726,
          0.6935, -0.8333, -0.2887, -0.9915, -0.0585, -0.0686,  0.3236,  0.8759,
          0.0255, -0.1230,  0.4656, -0.8305,  0.7619,  0.0970,  0.8696,  0.7969,
         -0.1653, -0.1789,  0.5391,  0.4072, -0.3172,  0.8840, -0.8577,  0.9962,
          0.1104, -0.9958, -0.4461,  0.7869, -0.9953, -0.6532, -0.9960,  0.9151,
         -0.7518, -0.5049, -0.6004, -0.9929, -0.9995, -0.4479, -0.4705, -0.3183,
         -0.0471,  0.9840, -0.1491,  0.3699,  0.1207, -0.8933,  0.2962, -0.4966,
         -0.1535, -0.9978,  0.4490,  0.8382, -0.3169, -0.4491, -0.7552,  0.5248,
         -0.9116,  0.4630,  0.8467,  0.5627,  0.8810, -0.9821,  0.9270, -0.4367,
          0.4822, -0.7364, -0.9312,  0.9965, -0.5145,  0.0275, -0.2932, -0.9832,
          0.2495, -0.4358, -0.0865, -0.8925,  0.4770, -0.7918, -0.9952,  0.1885,
          0.6329,  0.9884,  0.8772,  0.8046, -0.3310, -0.6608,  0.3114, -0.9984,
         -0.2093,  0.1256, -0.8668,  0.8450,  0.9727,  0.9290, -0.0302, -0.6578,
         -0.7467,  0.4053,  0.7258,  0.7900, -0.6940,  0.0239, -0.1412, -0.9395,
         -0.5528,  0.9853, -0.7911,  0.9209,  0.7268,  0.6869,  0.5466,  0.0981,
          0.0268, -0.9981, -0.8207, -0.1036, -0.9962,  0.9919, -0.9980, -0.2764,
          0.7935, -0.1045,  0.9183,  0.0128, -0.9974, -0.9961, -0.7957, -0.6043,
          0.9613,  0.1563,  0.3043,  0.8191,  0.5592,  0.9901, -0.1068,  0.8008,
         -0.4218,  0.9923,  0.3961, -0.9645,  0.6119, -0.9947,  0.8404,  0.6246,
          0.8487,  0.7772, -0.7612,  0.9971, -0.9966,  0.9979, -0.9986, -0.6711,
          0.9937, -0.9301,  0.5826, -0.9873, -0.3922, -0.7255,  0.0597, -0.7279,
          0.9492, -0.9729, -0.9879,  0.2450,  0.6621,  0.5220,  0.6862,  0.6489,
          0.9535,  0.3526,  0.8315, -0.2207,  0.5820,  0.9996, -0.5547,  0.4058,
         -0.9457,  0.9135,  0.5540,  0.3636,  0.6023, -0.3918,  0.5987,  0.6785,
         -0.9805, -0.3343, -0.9733,  0.4612, -0.6344,  0.5130,  0.2402,  0.2463,
          0.1061, -0.9400,  0.4325, -0.9907,  0.8784, -0.4657,  0.3202, -0.2176,
          0.8858, -0.5582,  0.9867,  0.9908, -0.9999,  0.4295,  0.9400,  0.7440,
          0.7372, -0.9583, -0.4372,  0.8111, -0.4015,  0.9145,  0.6623,  0.0044,
          0.6848, -0.9637,  0.0200, -0.6658,  0.0084,  0.6638, -0.9160,  0.2562,
          0.6434,  0.3809, -0.9940, -0.7054, -0.9913, -0.3146,  0.9661,  0.9074,
          0.9961,  0.3857, -0.5198, -0.0116, -0.9943, -0.5923,  0.3417, -0.3161,
          0.4031,  0.5137,  0.5038, -0.4637, -0.9951,  0.3042,  0.7025, -0.0333,
          0.7819,  0.7897, -0.7137, -0.6654, -0.7646,  0.3016, -0.5179, -0.6500,
          0.3190, -0.7420,  0.9982, -0.9806, -0.4380, -0.5150, -0.7265,  0.2497,
          0.4148,  0.0383,  0.7382,  0.3440, -0.7363,  0.9494, -0.9713, -0.9481,
          0.9887, -0.4651, -0.2743, -0.2122, -0.5741, -0.2670, -0.1397,  0.8264,
         -0.3814, -0.4242, -0.9965, -0.6617,  0.0723, -0.9364, -0.7383, -0.4227,
         -1.0000,  0.9775,  0.5556,  0.9970, -0.9942,  0.6669,  0.4235,  0.9915,
         -0.1084, -0.7509,  0.1787,  0.9924,  0.4568, -0.6498, -0.0358, -0.4113,
         -0.1271, -0.6861,  0.3730, -0.0602,  0.7578, -0.9323, -0.9953,  0.9951,
         -0.1346,  0.9467,  0.1128,  0.5602, -0.6619,  0.8301, -0.2098, -0.8193,
         -0.9973,  0.8468, -0.9999, -0.9344,  0.5391,  0.9216, -0.9868, -0.8191,
         -0.0069, -0.9978, -0.2882, -0.5315,  0.3369, -0.9521,  0.6689, -0.5933,
          0.8218,  0.9034, -0.7964,  0.2817,  0.0321,  0.9863,  0.1222,  0.6543,
          0.5657, -0.5243, -0.1471, -0.5304, -0.6993, -0.9429, -0.4195,  0.9545,
          0.8437, -0.9938, -0.9596,  0.6253, -0.2568,  0.9182, -0.6713, -0.9959,
         -0.9976, -0.2159, -0.4009,  0.9509, -0.1220,  0.9998,  0.7477,  0.6132,
         -0.7429,  0.4070,  0.2464,  0.6087, -0.5002,  0.9989,  0.7043,  0.9375],
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, five, zero, hook2a]),
                          List([five, zero])])])
        : [-0.0536,  0.6557,  0.9790, -0.9358,  0.3839,  0.4686,  0.7389,  0.0607,
         -0.7969, -0.7108,  0.8346,  0.9680, -0.5858, -0.9445, -0.1296, -0.8280,
          0.8356, -0.1896, -0.9885, -0.1990, -0.2879, -0.9819,  0.5358,  0.9067,
          0.7165, -0.0384,  0.8466,  0.9906,  0.7645,  0.8481,  0.2224, -0.8960,
          0.5074, -0.9749, -0.0917, -0.3979, -0.5264, -0.1385, -0.0525, -0.4723,
         -0.9054,  0.1312, -0.3442, -0.4688,  0.2115,  0.5028,  0.1467, -0.0763,
          0.1202,  0.9848, -0.7812,  0.9942, -0.6100,  0.9771,  0.9588,  0.6701,
          0.9430,  0.1792, -0.6164,  0.6073,  0.9309, -0.1730,  0.6566, -0.4109,
         -0.4892, -0.3797,  0.1523, -0.0166, -0.0664,  0.5235,  0.2756,  0.3145,
          0.8922, -0.5516,  0.1731, -0.8393,  0.2731, -0.9904,  0.9114,  0.9789,
         -0.6507, -0.9833,  0.9717, -0.5732,  0.0419, -0.3512, -0.8555, -0.9733,
          0.3972, -0.7200, -0.4596, -0.9056,  0.3809,  0.1421,  0.9909,  0.7025,
         -0.2147,  0.3624,  0.6287, -0.4051, -0.5197,  0.7875,  0.6745, -0.4962,
          0.7724, -0.2135, -0.8273, -0.7108, -0.6317, -0.2103,  0.9411, -0.9063,
         -0.8646,  0.1441,  0.7642, -0.4823,  0.8772,  0.6724, -0.2717,  0.9917,
         -0.2717,  0.8668,  0.9304,  0.4817,  0.3887, -0.4235,  0.1332,  0.7248,
          0.2828, -0.7244,  0.4361, -0.9158, -0.7139,  0.9702, -0.1615,  0.9956,
         -0.9826,  0.6882, -0.9947, -0.5355,  0.5818, -0.3021, -0.7778,  0.7144,
          0.9309,  0.3996, -0.2134, -0.0069, -0.1198, -0.2241,  0.5358,  0.6627,
         -0.8916,  0.9922,  0.5253,  0.6079,  0.9031, -0.1689, -0.6369,  0.2755,
          0.9361, -0.9815,  0.7488, -0.4536,  0.9675,  0.8379,  0.4994, -0.4269,
          0.9803,  0.5393, -0.2663, -0.4949, -0.7636, -0.6384, -0.0148,  0.1226,
          0.7489,  0.9593, -0.8784,  0.9528,  0.9777, -0.2288,  0.6968,  0.4729,
         -0.8894, -0.7717, -0.9027,  0.6311, -0.7045, -0.0351, -0.1843,  0.5541,
          0.5448,  0.6413, -0.8756,  0.1843,  0.8514, -0.3326,  0.9910,  0.2134,
         -0.9857, -0.6859,  0.0080,  0.8350, -0.1960,  0.8769, -0.0018,  0.2496,
          0.9576, -0.9968,  0.5279,  0.1561, -0.4950,  0.6149,  0.9408, -0.6242,
         -0.4286, -0.1647,  0.1672,  0.9866, -0.9827, -0.1818,  0.4172, -0.9408,
         -0.9685,  0.9191, -0.4075,  0.4622, -0.3914, -0.8527, -0.0961, -0.0310,
          0.9452,  0.0289,  0.5073, -0.9873, -0.4268, -0.8511, -0.4475,  0.3256,
          0.4677, -0.3216, -0.6322, -0.7893,  0.7037, -0.1287,  0.3161, -0.4311,
          0.7676, -0.7384, -0.3682, -0.5307, -0.9120,  0.9729,  0.4244,  0.6547,
          0.8989, -0.9591,  0.4681, -0.8537, -0.0725, -0.9910, -0.1633, -0.7602,
          0.6972, -0.3802,  0.9330, -0.7982, -0.6307, -0.0795, -0.9891,  0.7377,
         -0.3301,  0.9713, -0.4406,  0.6746,  0.9381,  0.7485, -0.9538, -0.9918,
          0.7446,  0.9861, -0.8918, -0.2856,  0.9865, -0.5633, -0.8464, -0.8037,
         -0.9812, -0.9664, -0.2596, -0.3404, -0.2723,  0.8963, -0.4856, -0.1099,
          0.9402,  0.9935, -0.6134,  0.0040,  0.2582, -0.7224, -0.9880,  0.1391,
         -0.1452, -0.9942,  0.9787, -0.9798,  0.9920, -0.1902, -0.4914,  0.7596,
         -0.0606, -0.2045,  0.0099,  0.9941,  0.6878, -0.3037,  0.0206,  0.2004,
          0.3499, -0.0178,  0.5606,  0.1937,  0.0308, -0.7612,  0.1985,  0.2989,
         -0.9706,  0.5679,  0.0919,  0.8231, -0.0447,  0.6549,  0.8515, -0.1725,
          0.8274, -0.1919, -0.9053, -0.3311,  0.3356, -0.5868,  0.4433,  0.7065,
          0.9473, -0.7237,  0.9895, -0.3489,  0.7391, -0.7994,  0.9926, -0.9808,
          0.2825, -0.1992,  0.6865,  0.6885,  0.8748,  0.4223,  0.8205, -0.5135,
         -0.5223,  0.6855,  0.8701, -0.2760, -0.4068, -0.4485,  0.3132,  0.8540,
          0.1985,  0.0136, -0.4732, -0.3478,  0.7304, -0.2329,  0.1451, -0.0519,
          0.0294,  0.5320,  0.7061,  0.3035,  0.2800,  0.1383, -0.8824, -0.4620,
          0.7880,  0.9837, -0.9013, -0.2182,  0.9465, -0.4468, -0.0524,  0.4673,
          0.6895, -0.8638, -0.1694, -0.9768, -0.2648, -0.1442, -0.0375,  0.5666,
          0.1747, -0.1844,  0.3448, -0.6224,  0.7456, -0.4458,  0.8639,  0.7445,
         -0.1121, -0.1468,  0.3576,  0.4916, -0.0710,  0.9395, -0.8961,  0.9866,
          0.0122, -0.9828, -0.7277,  0.4548, -0.9851, -0.4481, -0.9829,  0.8676,
         -0.3652, -0.7943, -0.7602, -0.9774, -0.9864,  0.0580,  0.3477, -0.0542,
         -0.1164,  0.7265, -0.2636,  0.2942,  0.1293, -0.8437, -0.0555, -0.5025,
          0.4502, -0.9922, -0.4922,  0.9079, -0.2787, -0.3020, -0.8609,  0.8160,
         -0.8970,  0.2805,  0.9185,  0.0446,  0.8410, -0.9009,  0.9328, -0.0109,
          0.4862, -0.9284, -0.9343,  0.9878, -0.7205,  0.0237, -0.2712, -0.9712,
          0.6549, -0.2056, -0.6351, -0.9514,  0.4024, -0.8611, -0.9919,  0.2302,
          0.5893,  0.9793,  0.8823,  0.6285, -0.2864, -0.5431, -0.0873, -0.9952,
         -0.1219,  0.3727, -0.9410,  0.2872,  0.9659,  0.9217, -0.8287, -0.7417,
         -0.1473,  0.0758,  0.5948,  0.3574, -0.2845,  0.1839,  0.0316, -0.9479,
         -0.4586,  0.9690, -0.8736,  0.9243,  0.7869,  0.7868,  0.2012, -0.1565,
         -0.1810, -0.9876, -0.5921, -0.0988, -0.9871,  0.9766, -0.9881, -0.4250,
          0.5167,  0.1419,  0.9327,  0.4119, -0.9934, -0.9863, -0.9512, -0.1951,
          0.9307,  0.0840,  0.0174,  0.5760,  0.3590,  0.9734, -0.6170,  0.3006,
         -0.5201,  0.9788,  0.2818, -0.9613,  0.6924, -0.9885,  0.6333,  0.5572,
          0.8273,  0.7636, -0.8591,  0.9782, -0.9813,  0.9874, -0.9926, -0.7471,
          0.9904, -0.8634, -0.1042, -0.9757, -0.4015, -0.7048,  0.3918, -0.3852,
          0.8829, -0.9650, -0.9802,  0.0991,  0.3854,  0.5123,  0.7801,  0.0515,
          0.9588,  0.3284,  0.8159, -0.1281,  0.8088,  0.9972, -0.5360, -0.0627,
         -0.8653,  0.7766,  0.1193,  0.4727,  0.6256, -0.3303,  0.1535,  0.5562,
         -0.9748, -0.3071, -0.7631,  0.2891, -0.4760,  0.3897,  0.1580,  0.3131,
         -0.1244, -0.9257,  0.2435, -0.9646,  0.8889, -0.5241,  0.1792, -0.1911,
          0.4698, -0.7068,  0.9635,  0.9832, -0.9974,  0.4128,  0.9492,  0.5522,
          0.8762, -0.9043, -0.4938,  0.8489, -0.3906,  0.8815,  0.6269, -0.0292,
          0.8006, -0.9000, -0.2356, -0.7193, -0.1806,  0.7226, -0.8799,  0.2631,
          0.6176,  0.0614, -0.9868, -0.6080, -0.9828, -0.1239,  0.9652,  0.6252,
          0.9847,  0.6414, -0.5247, -0.0209, -0.9797, -0.7637,  0.2847, -0.2523,
          0.2520,  0.7458,  0.4398, -0.2458, -0.9805,  0.1350,  0.8037, -0.1520,
          0.7569,  0.6456, -0.7572, -0.8044, -0.8833,  0.1944, -0.0252, -0.8898,
          0.5816, -0.4683,  0.9861, -0.9359, -0.5867, -0.6725, -0.5032,  0.3117,
          0.3086,  0.1920, -0.3714,  0.5556, -0.8333,  0.8949, -0.9619, -0.9498,
          0.9782, -0.3719, -0.4245,  0.0291, -0.4987, -0.2617, -0.0356,  0.6298,
         -0.8333, -0.1835, -0.9823, -0.7180, -0.0210, -0.8960, -0.6812, -0.1284,
         -0.9988,  0.9496,  0.6767,  0.9886, -0.9833,  0.7774,  0.3320,  0.9669,
         -0.0144, -0.4939,  0.4272,  0.9888,  0.2344,  0.1321, -0.2357, -0.2265,
          0.2889, -0.5194,  0.3077, -0.0265,  0.4106, -0.9672, -0.9852,  0.9836,
         -0.2229,  0.8701,  0.3227,  0.7159, -0.8686,  0.8950, -0.2933, -0.6349,
         -0.9931,  0.8407, -0.9975, -0.9243,  0.4057,  0.9407, -0.9492, -0.8948,
         -0.0269, -0.9865,  0.6362, -0.7255,  0.2171, -0.9297,  0.6733, -0.5768,
          0.5074,  0.8896, -0.8609,  0.5494,  0.0024,  0.8647, -0.1266,  0.5742,
          0.0215, -0.7024, -0.2760, -0.6541,  0.0327, -0.9382, -0.6870,  0.9151,
          0.8429, -0.9604, -0.9473,  0.6974,  0.2121,  0.9127, -0.7349, -0.9838,
         -0.9930, -0.0307, -0.5316,  0.9418, -0.2394,  0.9959,  0.8254,  0.0236,
         -0.6051,  0.2768, -0.0823,  0.4764, -0.6176,  0.9944,  0.2343,  0.8836],
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, five, hook2a]),
                          List([five])])])
        : [-2.6264e-01,  7.4860e-01,  9.9577e-01, -9.4586e-01,  8.1546e-01,
         -5.4691e-01,  8.9901e-01,  2.2506e-01, -8.8496e-01, -8.2661e-01,
          8.8634e-01,  9.9100e-01, -5.3498e-01, -9.8598e-01, -8.0599e-01,
         -9.2141e-01,  9.2811e-01, -3.7806e-01, -9.9866e-01,  4.7300e-01,
          4.9014e-01, -9.9712e-01,  4.3539e-01,  7.7697e-01,  9.0053e-01,
          9.3141e-02,  9.5861e-01,  9.9856e-01,  6.2101e-01,  9.3998e-01,
          2.2750e-01, -9.4878e-01, -4.7959e-02, -9.8885e-01, -3.1208e-01,
         -2.9206e-01, -6.4502e-01, -1.6495e-01, -1.5285e-01, -4.1911e-01,
         -9.1119e-01,  6.6112e-01, -6.8746e-01, -5.5257e-01, -1.4632e-02,
          2.7278e-01,  5.5246e-01, -1.4505e-01, -3.1266e-02,  9.9865e-01,
         -8.5809e-01,  9.9997e-01, -2.1616e-01,  9.9680e-01,  9.8823e-01,
          8.1436e-01,  9.8431e-01,  2.1243e-02, -4.5250e-01,  8.8022e-01,
          9.6862e-01, -2.5549e-01,  8.8779e-01, -7.1610e-01, -6.9398e-01,
         -7.1674e-01,  5.2468e-01,  2.3123e-01, -3.6601e-01,  7.3553e-01,
          4.2476e-01,  3.0971e-01,  9.5962e-01, -7.6797e-01,  3.9817e-01,
         -8.3084e-01,  6.6352e-01, -9.9870e-01,  8.6775e-01,  9.9599e-01,
         -8.7847e-01, -9.9456e-01,  9.8397e-01, -5.8268e-01,  7.6549e-01,
         -9.1259e-01, -8.2418e-01, -9.8574e-01,  2.0808e-01, -8.0916e-01,
         -7.5043e-01, -9.4921e-01,  1.3572e-01,  7.8093e-01,  9.9807e-01,
          6.3357e-01, -3.1520e-01,  3.9240e-01,  6.1093e-01,  2.3364e-01,
         -8.0886e-01,  4.8974e-01,  4.2457e-01, -1.2498e-01,  7.8841e-01,
         -7.5157e-01, -8.7560e-01,  2.8045e-01, -9.0423e-01, -2.5999e-02,
          9.8216e-01, -9.7035e-01, -7.2341e-01,  2.4198e-01,  4.4721e-01,
         -4.2273e-01,  8.8112e-01,  1.4048e-01, -9.0963e-02,  9.9825e-01,
         -1.7136e-01,  5.9692e-01,  9.7450e-01,  1.9794e-02,  6.7082e-01,
         -4.6852e-01,  3.7536e-01,  6.3343e-01,  6.2651e-01, -7.8164e-01,
          5.7921e-01, -9.7422e-01, -6.1556e-01,  9.9200e-01, -2.3564e-01,
          9.9805e-01, -9.9381e-01,  6.5907e-01, -9.9889e-01, -5.6259e-02,
          5.8586e-01, -2.5811e-01, -6.5623e-01,  8.6749e-01,  9.5548e-01,
          2.6720e-01,  2.1173e-01,  7.5291e-01, -3.3838e-01,  7.6124e-01,
          8.2002e-01,  7.2306e-01, -9.4953e-01,  9.9996e-01,  3.9271e-01,
          7.1010e-01,  6.3904e-01,  1.2074e-02, -5.9418e-01, -5.0198e-01,
          9.5996e-01, -9.8612e-01,  7.5956e-01, -5.8622e-01,  9.9280e-01,
          9.1903e-01, -4.3085e-01, -2.5174e-01,  9.9554e-01,  7.9193e-01,
          3.3631e-01, -6.6409e-01, -7.6334e-01, -4.5503e-01,  2.4828e-01,
          3.6572e-01,  8.9721e-01,  9.9349e-01, -9.6783e-01,  9.8903e-01,
          9.9945e-01, -5.3617e-01,  8.9154e-01,  1.6693e-01, -9.7846e-01,
         -9.0925e-01, -9.1332e-01,  8.1218e-01, -9.4711e-01, -5.2697e-01,
         -3.4056e-01,  8.1814e-01,  5.3094e-01,  8.1483e-01, -9.8063e-01,
          2.6405e-01,  9.3177e-01, -2.4474e-01,  9.9686e-01,  5.1947e-01,
         -9.9581e-01, -4.9813e-02, -6.1697e-01,  8.3497e-01, -2.8302e-01,
          8.3942e-01,  2.7888e-01,  5.3579e-01,  9.0990e-01, -9.9998e-01,
          3.7438e-01,  1.2141e-01, -8.9797e-01,  7.6275e-01,  9.9166e-01,
          2.0010e-01, -4.5794e-01, -8.3465e-02,  2.0456e-01,  9.9709e-01,
         -9.9268e-01, -3.5585e-01,  5.2032e-01, -9.7316e-01, -9.9136e-01,
          9.6584e-01, -4.4059e-01,  2.1373e-01, -5.8981e-01, -9.7646e-01,
         -1.0915e-01,  2.8555e-02,  9.6477e-01,  4.8148e-01,  9.3421e-01,
         -9.9689e-01,  2.0570e-01, -8.8418e-01, -2.9457e-01,  9.9664e-02,
          6.3511e-01, -3.8603e-01, -7.8250e-01, -6.4738e-01,  5.8992e-01,
         -4.0092e-01,  4.5316e-01, -5.2091e-01,  8.8385e-01, -3.9862e-01,
         -8.9469e-01,  5.7284e-01, -9.7015e-01,  9.9200e-01,  5.8655e-01,
          5.3657e-01,  9.3694e-01, -9.8447e-01,  3.8471e-01, -5.8568e-01,
         -2.1900e-01, -9.9988e-01, -5.6564e-01, -9.2794e-01,  8.9010e-01,
         -4.2260e-01,  9.7138e-01, -9.2310e-01, -5.1936e-01, -7.1398e-01,
         -9.9639e-01,  8.0627e-01, -3.6990e-01,  9.8651e-01, -8.3598e-01,
          3.1922e-01,  9.7948e-01,  5.2060e-01, -9.7214e-01, -9.9766e-01,
         -6.9495e-02,  9.9980e-01, -9.6848e-01, -3.0511e-01,  9.9707e-01,
         -5.0872e-01, -9.2291e-01, -8.9176e-01, -9.9469e-01, -9.9148e-01,
         -4.8506e-01,  5.3267e-01, -2.8499e-01,  9.3034e-01, -6.7507e-01,
         -1.4506e-01,  9.8077e-01,  9.9972e-01, -5.3817e-01,  2.6888e-01,
          6.8787e-02, -8.7315e-01, -9.9989e-01, -3.8515e-01,  4.5337e-03,
         -9.9872e-01,  9.9289e-01, -9.8634e-01,  9.9993e-01, -5.9961e-01,
         -4.6896e-01,  7.6944e-01,  2.1443e-02,  8.3105e-02,  6.6032e-02,
          9.9869e-01,  8.0201e-01, -3.4788e-01,  3.8460e-01, -4.1914e-01,
          2.2922e-01, -5.8792e-01,  8.0584e-01,  7.7308e-01,  3.2570e-01,
         -8.6086e-01, -1.5679e-01, -2.2243e-01, -9.8475e-01,  2.7080e-01,
          2.0631e-01,  7.8510e-01,  3.0149e-01,  8.0195e-01,  9.2295e-01,
         -3.0477e-02,  9.2240e-01, -3.7277e-01, -9.9797e-01, -2.9653e-01,
          4.4770e-01, -6.0528e-01,  6.4787e-01,  5.9964e-01,  9.7662e-01,
         -7.5077e-01,  9.9954e-01, -3.3941e-01,  5.4755e-01, -6.1421e-01,
          9.9874e-01, -9.9828e-01,  2.8628e-01, -6.2996e-01,  3.4031e-01,
          8.7544e-01,  9.3187e-01,  5.6419e-02,  5.2092e-01, -3.8520e-01,
          7.3037e-02,  6.7140e-01,  9.4017e-01, -1.4721e-01, -4.1230e-01,
         -1.5066e-01,  9.0205e-01,  9.4544e-01,  4.2607e-01, -2.3067e-01,
         -7.8336e-01, -3.0707e-01,  7.5735e-01, -5.9248e-01,  9.0776e-01,
         -3.6130e-01,  3.5689e-01,  2.0788e-01,  7.5320e-01,  6.7590e-01,
         -9.4824e-02,  2.8336e-01, -9.5376e-01, -7.6047e-01,  6.6129e-01,
          9.9433e-01, -9.2901e-01, -4.7608e-01,  9.8131e-01, -4.6565e-01,
          1.4106e-01,  5.3196e-01,  7.4999e-01, -8.7048e-01, -1.7438e-01,
         -9.9346e-01, -2.7030e-01, -3.3602e-01,  6.3129e-01,  9.2464e-01,
         -1.6718e-03,  3.3027e-01,  3.1175e-01, -9.1250e-01,  8.4549e-01,
          1.6534e-01,  9.1301e-01,  8.4843e-01, -3.3755e-01, -1.2081e-01,
          5.6658e-01,  4.1957e-01, -4.4996e-01,  9.6911e-01, -9.5328e-01,
          9.9828e-01, -2.9308e-01, -9.9729e-01, -3.2739e-01,  8.9235e-01,
         -9.9775e-01, -7.3926e-01, -9.9763e-01,  9.3529e-01, -6.9748e-01,
         -4.3032e-01, -5.4808e-01, -9.9859e-01, -9.9986e-01, -7.0822e-01,
         -2.8977e-01, -1.7731e-01, -4.1994e-01,  9.9277e-01, -9.6092e-02,
          4.8969e-01, -9.0666e-02, -9.4189e-01,  3.1012e-01, -2.5872e-01,
         -2.1360e-02, -9.9889e-01,  2.9212e-01,  8.7661e-01,  3.3856e-02,
          3.4427e-01, -8.7925e-01,  5.9570e-01, -9.4723e-01,  3.3280e-01,
          9.0656e-01,  4.0668e-01,  8.7982e-01, -9.9297e-01,  9.7112e-01,
         -1.9184e-01,  2.7428e-01, -8.3965e-01, -9.5562e-01,  9.9740e-01,
         -8.6528e-02,  1.1721e-01, -4.9984e-01, -9.8889e-01,  3.4941e-01,
         -3.3770e-01, -2.8651e-02, -9.6897e-01,  5.3150e-01, -8.8528e-01,
         -9.9763e-01,  1.7489e-01,  4.4588e-01,  9.9612e-01,  9.0301e-01,
          7.4622e-01, -4.2993e-01, -8.2715e-01,  7.6907e-02, -9.9864e-01,
         -4.2825e-01, -1.2605e-01, -9.0715e-01,  8.2398e-01,  9.9082e-01,
          9.6213e-01, -2.8739e-01, -6.8786e-01, -5.0146e-01,  2.9938e-01,
          8.0460e-01,  7.4203e-01, -7.4438e-01,  2.3635e-01,  2.0104e-02,
         -9.5661e-01, -6.7468e-01,  9.8860e-01, -7.3272e-01,  9.7667e-01,
          5.5601e-01,  5.8993e-01,  5.5981e-01,  1.3676e-01,  1.2681e-01,
         -9.9944e-01, -9.0541e-01,  8.0559e-05, -9.9669e-01,  9.9651e-01,
         -9.9748e-01, -5.2890e-01,  8.6436e-01, -1.3833e-01,  9.5670e-01,
          2.3421e-01, -9.9812e-01, -9.9793e-01, -8.7719e-01, -5.5433e-01,
          9.7533e-01,  2.5896e-01,  2.6540e-01,  8.6108e-01,  7.0904e-01,
          9.9299e-01, -1.6021e-01,  8.2519e-01, -4.7944e-01,  9.9522e-01,
          6.5975e-01, -9.7862e-01,  6.2990e-01, -9.9679e-01,  8.0753e-01,
          8.6054e-01,  9.2572e-01,  8.7751e-01, -6.7182e-01,  9.9702e-01,
         -9.9714e-01,  9.9930e-01, -9.9874e-01, -6.2135e-01,  9.9696e-01,
         -9.5777e-01,  8.0134e-01, -9.9404e-01, -5.3013e-01, -8.8195e-01,
          2.2669e-01, -6.8363e-01,  9.8089e-01, -9.8065e-01, -9.9028e-01,
          5.4673e-01,  8.1745e-01,  4.3122e-01,  8.3944e-01,  6.5311e-01,
          9.8998e-01,  5.2224e-01,  8.6803e-01, -3.9651e-01,  6.1346e-01,
          9.9992e-01, -3.7250e-01,  2.1090e-01, -9.7334e-01,  9.4456e-01,
          8.3059e-01,  4.4086e-01,  8.1598e-01, -4.0159e-01,  5.2830e-01,
          8.3764e-01, -9.8515e-01, -3.5103e-01, -9.8731e-01,  5.3756e-01,
         -7.8816e-01,  5.6718e-01,  2.7801e-01,  4.2508e-01, -3.4629e-02,
         -9.7965e-01,  4.1905e-01, -9.9004e-01,  9.3695e-01, -3.3950e-01,
          3.1758e-01, -1.3659e-01,  8.4417e-01, -5.9518e-01,  9.9160e-01,
          9.9399e-01, -9.9997e-01,  3.9440e-01,  9.6720e-01,  7.4642e-01,
          8.9896e-01, -9.6301e-01, -4.1185e-01,  8.2246e-01, -5.7165e-01,
          9.3119e-01,  7.5216e-01,  2.1331e-03,  8.5057e-01, -9.6894e-01,
          2.7703e-01, -8.1134e-01, -5.0053e-03,  8.7109e-01, -9.5691e-01,
          2.8589e-01,  4.3378e-01,  4.5549e-01, -9.9657e-01, -7.8829e-01,
         -9.9453e-01, -3.0742e-01,  9.7554e-01,  7.8306e-01,  9.9640e-01,
          6.2154e-01, -4.4921e-01, -1.5945e-01, -9.9340e-01, -6.7576e-01,
          2.0973e-01, -2.2138e-01,  4.8150e-01,  5.0749e-01,  4.5111e-01,
         -7.4524e-01, -9.9742e-01,  3.4187e-01,  6.8671e-01, -4.9845e-02,
          8.6144e-01,  9.3861e-01, -8.5106e-01, -6.7663e-01, -8.3774e-01,
          3.5842e-01, -3.3537e-01, -6.7449e-01,  6.1476e-01, -3.7126e-01,
          9.9863e-01, -9.8582e-01, -7.6798e-01, -8.3245e-01, -8.6391e-01,
         -1.9177e-01,  4.3934e-01,  3.2793e-01,  6.5501e-01,  1.5076e-01,
         -6.7812e-01,  9.6209e-01, -9.7668e-01, -9.6157e-01,  9.9213e-01,
         -5.3252e-01, -3.3316e-01, -1.0113e-01, -6.2874e-01, -1.4095e-01,
         -7.5483e-02,  7.0891e-01, -7.8584e-01, -2.9736e-01, -9.9892e-01,
         -8.9136e-01,  1.5772e-01, -9.5436e-01, -8.1500e-01, -3.8603e-01,
         -9.9999e-01,  9.9201e-01,  8.3119e-01,  9.9748e-01, -9.9623e-01,
          7.7365e-01,  4.7285e-01,  9.9309e-01,  5.9435e-02, -7.0350e-01,
         -3.0560e-03,  9.9528e-01,  6.9020e-01, -5.5645e-01, -2.0452e-01,
         -4.0665e-01,  3.5837e-02, -7.6916e-01,  5.2234e-01, -2.8436e-02,
          7.9487e-01, -9.7412e-01, -9.9743e-01,  9.9681e-01,  5.0000e-02,
          9.6435e-01,  1.9433e-01,  4.8093e-01, -8.0393e-01,  6.6415e-01,
         -2.2505e-01, -8.8928e-01, -9.9790e-01,  9.0999e-01, -9.9998e-01,
         -9.7461e-01,  4.3151e-01,  9.7624e-01, -9.8932e-01, -9.5402e-01,
         -6.2787e-02, -9.9762e-01, -3.5696e-01, -7.5082e-01,  3.1802e-01,
         -9.6597e-01,  5.0842e-01, -6.6386e-01,  9.5290e-01,  9.4220e-01,
         -9.2032e-01,  2.8082e-01, -2.9237e-01,  9.9160e-01, -4.5115e-02,
          6.2397e-01,  7.4651e-01, -6.5675e-01, -1.9516e-01, -7.9664e-01,
         -7.9427e-01, -9.7330e-01, -6.8270e-01,  9.6855e-01,  9.5014e-01,
         -9.9663e-01, -9.7722e-01,  7.6703e-01, -1.0422e-01,  9.6612e-01,
         -8.5950e-01, -9.9713e-01, -9.9855e-01, -1.9628e-01, -7.0062e-01,
          9.8119e-01, -5.2540e-02,  9.9994e-01,  7.5654e-01,  4.4310e-01,
         -7.8379e-01,  6.7014e-01,  3.3247e-01,  8.8321e-01, -4.3172e-01,
          9.9881e-01,  8.8534e-01,  9.6925e-01]
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

    encoding2 = encoding2.encoding(primitives2)

    primitives = [not_space, mk_uppercase, mk_lowercase, is_empty, is_space, is_uppercase, not_uppercase, is_lowercase,
                  not_lowercase, is_letter, not_letter, is_number, not_number, skip1, copy1, write1, copyskip1]
    learner = MyOwnLearner(prolog,encoding2)
    decoder = decoding.decoding(primitives)
    covered = [Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, two, hook2a]),
                          List([two])])]),
               Atom(b45, [
                   Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, three,one, hook2a]),
                                 List([three,one])])]),
               Atom(b45, [
                   Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, four,one, hook2a]),
                                 List([four,one])])]),
               Atom(b45, [
                   Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, two,one, hook2a]),
                                 List([two,one])])]),
               Atom(b45, [
                   Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, four, hook2a]),
                                 List([four])])]),

               ]

    learner.assert_knowledge(background)
    head = Atom(b45, [A])
    body1 = Atom(not_number, [A])
    body2 = Atom(write1, [A,C,D])
    body3 = Atom(write1, [B,E, A])
    body4 = Atom(write1, [B,A,E])
    body5 = Atom(write1, [F,G,B])
    body6 = Atom(write1, [C,I,J])
    clause1 = Clause(head, Body(body1, body2, body3, body4, body5, body6))

    print('coverage clause 1 : ', learner.evaluateClause(covered, clause1))
    body1 = Atom(is_lowercase, [A])
    clause2 = Clause(head, Body(body1))
    print('coverage clause 2 : ', learner.evaluateClause(covered, clause2))
    network1 = Network.Network.LoadNetwork('network1.txt')
    network2 = Network.Network.LoadNetwork('network2.txt')

    program = learner.learn(task, background, hs,network1,network2,encodingExamples,decoder,primitives)

    print(program)


