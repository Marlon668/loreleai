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
from MyModel import encoding2,decoding, Network
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

    def evaluateClause(self, examples, clause: Clause):
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
            with open('Search10.csv', 'w') as f:
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
    print('Search for b94')
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
            Structure(s, [List(['d', space2, 'a', space2,'v',space2, 'i', space2, 'd']),
                          List(['d','a','v','i','d'])])]),
        Atom(b45, [
            Structure(s, [List(['j', space2, 'e', space2, 'a', space2, 'n',space2,'n',space2,'e']),
                          List(['j', 'e', 'a', 'n', 'n','e'])])]),

        Atom(b45, [
            Structure(s, [List(['f', space2, 'r', space2, 'e', space2, 'd']),
                          List(['f','r','e','d'])])]),
        Atom(b45, [
            Structure(s, [List(['e',space2,'d',space2,'i',space2,'t',space2,'h']),
                          List(['e','d','i','t','h'])])]),
        Atom(b45, [
            Structure(s, [List(['i',space2,'n',space2,'g',space2,'e']),
                          List(['i','n','g','e'])])])}

    # negative examples
    neg = {Atom(b45, [
        Structure(s, [List(['d',komma2,'e',komma2,'r',komma2,'b',komma2,'y']),
                      List(['d','e','r','b','y'])])]),
            Atom(b45, [Structure(s, [List(
                ['f','r','a','n','k']),
                List([F2, 'r', 'a', 'n','k'])])]),
            Atom(b45, [Structure(s, [List(
                ['w','f','b','o',five,'d','o','e','o','d','n','m',nine,'x']),
                                     List([W2,F2,B2,O2,five,'d','o','e','d','n','m',nine,'x'])])]),
            Atom(b45, [
                Structure(s, [List(
                    [T2,'p','e','w','r',space2,W2,D2,'z','e','t','i',Y2,'V2']),
                              List([T2,P2,E2,W2,R2,'w','d','z','e','t','i','y','v'])])]),
            Atom(b45, [Structure(s, [List(
                [R2,X2,A2]),
                                     List(['r','x','a'])])])}
    task = Task(positive_examples=pos, negative_examples=neg)



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
            Structure(s, [List(['d', space2, 'a', space2, 'v', space2, 'i', space2, 'd']),
                          List(['d', 'a', 'v', 'i', 'd'])])])
        : [-3.7964e-01,  7.2364e-01,  9.9851e-01, -9.9512e-01,  8.2092e-01,
         -1.0483e-01,  9.8296e-01,  6.9717e-01, -9.8622e-01, -7.3434e-01,
          9.7434e-01,  9.9452e-01,  1.0776e-01, -9.9599e-01, -4.1116e-01,
         -9.8244e-01,  9.7909e-01, -2.2576e-01, -9.9908e-01, -1.1802e-01,
          1.1113e-02, -9.9858e-01,  3.5569e-01,  6.0159e-01,  9.7375e-01,
          2.1208e-01,  9.9011e-01,  9.9734e-01,  9.6384e-01,  7.5218e-01,
          3.7606e-01, -9.8749e-01,  9.2184e-02, -9.9787e-01,  2.5171e-01,
          1.0479e-01, -6.8754e-01, -1.0469e-01,  3.1844e-01, -5.9363e-02,
         -9.5432e-01,  8.7871e-01, -8.5728e-01, -5.0742e-01, -5.5257e-01,
          3.3505e-01,  7.3789e-01, -2.3072e-01, -1.8954e-01,  9.9937e-01,
         -9.6406e-01,  1.0000e+00,  5.7182e-01,  9.9991e-01,  9.9542e-01,
          6.3666e-01,  9.8977e-01,  4.2982e-01,  3.3252e-01,  6.4500e-01,
          9.6478e-01, -2.4261e-01,  9.6227e-01, -6.5503e-01, -4.4447e-01,
         -9.0495e-01,  5.1272e-01,  3.1248e-01, -7.6358e-01,  8.5748e-01,
          7.9140e-01,  1.6930e-01,  9.9714e-01, -9.4082e-01,  2.1529e-01,
         -9.1643e-01,  2.5647e-01, -9.9896e-01,  9.8121e-01,  9.9876e-01,
         -4.3664e-01, -9.9384e-01,  9.9560e-01,  4.8949e-02,  5.7084e-01,
         -9.2499e-01,  7.5034e-01, -9.9357e-01,  3.7568e-01, -8.1998e-01,
         -4.1683e-01, -9.8885e-01, -2.7595e-01,  6.9303e-01,  9.9871e-01,
          4.3721e-01, -7.9128e-02, -2.7053e-02,  6.4466e-01,  3.3283e-01,
         -8.7112e-01,  8.5406e-02, -6.8360e-01,  8.4611e-01, -1.8557e-01,
         -3.8694e-01, -9.4056e-01, -1.0730e-01, -4.9980e-01, -2.4406e-02,
          9.9576e-01, -9.8524e-01, -6.3049e-01,  1.5018e-02, -4.7559e-01,
          3.8498e-01,  9.9405e-01, -8.8892e-02, -2.8884e-01,  9.9729e-01,
          2.8369e-01, -2.6927e-01,  9.9358e-01, -6.3461e-01,  8.7161e-01,
         -3.8830e-01,  1.8060e-01,  4.7157e-02,  4.5270e-01, -7.3486e-01,
          5.7269e-01, -9.8425e-01,  5.7359e-01,  9.9583e-01, -3.1755e-02,
          9.9858e-01, -9.9717e-01,  1.3902e-01, -9.9924e-01,  4.6287e-02,
          7.2168e-01, -1.3158e-01,  2.0140e-01,  5.3984e-01,  9.9414e-01,
          3.6348e-01, -2.9245e-02,  5.8541e-01, -4.3565e-01,  5.2924e-01,
          8.6645e-01,  9.5857e-01, -9.7266e-01,  9.9999e-01, -3.8468e-01,
          2.1918e-01,  1.2374e-01,  2.9760e-01,  2.6988e-01, -4.5867e-01,
          9.9133e-01, -9.9526e-01,  2.3377e-01,  5.2942e-01,  9.9636e-01,
          9.8409e-01, -1.9342e-01, -1.2745e-01,  9.9598e-01,  8.5229e-01,
          5.3366e-01, -6.3078e-01, -7.8523e-01,  4.1210e-01,  1.1539e-01,
          4.1936e-01,  9.4242e-01,  9.9477e-01, -9.8475e-01,  9.8408e-01,
          9.9987e-01, -6.0450e-01,  9.2047e-01, -3.5916e-01, -9.9573e-01,
         -9.9705e-01, -9.8335e-01,  5.5608e-01, -5.9893e-01,  4.3887e-02,
         -5.2290e-01,  9.4169e-01, -7.4117e-01,  8.8669e-01, -9.7768e-01,
          2.0704e-01,  9.8328e-01, -2.4677e-01,  9.9802e-01,  3.8014e-01,
         -9.9687e-01, -1.4394e-02, -6.0568e-01,  9.9048e-01, -3.6705e-01,
          9.8044e-01,  4.4032e-01,  4.0794e-01,  7.3892e-01, -1.0000e+00,
         -8.0138e-01, -2.8270e-01, -8.3666e-01,  7.3902e-01,  9.9581e-01,
          5.4354e-01, -2.3722e-01, -1.7350e-01,  3.3915e-01,  9.9893e-01,
         -9.9550e-01, -4.9526e-01,  6.8056e-01, -9.9472e-01, -9.9704e-01,
          9.7754e-01, -1.4107e-01,  7.7573e-01, -4.6025e-01, -9.5549e-01,
         -1.6088e-02,  3.8856e-02,  9.8576e-01,  1.6899e-01,  8.4316e-01,
         -9.9829e-01,  8.8159e-01, -9.4192e-01,  3.8782e-01,  2.7937e-01,
          5.9797e-01, -4.4509e-01, -9.5267e-01,  5.3938e-01,  9.3945e-01,
         -3.4189e-01, -1.9306e-01, -5.9307e-01,  8.8791e-01,  8.1799e-01,
         -8.2613e-01, -3.2034e-02, -9.9027e-01,  9.9080e-01,  7.0699e-01,
         -8.8076e-02,  9.9338e-01, -9.8612e-01,  5.6985e-02,  5.5760e-01,
         -1.0602e-01, -9.9999e-01, -2.0396e-01, -8.8654e-01,  8.2445e-01,
         -1.9052e-01,  9.9299e-01, -9.7613e-01, -4.3213e-01, -5.6813e-01,
         -9.9812e-01,  9.1908e-01, -2.6394e-01,  9.9250e-01, -9.3559e-01,
         -1.6676e-01,  9.9570e-01,  4.9463e-01, -9.9296e-01, -9.9827e-01,
          4.5904e-02,  9.9995e-01, -9.8308e-01, -2.8542e-01,  9.9901e-01,
          7.7689e-01, -9.4945e-01, -9.2005e-01, -9.9917e-01, -9.9521e-01,
         -7.1391e-01,  4.3780e-01, -6.0136e-02,  9.8245e-01, -3.7744e-01,
         -1.4962e-02,  9.9307e-01,  9.9995e-01, -8.7057e-02,  9.5699e-02,
          1.1221e-01, -9.6129e-01, -9.9999e-01,  3.0738e-01,  1.1711e-01,
         -9.9893e-01,  9.9554e-01, -9.9235e-01,  9.9999e-01, -8.5749e-01,
          5.6429e-01,  8.6041e-01, -1.7673e-01,  1.7295e-01,  4.4177e-02,
          9.9865e-01,  9.8890e-01, -1.6489e-01,  1.1342e-01, -3.8300e-01,
          9.7569e-02, -6.3262e-01,  7.4402e-01,  4.2507e-01, -1.8103e-01,
         -7.7881e-01, -4.8575e-01, -4.2135e-01, -9.9404e-01, -1.6187e-01,
          3.9464e-01,  9.3777e-01, -2.4267e-01,  9.4859e-01,  9.9112e-01,
          1.7745e-01,  5.4813e-01, -7.5561e-01, -9.9847e-01,  4.8516e-01,
          5.2153e-01, -3.6497e-01,  6.3584e-01, -3.9545e-01,  9.9488e-01,
         -9.9268e-01,  9.9991e-01, -2.5587e-01,  2.3924e-01,  5.9299e-01,
          9.9838e-01, -9.9984e-01,  3.9001e-01, -2.9898e-01,  4.1864e-01,
          7.8057e-01,  9.9303e-01, -2.7504e-01,  4.5219e-01, -5.9883e-01,
         -2.9596e-01,  9.2847e-01,  9.8237e-01,  2.7924e-01, -1.7663e-01,
          9.0735e-01,  8.8833e-01,  9.7185e-01, -5.2468e-01, -6.2253e-02,
         -7.8653e-01,  4.0546e-01,  9.7759e-01, -4.7088e-02, -6.6821e-01,
         -3.1667e-01,  1.1420e-01, -5.1208e-01, -6.2376e-01,  7.6246e-01,
          2.1530e-01,  3.7935e-01, -9.9489e-01, -1.0802e-02,  5.1035e-01,
          9.9632e-01, -9.9324e-01, -4.2176e-01,  9.9512e-01, -4.8428e-01,
          2.1425e-01,  8.3895e-01, -1.6803e-01, -9.7506e-01, -1.0616e-01,
         -9.9887e-01, -2.6737e-01, -3.1503e-01,  5.2688e-01,  6.3056e-01,
         -6.3778e-03,  4.9718e-01, -1.4459e-01, -5.8788e-01,  9.0497e-01,
          6.0283e-01,  9.9193e-01,  6.1992e-01, -5.9592e-01, -2.1659e-01,
          5.5071e-01,  3.8840e-01,  5.1947e-02,  9.8880e-01, -9.9330e-01,
          9.9942e-01, -5.6936e-01, -9.9809e-01,  8.7168e-01,  9.9387e-02,
         -9.9733e-01, -3.5499e-01, -9.9959e-01,  9.9150e-01, -6.1149e-01,
          7.1002e-01,  7.4591e-01, -9.9987e-01, -9.9999e-01, -5.7465e-01,
         -6.1994e-01, -2.4785e-01,  1.3614e-04,  9.9732e-01, -1.5854e-01,
          5.3758e-01,  1.8535e-01, -9.2008e-01,  1.3300e-01,  8.5625e-01,
         -9.1493e-02, -9.9905e-01,  5.8491e-01,  3.1285e-01,  6.0210e-01,
          3.7598e-01, -9.2268e-01,  7.8980e-01, -9.4888e-01,  6.7918e-01,
          9.6679e-01,  7.1766e-01,  4.1577e-01, -9.9091e-01,  9.9053e-01,
          5.3904e-01,  1.5605e-01, -6.8620e-01, -9.8875e-01,  9.9847e-01,
         -4.1672e-01, -5.1322e-02, -1.8934e-01, -9.9167e-01, -1.7118e-01,
          7.2816e-01, -3.9094e-01, -9.9571e-01,  3.5789e-01, -9.9310e-01,
         -9.9947e-01,  3.4170e-01, -6.3557e-01,  9.9968e-01,  9.3708e-01,
          5.8092e-02,  4.6388e-02, -9.5247e-01,  8.2342e-02, -9.9940e-01,
         -1.5495e-01,  5.8157e-02, -9.9275e-01,  8.3920e-01,  9.7686e-01,
          9.9643e-01, -1.4264e-01,  1.0647e-01, -6.6672e-01,  5.7670e-01,
          9.5672e-01,  8.4321e-01, -3.1321e-01, -8.2899e-02, -1.3293e-01,
         -9.9335e-01, -9.0429e-01,  9.9470e-01,  6.7745e-01,  9.9448e-01,
         -7.3938e-01, -1.1431e-01, -5.9234e-02, -2.7471e-01,  3.9689e-01,
         -9.9997e-01, -8.6938e-01,  3.3880e-01, -9.9887e-01,  9.9868e-01,
         -9.9892e-01, -6.5395e-01,  5.6463e-01, -9.2767e-02,  9.8290e-01,
          4.8830e-01, -9.9859e-01, -9.9802e-01, -9.1612e-01,  3.9332e-02,
          9.9337e-01, -2.0837e-01,  1.7731e-01,  5.1846e-01,  7.6428e-01,
          9.9852e-01,  5.0968e-01,  8.5384e-01,  7.1997e-01,  9.9623e-01,
          3.6213e-01, -9.9693e-01, -3.0472e-02, -9.9713e-01,  3.4520e-01,
          9.3811e-01,  9.7650e-01,  9.8453e-01,  7.2118e-01,  9.9834e-01,
         -9.9932e-01,  9.9989e-01, -9.9764e-01,  6.4058e-01,  9.9814e-01,
         -9.7722e-01,  2.7812e-01, -9.9153e-01,  3.8222e-01, -8.3562e-01,
          1.5387e-01, -7.7907e-01,  9.9524e-01, -9.9569e-01, -9.9760e-01,
          5.2560e-01,  5.3337e-01,  9.0474e-01, -2.9120e-01,  2.0395e-01,
          9.9801e-01,  1.0530e-01,  9.8106e-01, -2.8124e-01,  1.7988e-01,
          1.0000e+00, -2.5546e-01, -1.9474e-02, -9.9171e-01,  9.8943e-01,
          2.8467e-01,  1.3628e-01,  9.4296e-01, -3.5139e-01, -2.0157e-02,
          7.7597e-01, -9.9040e-01, -4.4210e-01, -9.8253e-01,  2.6717e-01,
         -7.3516e-01,  4.0295e-01,  2.7317e-01, -3.4398e-02,  1.3901e-01,
         -9.9028e-01,  1.5272e-01, -9.9540e-01,  9.7458e-01,  2.7994e-01,
          4.2439e-02,  1.7534e-01,  5.8113e-01, -8.9049e-01,  9.9125e-01,
          9.9620e-01, -1.0000e+00, -1.7293e-01,  9.9091e-01,  6.0601e-01,
          9.9231e-01, -9.8774e-01, -3.8787e-01,  7.5471e-01, -8.1193e-01,
          9.7987e-01,  7.6333e-01, -4.2850e-01,  9.7939e-01, -9.9203e-01,
          3.2681e-01, -7.9856e-01, -1.7907e-01,  9.0443e-01, -9.7466e-01,
          1.5783e-01, -1.0531e-01, -1.3061e-01, -9.9730e-01, -7.0145e-01,
         -9.9708e-01, -2.7534e-01,  9.9517e-01,  2.5752e-01,  9.9888e-01,
          8.0892e-01, -4.8909e-01,  2.1062e-01, -9.9717e-01,  5.2940e-01,
          6.3425e-02,  1.1494e-01,  5.6588e-01, -8.0266e-01,  3.1096e-02,
         -9.1816e-01, -9.9722e-01,  8.9950e-02, -3.6546e-02,  6.6327e-02,
          8.8609e-01,  6.0867e-01, -9.9162e-01, -6.5094e-01, -8.7423e-01,
          4.7569e-01, -7.0518e-01, -9.8848e-01,  9.6538e-01, -5.1900e-01,
          9.9906e-01, -9.9318e-01, -9.3440e-01, -9.6543e-01, -4.2094e-01,
         -9.2096e-02,  3.7672e-01,  5.3270e-01,  5.5530e-02, -4.2893e-01,
         -6.7509e-01,  9.7869e-01, -9.9218e-01, -9.9462e-01,  9.9420e-01,
         -8.5660e-01,  3.0993e-01, -2.3019e-01, -5.4510e-01, -3.5273e-01,
         -4.1465e-01,  2.8991e-01, -9.3237e-01, -5.3429e-01, -9.9996e-01,
         -8.7728e-01,  6.2717e-01, -9.9749e-01, -8.5821e-01, -4.8706e-01,
         -1.0000e+00,  9.9425e-01,  9.4977e-01,  9.9802e-01, -9.9763e-01,
          7.3891e-01,  2.7439e-01,  9.9593e-01, -2.2535e-01, -4.0760e-01,
          2.6068e-01,  9.9698e-01,  7.3983e-01, -8.8372e-01,  1.3745e-01,
         -2.8901e-01,  1.3576e-01, -7.4638e-01, -5.7669e-01,  5.0502e-01,
          7.8989e-01, -9.8944e-01, -9.9720e-01,  9.9816e-01, -1.7935e-01,
          9.7004e-01,  4.8636e-01, -3.4460e-01, -9.3761e-01,  5.1433e-01,
          3.2360e-01, -8.8376e-01, -9.9940e-01,  5.8429e-01, -1.0000e+00,
         -9.9408e-01,  4.6305e-01,  9.8181e-01, -9.9074e-01, -9.9071e-01,
          3.4700e-03, -9.9895e-01, -6.5213e-01, -5.6536e-02,  4.0507e-01,
         -9.8981e-01, -6.1397e-01, -5.3615e-01,  8.3208e-01,  9.8340e-01,
         -9.8542e-01,  2.7404e-01, -9.2185e-01,  9.9602e-01, -9.1982e-02,
          3.4339e-01,  3.4210e-01,  1.8727e-01,  3.1275e-01, -9.7757e-01,
         -8.4375e-01, -9.9436e-01, -9.4982e-01,  9.9126e-01,  9.8750e-01,
         -9.9738e-01, -9.8852e-01, -1.0549e-01, -9.5537e-02,  9.9345e-01,
         -6.1161e-01, -9.9834e-01, -9.9833e-01,  2.0625e-01, -6.7864e-01,
          9.9673e-01, -4.8022e-01,  9.9999e-01,  8.3501e-01,  5.0747e-01,
         -5.5245e-01,  5.7219e-01,  5.7449e-01,  6.8067e-01,  2.6156e-02,
          9.9890e-01,  4.4760e-01,  9.9609e-01],
        Atom(b45, [
            Structure(s, [List(['j', space2, 'e', space2, 'a', space2, 'n', space2, 'n', space2, 'e']),
                          List(['j', 'e', 'a', 'n', 'n', 'e'])])])
        : [-0.3177,  0.5921,  0.9976, -0.9928,  0.7902,  0.2770,  0.9762, -0.4139,
         -0.9272, -0.7432,  0.9831,  0.9943, -0.8170, -0.9942, -0.1698, -0.9777,
          0.9827, -0.1400, -0.9986, -0.3193, -0.1930, -0.9982,  0.1486,  0.9485,
          0.9451,  0.2478,  0.9889,  0.9978,  0.9018,  0.6745, -0.0256, -0.9897,
          0.6674, -0.9968, -0.0282, -0.4669, -0.1054, -0.2604,  0.6359, -0.6665,
         -0.8364,  0.7081, -0.3633, -0.2589,  0.4340,  0.1859,  0.5406, -0.1057,
         -0.0369,  0.9989, -0.9160,  1.0000, -0.3897,  0.9991,  0.9950,  0.8508,
          0.9927,  0.2566, -0.7820,  0.8470,  0.9646, -0.3137,  0.8978, -0.5631,
         -0.4358, -0.7133, -0.0399, -0.2066, -0.4498,  0.8316,  0.5722,  0.1236,
          0.9953, -0.9230,  0.4218, -0.9631,  0.3954, -0.9990,  0.9820,  0.9977,
          0.0020, -0.9954,  0.9968,  0.1174,  0.1598, -0.7159, -0.6769, -0.9941,
          0.4211, -0.7506,  0.1828, -0.9885,  0.4061,  0.0435,  0.9981, -0.1520,
         -0.0878, -0.1952,  0.7741, -0.2438, -0.8507,  0.3629,  0.6768, -0.1035,
          0.6693, -0.2657, -0.9380, -0.8078, -0.5976, -0.2884,  0.9915, -0.9211,
         -0.9063, -0.1371,  0.0733, -0.7956,  0.9933,  0.8943, -0.2370,  0.9984,
          0.0514,  0.7354,  0.9964, -0.1633,  0.3651, -0.1806, -0.1618,  0.4646,
          0.0501, -0.8246,  0.7364, -0.9908, -0.5017,  0.9954,  0.0239,  0.9986,
         -0.9972,  0.8519, -0.9994, -0.1441,  0.7088, -0.3156, -0.5541,  0.8302,
          0.9891,  0.1507, -0.7850,  0.4548, -0.0783, -0.1303,  0.7676,  0.9436,
         -0.9333,  0.9998,  0.8802,  0.6659,  0.8213, -0.0178, -0.5763, -0.0109,
          0.9773, -0.9962, -0.2022, -0.4709,  0.9954,  0.9911,  0.0666, -0.8249,
          0.9969,  0.6457,  0.0799, -0.8758, -0.7266, -0.7629,  0.1449,  0.2039,
          0.8849,  0.9874, -0.9876,  0.9858,  0.9968, -0.5255,  0.9284,  0.7558,
         -0.9934, -0.9898, -0.9897,  0.5345, -0.3221, -0.2733, -0.0431,  0.9640,
          0.2993,  0.7510, -0.9606,  0.2486,  0.9805, -0.1387,  0.9985,  0.8184,
         -0.9959, -0.1207, -0.1401,  0.9793, -0.0943,  0.9676,  0.0526,  0.2387,
          0.9829, -0.9999,  0.3741, -0.2546, -0.6492,  0.6079,  0.9874, -0.4999,
         -0.1459, -0.2558, -0.1007,  0.9991, -0.9929, -0.2134,  0.5308, -0.9954,
         -0.9956,  0.9837, -0.2066,  0.6115, -0.1740, -0.8130, -0.0822,  0.6266,
          0.9933,  0.1081,  0.5302, -0.9988, -0.3153, -0.9441, -0.6564,  0.1106,
          0.4131, -0.4646, -0.9281, -0.5442,  0.9535,  0.2323, -0.5571, -0.0544,
          0.7594, -0.5756, -0.3856, -0.3919, -0.9875,  0.9949,  0.4228,  0.7990,
          0.9909, -0.9821,  0.6595, -0.8104, -0.1731, -0.9999,  0.0806, -0.7573,
          0.2952, -0.0883,  0.9943, -0.9281, -0.7716,  0.5376, -0.9972,  0.9289,
         -0.1045,  0.9901, -0.4537,  0.3840,  0.9940,  0.8635, -0.9944, -0.9982,
          0.8544,  0.9968, -0.9738, -0.1854,  0.9986, -0.6375, -0.9323, -0.9456,
         -0.9980, -0.9971, -0.3645, -0.5002, -0.2462,  0.9697,  0.0015, -0.0990,
          0.9883,  0.9999,  0.1026, -0.2937,  0.2358, -0.9379, -0.9995,  0.7615,
         -0.1997, -0.9989,  0.9954, -0.9894,  0.9999, -0.3010, -0.6082,  0.6751,
          0.1238, -0.5330,  0.0948,  0.9988,  0.9645, -0.4357, -0.1260,  0.3743,
          0.1779, -0.2509,  0.4474,  0.1024, -0.3817, -0.6304,  0.5345,  0.3684,
         -0.9918,  0.8041,  0.2170,  0.9424, -0.7479,  0.9235,  0.9851,  0.1133,
          0.6935,  0.0256, -0.9748, -0.5351,  0.0858, -0.9324,  0.3101,  0.8363,
          0.9919, -0.9786,  0.9996, -0.1076,  0.7033, -0.5099,  0.9987, -0.9994,
         -0.0096,  0.3779, -0.3009,  0.4152,  0.9939,  0.8107,  0.9406, -0.3902,
         -0.4694,  0.7447,  0.9782, -0.6032, -0.2613, -0.4003,  0.7549,  0.9788,
          0.1835,  0.0867, -0.4312, -0.7814,  0.9583, -0.6987, -0.3356, -0.1742,
         -0.4389,  0.0900,  0.7269,  0.1123,  0.7002,  0.1793, -0.9932,  0.6491,
          0.8968,  0.9973, -0.9942, -0.2268,  0.9926, -0.2522, -0.4057,  0.7699,
          0.8993, -0.9801, -0.2284, -0.9987,  0.0118, -0.5044,  0.6255,  0.5479,
         -0.0234, -0.4712,  0.6615, -0.6409,  0.8989,  0.4624,  0.9728,  0.3786,
         -0.4180,  0.0213,  0.5335,  0.1973,  0.0116,  0.9882, -0.9842,  0.9982,
          0.2829, -0.9980, -0.2461, -0.1431, -0.9977,  0.1174, -0.9989,  0.9890,
         -0.4032, -0.6988, -0.6427, -0.9995, -0.9999,  0.0538,  0.0422, -0.1108,
         -0.6408,  0.9522, -0.2245,  0.3133,  0.2649, -0.9415,  0.0150, -0.3883,
          0.2044, -0.9990,  0.0613,  0.9277, -0.7287, -0.5981, -0.9611,  0.8029,
         -0.8908,  0.6671,  0.9548,  0.5323,  0.2515, -0.9889,  0.9735, -0.6667,
          0.1481, -0.9171, -0.9728,  0.9985, -0.3053,  0.0102, -0.0738, -0.9946,
          0.6324, -0.4888, -0.9112, -0.9933,  0.2130, -0.9899, -0.9993,  0.3218,
          0.5652,  0.9991,  0.9495,  0.3794,  0.0497, -0.8789, -0.2057, -0.9993,
          0.1627,  0.7839, -0.9914,  0.5431,  0.9834,  0.9932, -0.8710, -0.8459,
         -0.3349,  0.6158,  0.8996,  0.5213, -0.4740, -0.2688, -0.2570, -0.9905,
         -0.9551,  0.9962, -0.7264,  0.9850,  0.3860,  0.7853, -0.1101,  0.1617,
         -0.5699, -0.9994, -0.8089,  0.0240, -0.9989,  0.9979, -0.9977, -0.1140,
          0.2851,  0.2032,  0.9720,  0.2585, -0.9988, -0.9988, -0.7990, -0.0125,
          0.9917,  0.2360,  0.3427,  0.4796, -0.0495,  0.9979, -0.2758,  0.5066,
         -0.6412,  0.9977, -0.0681, -0.9949,  0.9092, -0.9984,  0.9370,  0.9598,
          0.9860,  0.9700, -0.7068,  0.9984, -0.9992,  0.9994, -0.9977, -0.6386,
          0.9983, -0.9713,  0.0088, -0.9920, -0.8416, -0.0334, -0.0942, -0.6045,
          0.9951, -0.9955, -0.9977,  0.6090,  0.0749,  0.4485,  0.7915, -0.0184,
          0.9952, -0.0626,  0.9764, -0.1423,  0.9217,  0.9999, -0.6586,  0.3747,
         -0.9908,  0.9886, -0.3471,  0.1691,  0.9007, -0.3236,  0.0351,  0.8481,
         -0.9950,  0.0807, -0.9416,  0.8488, -0.2553,  0.4495,  0.0963, -0.2875,
          0.0618, -0.9911,  0.1869, -0.9956,  0.9834, -0.2589, -0.1061, -0.1404,
          0.8647, -0.7531,  0.9941,  0.9987, -0.9999,  0.0848,  0.9866,  0.3933,
          0.9826, -0.9879, -0.0376,  0.9621, -0.3832,  0.9794,  0.6868,  0.0127,
          0.9748, -0.9907, -0.5014, -0.8397,  0.1560,  0.8450, -0.9721, -0.0402,
          0.7180,  0.0956, -0.9979, -0.2843, -0.9971,  0.0233,  0.9970,  0.7677,
          0.9984,  0.5266, -0.4579,  0.2780, -0.9970, -0.7588,  0.2535,  0.1290,
         -0.2390,  0.7372, -0.1434, -0.0484, -0.9936,  0.1509,  0.8006,  0.0988,
          0.7578,  0.0760, -0.9785, -0.9555, -0.5762,  0.2278, -0.2600, -0.9493,
          0.8528, -0.8314,  0.9987, -0.9923, -0.8836, -0.9716, -0.1499,  0.7873,
          0.3507,  0.4020, -0.2869,  0.6203, -0.8754,  0.9845, -0.9931, -0.9944,
          0.9961, -0.2887, -0.7885,  0.0239, -0.4748, -0.4481, -0.0187,  0.8898,
         -0.8872, -0.2577, -0.9997, -0.7826, -0.0802, -0.9949, -0.9024, -0.3278,
         -0.9999,  0.9957,  0.9597,  0.9973, -0.9988,  0.8786,  0.3368,  0.9961,
         -0.1509, -0.5333,  0.7150,  0.9971, -0.0379, -0.2873, -0.2508, -0.0412,
          0.3389, -0.8467,  0.5189, -0.2762,  0.6336, -0.9928, -0.9973,  0.9973,
          0.1384,  0.9858,  0.2161,  0.7270, -0.9348,  0.9419, -0.2757, -0.8357,
         -0.9991,  0.5482, -1.0000, -0.9900,  0.2565,  0.9890, -0.9941, -0.9619,
         -0.1912, -0.9983,  0.2129, -0.7842, -0.0858, -0.9953,  0.7526, -0.3962,
         -0.0308,  0.9827, -0.9723,  0.8882, -0.5453,  0.9658, -0.3127,  0.2519,
         -0.1583, -0.7139, -0.3671, -0.9375,  0.2442, -0.9883, -0.9656,  0.9936,
          0.9875, -0.9964, -0.9867,  0.7528, -0.1320,  0.9936, -0.6759, -0.9986,
         -0.9990, -0.0854, -0.5068,  0.9962, -0.2003,  0.9995,  0.8862,  0.5481,
         -0.1298,  0.0533,  0.0241,  0.2643, -0.0318,  0.9981, -0.3864,  0.9946],
        Atom(b45, [
            Structure(s, [List(['f', space2, 'r', space2, 'e', space2, 'd']),
                          List(['f', 'r', 'e', 'd'])])])
        : [-5.7287e-01,  7.5739e-01,  9.9895e-01, -9.9641e-01,  6.4006e-01,
          8.0322e-02,  9.8618e-01, -3.6476e-01, -9.9267e-01, -7.9728e-01,
          9.8752e-01,  9.9833e-01, -8.5380e-01, -9.9716e-01, -4.5277e-02,
         -9.7430e-01,  9.8984e-01, -2.7724e-01, -9.9972e-01, -8.5209e-01,
          1.2231e-01, -9.9934e-01,  3.5135e-01,  9.2994e-01,  9.5854e-01,
          1.8266e-01,  9.9288e-01,  9.9904e-01,  9.6591e-01,  5.9717e-01,
          3.4524e-01, -9.9615e-01,  3.4646e-01, -9.9862e-01,  1.4431e-01,
         -2.8768e-01, -4.1043e-01, -1.3948e-02,  7.9449e-01, -3.5277e-01,
         -9.4441e-01,  5.9126e-01, -6.6250e-01, -3.5904e-01,  3.1075e-02,
          7.1258e-01,  5.6953e-01, -5.3520e-02, -1.4333e-01,  9.9912e-01,
         -9.7388e-01,  9.9999e-01, -4.0613e-01,  9.9985e-01,  9.9620e-01,
          7.5688e-01,  9.9677e-01,  3.9427e-01, -7.0130e-01,  7.3467e-01,
          9.9123e-01, -4.5379e-01,  9.7458e-01, -4.7104e-01, -8.2287e-02,
         -8.3909e-01, -1.9470e-01,  3.2105e-01, -5.6593e-01,  7.5407e-01,
          8.3772e-01,  2.7831e-01,  9.9797e-01, -9.6212e-01,  3.4216e-01,
         -9.8122e-01,  5.9940e-01, -9.9972e-01,  9.8034e-01,  9.9942e-01,
         -1.2636e-01, -9.9804e-01,  9.9695e-01, -9.3016e-03,  4.3297e-01,
         -7.3476e-01, -5.7754e-01, -9.9749e-01,  2.9756e-01, -9.6280e-01,
          9.6601e-03, -9.9604e-01,  2.1804e-01,  3.6728e-01,  9.9945e-01,
          2.7162e-04,  2.2851e-03,  3.6156e-02,  8.1116e-01, -6.6144e-01,
         -8.7862e-01,  6.2508e-01,  5.0199e-01, -1.2316e-01,  8.5189e-01,
          9.5603e-02, -9.3677e-01, -7.7269e-01,  5.7928e-02, -9.0513e-02,
          9.9708e-01, -9.7208e-01, -9.2998e-01, -1.6182e-01, -8.8247e-02,
         -7.3181e-01,  9.9566e-01,  7.8856e-01, -3.3990e-01,  9.9924e-01,
          5.2508e-02,  5.4389e-01,  9.9788e-01, -3.7657e-01,  2.3933e-01,
         -1.2373e-01, -5.1382e-01,  8.5528e-01, -2.2407e-01, -8.7180e-01,
          7.0609e-01, -9.9265e-01, -2.0023e-01,  9.9750e-01, -5.6260e-02,
          9.9945e-01, -9.9855e-01,  8.6420e-01, -9.9971e-01, -7.2682e-01,
          3.1039e-02, -4.1367e-01, -6.2531e-01,  8.4157e-01,  9.9734e-01,
          3.1826e-01, -4.6973e-01,  1.8564e-01, -1.0481e-01, -2.1318e-01,
          5.6050e-01,  9.6083e-01, -9.7757e-01,  9.9993e-01,  8.9780e-01,
          4.6610e-01,  8.2627e-01,  2.6198e-02, -7.0824e-01,  1.9406e-01,
          9.9217e-01, -9.9769e-01,  4.7748e-01, -5.5753e-01,  9.9757e-01,
          9.9300e-01,  4.7970e-01, -6.7105e-01,  9.9879e-01,  4.0798e-01,
          2.0848e-01, -8.5367e-01, -6.6637e-01, -7.0324e-01,  1.3761e-01,
          3.5776e-01,  9.6567e-01,  9.9322e-01, -9.9287e-01,  9.9035e-01,
          9.9836e-01, -3.5854e-01,  9.2506e-01,  6.9383e-01, -9.9868e-01,
         -9.9637e-01, -9.9404e-01,  4.6454e-01, -2.7284e-01,  4.2088e-01,
         -3.1088e-01,  9.5618e-01,  3.1386e-01,  9.0015e-01, -9.6779e-01,
          2.6756e-01,  9.8531e-01,  8.9551e-02,  9.9911e-01,  4.8388e-01,
         -9.9898e-01, -5.4047e-01, -1.1897e-01,  9.8717e-01, -3.0699e-01,
          9.8890e-01,  1.3151e-01,  9.7149e-02,  9.7124e-01, -9.9997e-01,
          3.0325e-01, -1.0962e-01, -5.2463e-01,  7.6984e-01,  9.9533e-01,
         -6.3056e-01, -1.5922e-01, -1.0145e-01, -1.8244e-01,  9.9941e-01,
         -9.9659e-01, -5.1907e-01,  8.4806e-01, -9.9691e-01, -9.9868e-01,
          9.8601e-01, -2.3313e-01,  2.5771e-01, -3.3083e-01, -6.9950e-01,
         -4.3505e-02,  6.4994e-01,  9.9299e-01, -6.5455e-02,  5.1652e-01,
         -9.9950e-01,  7.5463e-02, -9.8353e-01, -3.6219e-01,  2.4187e-01,
          5.5757e-01, -4.7904e-01, -9.9191e-01, -4.5602e-01,  9.4354e-01,
          5.9411e-01, -5.1166e-01, -4.3565e-01,  4.2897e-01, -3.1210e-01,
          9.2330e-02, -7.3310e-01, -9.8692e-01,  9.9773e-01,  3.4768e-01,
          6.5125e-01,  9.8964e-01, -9.9452e-01,  8.4418e-01, -6.4453e-01,
         -1.6469e-01, -9.9995e-01,  9.2274e-02, -5.4135e-01,  5.9414e-01,
         -2.5366e-02,  9.9696e-01, -9.8597e-01, -3.1343e-01,  5.4415e-01,
         -9.9875e-01,  9.5250e-01, -2.4982e-01,  9.9733e-01, -4.4775e-01,
          4.2177e-01,  9.9743e-01,  9.3312e-01, -9.9180e-01, -9.9936e-01,
          8.6309e-01,  9.9781e-01, -9.9218e-01, -1.2785e-01,  9.9944e-01,
         -4.1676e-01, -9.4722e-01, -9.6338e-01, -9.9891e-01, -9.9830e-01,
         -6.1161e-01, -4.7164e-01,  2.6569e-02,  9.8713e-01,  4.0287e-02,
         -1.4884e-02,  9.9290e-01,  9.9992e-01, -9.2218e-02, -1.5455e-01,
          5.7318e-02, -9.5301e-01, -9.9958e-01,  8.9052e-01, -5.1118e-03,
         -9.9935e-01,  9.9781e-01, -9.9326e-01,  9.9996e-01, -7.2158e-01,
         -4.9930e-01,  8.8545e-01, -1.8363e-02, -5.3757e-01,  2.8230e-01,
          9.9939e-01,  9.9099e-01, -2.4492e-01,  1.6800e-01, -4.2393e-02,
          1.1265e-01, -2.3466e-01,  4.9564e-01,  6.0451e-01, -1.3360e-01,
         -7.6295e-01,  1.3917e-01,  3.7966e-01, -9.9426e-01,  7.6093e-01,
          3.6365e-01,  9.5792e-01, -8.5236e-01,  9.6263e-01,  9.8949e-01,
          1.6859e-01,  5.5143e-01, -1.3127e-02, -9.8227e-01, -6.8482e-01,
          3.7134e-01, -9.1751e-01,  3.9532e-01,  6.2731e-01,  9.9423e-01,
         -9.9164e-01,  9.9984e-01, -1.2428e-01,  7.4727e-01, -5.8044e-01,
          9.9952e-01, -9.9989e-01, -9.7348e-03,  3.4197e-01,  3.4065e-01,
          6.1521e-01,  9.9399e-01,  7.1935e-01,  9.3182e-01, -7.1439e-01,
         -5.3723e-01,  9.6461e-01,  9.8407e-01, -4.1176e-01, -3.6685e-01,
         -1.0541e-01,  7.2732e-01,  9.8014e-01,  5.1152e-01, -4.4836e-02,
         -7.0485e-01, -7.6462e-01,  9.8643e-01, -6.8405e-01, -7.1125e-01,
         -2.2168e-01, -5.2093e-01,  5.2039e-01,  5.7765e-01,  6.0053e-01,
          5.8123e-01,  2.7038e-01, -9.9536e-01,  5.6421e-01,  9.0019e-01,
          9.9859e-01, -9.9604e-01, -3.1242e-01,  9.9619e-01, -1.2562e-01,
         -2.7117e-01,  7.7109e-01,  8.9724e-01, -9.9329e-01, -2.5781e-01,
         -9.9926e-01, -3.6534e-01, -5.2908e-01,  5.4416e-01,  3.1994e-01,
         -3.7667e-02, -1.9373e-01,  5.7817e-01, -6.7218e-01,  8.1985e-01,
          1.1945e-01,  9.8394e-01,  3.0604e-03, -5.1643e-01, -2.8859e-01,
          6.9489e-01,  3.5370e-01,  1.9653e-01,  9.9344e-01, -9.9166e-01,
          9.9927e-01,  4.1158e-01, -9.9883e-01,  2.6192e-01, -1.7792e-01,
         -9.9800e-01,  3.6137e-01, -9.9915e-01,  9.9367e-01, -5.7533e-01,
         -6.9278e-01, -5.6370e-01, -9.9967e-01, -9.9998e-01,  3.5949e-01,
         -3.8757e-02, -9.2495e-02, -3.9940e-01,  9.5931e-01, -1.5536e-01,
          2.3268e-01, -7.8038e-02, -9.7372e-01, -5.6429e-01, -3.2685e-01,
          2.4786e-01, -9.9948e-01, -2.8014e-01,  9.1632e-01, -4.5406e-01,
         -4.7145e-02, -9.4985e-01,  8.2291e-01, -9.7050e-01,  7.9104e-01,
          9.8231e-01,  6.6945e-01,  2.9161e-01, -9.9233e-01,  9.9287e-01,
         -1.6436e-01, -4.2826e-02, -9.3725e-01, -9.9426e-01,  9.9963e-01,
          1.3479e-01, -1.6790e-01, -2.6589e-01, -9.9632e-01,  5.7949e-01,
         -2.0110e-01, -8.1237e-01, -9.9499e-01,  5.2892e-02, -9.9371e-01,
         -9.9966e-01,  4.6190e-01,  3.9403e-01,  9.9969e-01,  9.7382e-01,
          4.1798e-01, -1.6332e-02, -9.4637e-01, -3.0390e-02, -9.9979e-01,
          7.3661e-02,  8.0103e-01, -9.8978e-01,  2.2847e-02,  9.8696e-01,
          9.9460e-01, -8.4571e-01, -8.0208e-01, -7.8753e-01,  7.2113e-01,
          9.8140e-01,  5.3083e-01, -4.4603e-01, -7.6596e-02, -1.4650e-01,
         -9.9470e-01, -9.4455e-01,  9.9819e-01, -4.9859e-01,  9.8945e-01,
          1.5442e-01,  7.9977e-01, -1.8714e-01, -2.7450e-01, -3.0111e-01,
         -9.9994e-01, -7.8744e-01,  2.0769e-01, -9.9958e-01,  9.9897e-01,
         -9.9940e-01, -2.4598e-01,  6.4502e-02,  5.9666e-01,  9.8660e-01,
          5.6678e-01, -9.9962e-01, -9.9921e-01, -9.7645e-01, -1.4184e-01,
          9.9653e-01,  1.4394e-01,  1.0562e-01,  3.6948e-01,  6.1591e-02,
          9.9624e-01,  4.7598e-02,  3.3356e-01, -2.8739e-01,  9.9910e-01,
          4.9893e-01, -9.9645e-01,  7.3801e-01, -9.9818e-01,  9.0974e-01,
          9.7337e-01,  9.8163e-01,  9.8619e-01, -7.0868e-01,  9.9949e-01,
         -9.9970e-01,  9.9988e-01, -9.9935e-01, -6.9731e-01,  9.9901e-01,
         -9.9462e-01, -4.5568e-01, -9.9595e-01, -7.4261e-01, -4.1857e-01,
          2.0958e-01, -7.0018e-01,  9.9460e-01, -9.9781e-01, -9.9817e-01,
          6.2496e-01,  2.5773e-01,  3.0692e-01,  7.8464e-01,  4.9414e-02,
          9.9643e-01,  9.4985e-02,  9.8964e-01, -2.3947e-01,  8.8521e-01,
          9.9999e-01, -6.4262e-01, -1.1594e-01, -9.9657e-01,  9.9637e-01,
         -3.4432e-01,  1.2871e-01,  9.3371e-01, -2.3211e-01, -6.6747e-02,
          8.2392e-01, -9.9538e-01, -2.8067e-01, -9.4433e-01,  9.2512e-01,
         -1.9581e-01,  4.1014e-01,  1.5732e-01, -2.8610e-01,  2.2825e-01,
         -9.9316e-01,  8.7344e-02, -9.9813e-01,  9.9762e-01, -2.0097e-01,
         -5.4145e-03,  1.2493e-01,  7.7946e-01, -9.2669e-01,  9.9706e-01,
          9.9872e-01, -9.9999e-01,  2.3497e-01,  9.9512e-01,  1.7694e-01,
          9.9298e-01, -9.8990e-01, -2.5011e-01,  9.7451e-01, -5.8868e-01,
          9.8414e-01,  5.6311e-01, -2.4267e-01,  9.7391e-01, -9.9476e-01,
          1.0887e-02, -7.5065e-01, -1.2423e-01,  8.5358e-01, -9.8685e-01,
          8.2592e-02,  6.7578e-01,  1.3652e-01, -9.9918e-01, -5.0114e-01,
         -9.9794e-01, -1.7166e-01,  9.9762e-01,  3.3334e-01,  9.9933e-01,
          1.8072e-01, -5.1689e-01,  2.0668e-01, -9.9886e-01, -7.0821e-01,
          1.3985e-01,  1.2827e-01,  4.4299e-01,  6.3108e-01,  2.0460e-01,
         -4.1334e-01, -9.9807e-01,  2.6136e-01,  8.3826e-01,  9.4714e-02,
          9.3921e-01,  2.7737e-01, -9.9426e-01, -9.5402e-01, -8.1975e-01,
          3.0735e-01,  6.5918e-02, -9.8629e-01,  8.4079e-01, -8.7933e-01,
          9.9961e-01, -9.9430e-01, -8.6146e-01, -9.7705e-01, -3.2851e-01,
          4.3911e-01,  4.0214e-01,  4.1487e-01, -5.0144e-01,  5.8008e-01,
         -9.3492e-01,  9.9211e-01, -9.9472e-01, -9.9574e-01,  9.9813e-01,
         -2.8329e-01, -7.5125e-01,  1.9631e-01, -5.4571e-01, -5.4410e-01,
         -2.7827e-01,  9.2650e-01, -9.7369e-01, -4.5977e-01, -9.9994e-01,
         -7.5137e-01, -6.1942e-01, -9.9721e-01, -9.4996e-01, -4.2251e-01,
         -9.9994e-01,  9.9676e-01,  9.7764e-01,  9.9941e-01, -9.9930e-01,
          9.5303e-01,  1.8769e-01,  9.9841e-01, -1.8022e-01, -4.3869e-01,
          6.8448e-01,  9.9836e-01,  1.3707e-01, -1.0268e-01, -2.8350e-01,
         -8.0697e-03,  2.4456e-01, -7.2475e-01,  4.9879e-01, -1.3575e-01,
          4.2700e-01, -9.9356e-01, -9.9813e-01,  9.9879e-01, -4.7360e-02,
          9.9282e-01,  2.9508e-01,  4.7730e-01, -9.6248e-01,  9.4457e-01,
         -2.2028e-01, -9.4172e-01, -9.9969e-01,  6.7530e-01, -9.9999e-01,
         -9.9581e-01,  6.3464e-01,  9.8920e-01, -9.9720e-01, -9.8963e-01,
         -1.2207e-01, -9.9945e-01,  2.2122e-01, -7.6008e-01,  8.8730e-02,
         -9.9712e-01,  7.3356e-01, -2.6536e-01,  9.4266e-02,  9.8677e-01,
         -9.8623e-01,  8.2868e-01, -6.5824e-01,  9.5350e-01, -1.8004e-01,
          3.4689e-01, -3.3729e-01, -7.4837e-01, -3.6920e-01, -9.7733e-01,
         -6.4816e-01, -9.9652e-01, -9.6354e-01,  9.9718e-01,  9.8871e-01,
         -9.9742e-01, -9.9404e-01,  8.1275e-01, -3.1389e-01,  9.9604e-01,
         -4.2956e-01, -9.9940e-01, -9.9912e-01,  6.8408e-02, -6.4509e-01,
          9.9599e-01, -4.1888e-01,  9.9985e-01,  9.4479e-01,  6.5758e-01,
          1.3966e-01, -4.7134e-01,  4.0407e-01, -4.2699e-02, -5.6515e-02,
          9.9949e-01, -7.7805e-01,  9.9751e-01],
        Atom(b45, [
            Structure(s, [List(['e',space2,'d',space2,'i',space2,'t',space2,'h']),
                          List(['e','d','i','t','h'])])])
        : [-0.3351,  0.7164,  0.9990, -0.9961,  0.8467,  0.1265,  0.9800,  0.1887,
         -0.9868, -0.7898,  0.9849,  0.9958, -0.3159, -0.9977, -0.0412, -0.9917,
          0.9919, -0.3140, -0.9993, -0.5668,  0.1694, -0.9990,  0.3358,  0.8784,
          0.9668,  0.2789,  0.9948,  0.9987,  0.9413,  0.6572,  0.2123, -0.9904,
          0.6951, -0.9982,  0.2022, -0.0884, -0.1487, -0.1395,  0.6721, -0.1643,
         -0.9299,  0.2519, -0.6193, -0.3659, -0.1212,  0.5928,  0.7823, -0.1779,
         -0.0446,  0.9999, -0.9479,  1.0000,  0.1922,  0.9999,  0.9978,  0.8346,
          0.9945,  0.4911, -0.0283,  0.7735,  0.9890, -0.3774,  0.9744, -0.6968,
         -0.4498, -0.8410,  0.1082,  0.3997, -0.6629,  0.8456,  0.8596,  0.1579,
          0.9981, -0.9675,  0.2221, -0.9675,  0.7206, -0.9996,  0.9869,  0.9994,
         -0.1171, -0.9974,  0.9964,  0.1633,  0.3855, -0.7923,  0.5243, -0.9969,
          0.3781, -0.9396, -0.3421, -0.9938,  0.0224,  0.5186,  0.9993, -0.1216,
         -0.0977,  0.0209,  0.6649, -0.2295, -0.8427,  0.2893, -0.6768,  0.6466,
          0.2826, -0.4091, -0.9591, -0.1236, -0.3998,  0.0477,  0.9970, -0.9855,
         -0.8187, -0.0739, -0.2432, -0.2049,  0.9946,  0.7843, -0.0926,  0.9992,
          0.0772, -0.1733,  0.9952, -0.3014,  0.5748, -0.2491, -0.3630,  0.5334,
          0.1671, -0.8853,  0.5591, -0.9914,  0.2377,  0.9978, -0.0940,  0.9992,
         -0.9985,  0.3991, -0.9995, -0.3001,  0.4209, -0.2425,  0.0999,  0.7945,
          0.9959,  0.2071, -0.2956,  0.3029,  0.1041,  0.1631,  0.5746,  0.9280,
         -0.9696,  1.0000,  0.6515,  0.0811,  0.6982,  0.1020,  0.1232,  0.1088,
          0.9918, -0.9977,  0.5585,  0.3353,  0.9979,  0.9942, -0.0895, -0.4344,
          0.9980,  0.7323,  0.1642, -0.8401, -0.6983,  0.1607,  0.0777,  0.5442,
          0.9746,  0.9978, -0.9909,  0.9975,  0.9998, -0.5682,  0.9506,  0.0241,
         -0.9978, -0.9950, -0.9899,  0.6643, -0.4963,  0.6050, -0.3776,  0.9790,
         -0.6409,  0.8839, -0.9801, -0.0427,  0.9931, -0.1805,  0.9989,  0.7765,
         -0.9980, -0.4882, -0.5124,  0.9823, -0.4739,  0.9859,  0.3847,  0.2853,
          0.9247, -1.0000, -0.6932, -0.2949, -0.6634,  0.7490,  0.9946,  0.0622,
         -0.2644, -0.2441,  0.0805,  0.9996, -0.9970, -0.5563,  0.6252, -0.9958,
         -0.9975,  0.9583, -0.2264, -0.1179, -0.5647, -0.8721,  0.1300,  0.6972,
          0.9874,  0.1740,  0.7570, -0.9988,  0.8345, -0.9842, -0.1766,  0.1342,
          0.6837, -0.5681, -0.9785,  0.4161,  0.8995,  0.0717, -0.3178, -0.1284,
          0.8758,  0.6232, -0.3544, -0.3711, -0.9921,  0.9969,  0.3672,  0.1353,
          0.9941, -0.9936,  0.5397,  0.1971, -0.2377, -1.0000, -0.0773, -0.8211,
          0.7391, -0.1559,  0.9956, -0.9877, -0.1573, -0.1502, -0.9989,  0.9380,
         -0.2214,  0.9953, -0.8140,  0.4676,  0.9962,  0.7106, -0.9953, -0.9992,
          0.4810,  0.9999, -0.9881, -0.2294,  0.9995,  0.6197, -0.9706, -0.9398,
         -0.9982, -0.9977, -0.6946, -0.1522, -0.0031,  0.9832,  0.0761, -0.0619,
          0.9905,  1.0000, -0.2049,  0.0862,  0.1390, -0.8877, -1.0000,  0.7286,
         -0.0398, -0.9997,  0.9974, -0.9929,  1.0000, -0.8085,  0.0385,  0.8644,
          0.1090, -0.2138,  0.2501,  0.9994,  0.9866, -0.3897,  0.0876, -0.3692,
          0.3040, -0.2900,  0.6304,  0.5074, -0.2747, -0.8453, -0.4157,  0.0564,
         -0.9962,  0.1621,  0.5748,  0.9621, -0.5242,  0.9399,  0.9878, -0.0336,
          0.6784, -0.4787, -0.9987, -0.0243,  0.3323, -0.7928,  0.5680,  0.6874,
          0.9953, -0.9854,  1.0000, -0.3155,  0.4984,  0.3399,  0.9997, -0.9999,
          0.2735,  0.3714, -0.1272,  0.6205,  0.9924,  0.4352,  0.7705, -0.2855,
         -0.3074,  0.9405,  0.9903, -0.2680, -0.4443,  0.8069,  0.7312,  0.9913,
          0.1040, -0.1670, -0.7742, -0.1151,  0.9859, -0.4269, -0.1552, -0.3558,
         -0.2233,  0.0025, -0.3123,  0.5626,  0.2209,  0.4142, -0.9911,  0.0244,
          0.7591,  0.9985, -0.9961, -0.2711,  0.9931, -0.2084, -0.2462,  0.8642,
          0.1051, -0.9718, -0.2535, -0.9992, -0.2207, -0.4990,  0.6292,  0.3674,
          0.0054, -0.3041,  0.5144, -0.8227,  0.8409,  0.1878,  0.9891, -0.0148,
         -0.5894, -0.2941,  0.6342,  0.3849,  0.1673,  0.9938, -0.9929,  0.9995,
         -0.0254, -0.9990,  0.8555, -0.2157, -0.9992,  0.1520, -0.9998,  0.9934,
         -0.4898,  0.5112,  0.6304, -0.9999, -1.0000, -0.1389,  0.0298, -0.2799,
         -0.5428,  0.9960, -0.2191,  0.4958,  0.1245, -0.9542, -0.3735,  0.6927,
          0.2763, -0.9994,  0.2473,  0.7776, -0.0425, -0.1848, -0.9498,  0.8087,
         -0.9738,  0.7561,  0.9852,  0.5529,  0.2180, -0.9950,  0.9901, -0.2166,
          0.1559, -0.8210, -0.9915,  0.9991, -0.2304, -0.1643, -0.1864, -0.9963,
          0.3115, -0.4772, -0.6071, -0.9965,  0.3594, -0.9962, -0.9997,  0.4530,
         -0.2929,  0.9999,  0.9859,  0.5057, -0.0520, -0.9615, -0.0674, -0.9998,
         -0.0236,  0.5204, -0.9919,  0.4127,  0.9909,  0.9920, -0.5469, -0.6824,
         -0.7335,  0.7475,  0.9723,  0.6998, -0.5464, -0.0685, -0.2710, -0.9952,
         -0.9488,  0.9970,  0.5510,  0.9915, -0.6009, -0.0902, -0.0668, -0.1735,
          0.1162, -1.0000, -0.8686,  0.1664, -0.9995,  0.9994, -0.9996, -0.4397,
          0.3601,  0.4080,  0.9880,  0.6184, -0.9995, -0.9991, -0.9515, -0.1197,
          0.9962, -0.0919,  0.2943,  0.4565,  0.0637,  0.9984, -0.3773,  0.4632,
          0.4587,  0.9979,  0.3287, -0.9959,  0.5651, -0.9986,  0.8045,  0.9693,
          0.9702,  0.9899,  0.4255,  0.9993, -0.9997,  0.9999, -0.9990,  0.4660,
          0.9993, -0.9916,  0.1377, -0.9969, -0.1156, -0.3322,  0.1355, -0.7793,
          0.9913, -0.9972, -0.9976,  0.7308,  0.4457,  0.5369,  0.2154,  0.1892,
          0.9938,  0.2299,  0.9882, -0.4297,  0.5532,  1.0000, -0.5255, -0.0863,
         -0.9949,  0.9925,  0.0028,  0.1935,  0.9547, -0.4154, -0.0962,  0.8902,
         -0.9935, -0.5817, -0.9647,  0.6313, -0.5086,  0.5163,  0.1755, -0.2685,
          0.2227, -0.9912,  0.2492, -0.9974,  0.9899,  0.4155,  0.1265, -0.0338,
          0.8097, -0.8912,  0.9972,  0.9989, -1.0000, -0.0589,  0.9952,  0.2382,
          0.9901, -0.9939, -0.3011,  0.8879, -0.7838,  0.9908,  0.6801, -0.2450,
          0.9882, -0.9927,  0.1749, -0.8752, -0.1455,  0.8433, -0.9801,  0.0621,
          0.4195,  0.2833, -0.9991, -0.5724, -0.9981, -0.2466,  0.9967,  0.4515,
          0.9992,  0.7708, -0.4549,  0.3131, -0.9979,  0.3828,  0.2479, -0.0343,
          0.1519, -0.5792, -0.1195, -0.5290, -0.9985,  0.4430,  0.1591,  0.0978,
          0.8971,  0.5674, -0.9895, -0.9072, -0.8507,  0.4464, -0.5301, -0.9349,
          0.9100, -0.7396,  0.9996, -0.9934, -0.9353, -0.9696, -0.3618,  0.3870,
          0.3454,  0.5673, -0.3376,  0.0968, -0.7701,  0.9845, -0.9948, -0.9958,
          0.9960,  0.0738, -0.2290, -0.5072, -0.6414, -0.4392, -0.3373,  0.7707,
         -0.9372, -0.6397, -1.0000, -0.4078, -0.1129, -0.9965, -0.8860, -0.6151,
         -1.0000,  0.9956,  0.9730,  0.9989, -0.9985,  0.8808,  0.1345,  0.9975,
         -0.2485, -0.4822,  0.4097,  0.9986,  0.3131, -0.6369,  0.0038, -0.1806,
          0.2826, -0.8939, -0.0804, -0.2646,  0.5944, -0.9954, -0.9983,  0.9993,
         -0.1788,  0.9909,  0.4010,  0.5233, -0.9524,  0.8419,  0.0389, -0.9184,
         -0.9996,  0.5675, -1.0000, -0.9920,  0.5045,  0.9802, -0.9969, -0.9902,
         -0.0673, -0.9995, -0.2235, -0.3876,  0.2514, -0.9954, -0.4020, -0.5799,
          0.6760,  0.9812, -0.9778,  0.7223, -0.7853,  0.9938,  0.0763,  0.3428,
         -0.2168, -0.2555,  0.0637, -0.9832, -0.4791, -0.9951, -0.9454,  0.9935,
          0.9902, -0.9983, -0.9920,  0.3035, -0.2068,  0.9923, -0.4352, -0.9993,
         -0.9994,  0.0833, -0.6112,  0.9955, -0.4125,  1.0000,  0.9343,  0.5979,
         -0.3118,  0.3366,  0.5329,  0.1716, -0.0452,  0.9995, -0.2318,  0.9957],
        Atom(b45, [
            Structure(s, [List(['i', space2, 'n', space2, 'g', space2, 'e']),
                          List(['i', 'n', 'g', 'e'])])])
        : [-0.3677,  0.7104,  0.9988, -0.9967,  0.8124,  0.2351,  0.9806, -0.2421,
         -0.9626, -0.7704,  0.9766,  0.9974, -0.8007, -0.9971,  0.0383, -0.9876,
          0.9904, -0.3649, -0.9995, -0.7317, -0.2041, -0.9994,  0.2491,  0.9655,
          0.9648,  0.2025,  0.9909,  0.9990,  0.9528,  0.4583,  0.3348, -0.9938,
          0.6433, -0.9987,  0.0666, -0.3872,  0.3441, -0.1258,  0.8009, -0.5722,
         -0.9245,  0.4205, -0.0797, -0.4824,  0.5083,  0.6265,  0.4490, -0.0357,
         -0.0389,  0.9997, -0.9432,  1.0000, -0.5633,  0.9994,  0.9955,  0.8612,
          0.9968,  0.3526, -0.6540,  0.8146,  0.9863, -0.4622,  0.9471, -0.5905,
         -0.0671, -0.8184, -0.4343, -0.0398, -0.5984,  0.8757,  0.6165,  0.1325,
          0.9961, -0.9428,  0.2403, -0.9533,  0.0549, -0.9997,  0.9601,  0.9994,
          0.1415, -0.9979,  0.9978,  0.0663, -0.1468, -0.5178, -0.4058, -0.9969,
          0.4862, -0.9174, -0.0512, -0.9877,  0.5406,  0.1654,  0.9995, -0.2188,
         -0.2053, -0.1751,  0.7042, -0.6210, -0.9155,  0.4819,  0.4334, -0.0608,
          0.6374,  0.0360, -0.9684, -0.5946, -0.1002, -0.1345,  0.9950, -0.9794,
         -0.9524, -0.1025, -0.1678, -0.7864,  0.9932,  0.9281, -0.1963,  0.9993,
          0.0564,  0.6118,  0.9971, -0.0237,  0.3708, -0.1802, -0.7369,  0.6923,
         -0.3474, -0.8633,  0.7074, -0.9934, -0.2684,  0.9981, -0.0599,  0.9995,
         -0.9986,  0.7157, -0.9998, -0.6505,  0.3852, -0.1468, -0.5368,  0.7260,
          0.9961,  0.2657, -0.5948,  0.2952,  0.1137,  0.0573,  0.6715,  0.9303,
         -0.9577,  0.9998,  0.9188,  0.5120,  0.7869,  0.0896, -0.4552,  0.2404,
          0.9725, -0.9981,  0.3004, -0.1556,  0.9977,  0.9898,  0.6427, -0.8578,
          0.9992,  0.4650, -0.3819, -0.7158, -0.6821, -0.7426,  0.2923,  0.5724,
          0.9615,  0.9956, -0.9952,  0.9947,  0.9931, -0.2967,  0.9418,  0.7618,
         -0.9972, -0.9919, -0.9921,  0.4103, -0.1833,  0.4473,  0.0592,  0.9765,
         -0.0296,  0.8616, -0.9678, -0.0306,  0.9870, -0.1036,  0.9992,  0.6763,
         -0.9989, -0.4578, -0.1955,  0.9852, -0.4296,  0.9787, -0.1584,  0.0266,
          0.9685, -0.9999,  0.1745, -0.0137, -0.5702,  0.6720,  0.9894, -0.7145,
         -0.1668, -0.2522, -0.0396,  0.9997, -0.9969, -0.6174,  0.7755, -0.9975,
         -0.9979,  0.9737, -0.2415,  0.1670, -0.4478, -0.6170,  0.1496,  0.8039,
          0.9943, -0.3036,  0.0787, -0.9996,  0.1054, -0.9620, -0.5348,  0.3415,
          0.5947, -0.5333, -0.9800, -0.4057,  0.9304,  0.5014, -0.5546,  0.1086,
          0.3138, -0.1425, -0.2708, -0.5770, -0.9897,  0.9984,  0.0734,  0.7148,
          0.9858, -0.9910,  0.8848, -0.6565, -0.0789, -0.9999,  0.2562, -0.5284,
          0.4863, -0.0657,  0.9969, -0.9702, -0.4618,  0.6302, -0.9987,  0.9563,
         -0.2329,  0.9960, -0.6549,  0.0991,  0.9959,  0.9233, -0.9943, -0.9995,
          0.9285,  0.9914, -0.9874, -0.1274,  0.9995, -0.4801, -0.9738, -0.9454,
         -0.9981, -0.9989, -0.3423, -0.7329, -0.3690,  0.9804,  0.1745, -0.1580,
          0.9925,  0.9998,  0.0311, -0.2000,  0.2989, -0.9459, -0.9991,  0.9185,
          0.0422, -0.9997,  0.9983, -0.9959,  0.9998, -0.7053, -0.5757,  0.7589,
          0.1742, -0.4101,  0.2969,  0.9995,  0.9839, -0.3580, -0.0847,  0.1172,
          0.0015,  0.0526,  0.6862, -0.0630, -0.4594, -0.8424,  0.4644,  0.6024,
         -0.9936,  0.8013,  0.4028,  0.9610, -0.8751,  0.9592,  0.9888,  0.0173,
          0.4755, -0.0720, -0.9680, -0.6043,  0.2770, -0.9249,  0.0721,  0.9112,
          0.9948, -0.9845,  0.9996, -0.3678,  0.7776, -0.4431,  0.9996, -0.9997,
          0.2219,  0.4534, -0.2491,  0.2378,  0.9942,  0.8180,  0.8960, -0.5623,
         -0.5287,  0.8737,  0.9851, -0.6593, -0.1570, -0.2278,  0.5930,  0.9924,
          0.3537, -0.1619, -0.2915, -0.7636,  0.9849, -0.4876, -0.6290, -0.2967,
         -0.7745,  0.6042,  0.6166, -0.1772,  0.6115,  0.4166, -0.9932,  0.3266,
          0.9235,  0.9989, -0.9922,  0.0066,  0.9915, -0.2987, -0.5642,  0.8166,
          0.8810, -0.9870, -0.4037, -0.9996, -0.1458, -0.3487,  0.5460,  0.4259,
          0.1293, -0.5836,  0.7715, -0.8458,  0.8974,  0.2288,  0.9863,  0.3159,
         -0.5201, -0.1667,  0.5210,  0.2192,  0.1674,  0.9896, -0.9900,  0.9992,
          0.3454, -0.9991,  0.1390, -0.4748, -0.9986,  0.6054, -0.9993,  0.9912,
         -0.4963, -0.5462, -0.4172, -0.9996, -0.9999,  0.0137,  0.4056, -0.2810,
         -0.5238,  0.9333, -0.1563,  0.1976, -0.0386, -0.9493, -0.1940, -0.1178,
          0.6704, -0.9997, -0.3644,  0.9547, -0.6555, -0.7663, -0.9497,  0.8666,
         -0.9490,  0.6759,  0.9841,  0.2947, -0.0163, -0.9953,  0.9823, -0.7911,
          0.1864, -0.9456, -0.9838,  0.9997, -0.1120, -0.2646, -0.0706, -0.9972,
          0.6936, -0.6012, -0.8872, -0.9934,  0.2407, -0.9916, -0.9996,  0.2940,
          0.3331,  0.9995,  0.9711,  0.4276, -0.2195, -0.9198, -0.1115, -0.9999,
         -0.0725,  0.8958, -0.9941, -0.0778,  0.9921,  0.9919, -0.8422, -0.9019,
         -0.7075,  0.5317,  0.9754,  0.2185, -0.5386, -0.0915, -0.3053, -0.9954,
         -0.9438,  0.9974, -0.5728,  0.9898,  0.1451,  0.6871, -0.2087,  0.0709,
         -0.5313, -0.9998, -0.8634,  0.2107, -0.9998,  0.9996, -0.9995, -0.0811,
          0.0839,  0.6342,  0.9860,  0.2976, -0.9996, -0.9992, -0.8390,  0.3371,
          0.9961,  0.0196,  0.4205,  0.1162,  0.1524,  0.9987, -0.5220,  0.0484,
         -0.4451,  0.9990,  0.3035, -0.9959,  0.8388, -0.9985,  0.9486,  0.9813,
          0.9749,  0.9789, -0.6002,  0.9993, -0.9997,  0.9999, -0.9991, -0.5209,
          0.9995, -0.9926, -0.2335, -0.9964, -0.7539, -0.0468,  0.1962, -0.5548,
          0.9962, -0.9985, -0.9986,  0.6285,  0.0663,  0.3591,  0.7586, -0.4951,
          0.9955,  0.0753,  0.9871, -0.1005,  0.9130,  1.0000, -0.6974, -0.1065,
         -0.9935,  0.9955, -0.5054,  0.1453,  0.9490, -0.5890, -0.0915,  0.8626,
         -0.9964, -0.1078, -0.8637,  0.7874, -0.1460,  0.6604,  0.0983, -0.2779,
          0.0748, -0.9943,  0.2801, -0.9987,  0.9921, -0.0723,  0.0768, -0.2792,
          0.7370, -0.8982,  0.9978,  0.9990, -1.0000,  0.2542,  0.9921, -0.2266,
          0.9849, -0.9952, -0.2857,  0.9675, -0.5389,  0.9864,  0.5223, -0.1226,
          0.9889, -0.9933, -0.5414, -0.8559,  0.0730,  0.8294, -0.9836, -0.0545,
          0.7989,  0.1353, -0.9995, -0.5078, -0.9985, -0.2253,  0.9968,  0.1981,
          0.9993,  0.2283, -0.4189,  0.1743, -0.9985, -0.7478,  0.2521,  0.0882,
         -0.0775,  0.5471, -0.2777,  0.1539, -0.9982,  0.2990,  0.8183,  0.1427,
          0.9027, -0.2039, -0.9931, -0.9648, -0.7914,  0.2423, -0.2340, -0.9271,
          0.9039, -0.9274,  0.9997, -0.9911, -0.9480, -0.9808, -0.0813,  0.8122,
          0.4753,  0.4555, -0.6201,  0.6608, -0.9010,  0.9928, -0.9962, -0.9954,
          0.9981,  0.0926, -0.6737,  0.1716, -0.6717, -0.4264, -0.1863,  0.9107,
         -0.9637, -0.3332, -0.9999, -0.4878, -0.4210, -0.9964, -0.8938, -0.4591,
         -0.9997,  0.9975,  0.9811,  0.9993, -0.9992,  0.9557,  0.3456,  0.9986,
         -0.1812, -0.3902,  0.6766,  0.9988, -0.5767,  0.0733, -0.1986, -0.3024,
          0.5450, -0.7946,  0.5936, -0.3697,  0.4344, -0.9924, -0.9989,  0.9992,
         -0.1652,  0.9909,  0.3619,  0.4643, -0.9528,  0.9391, -0.0569, -0.8968,
         -0.9998,  0.5890, -0.9999, -0.9935,  0.3971,  0.9894, -0.9988, -0.9840,
         -0.1708, -0.9993,  0.2695, -0.7941, -0.0578, -0.9962,  0.7039, -0.4958,
         -0.1940,  0.9891, -0.9570,  0.8960, -0.5731,  0.9243, -0.0995,  0.4913,
         -0.4952, -0.7543, -0.2427, -0.9746,  0.1229, -0.9913, -0.9595,  0.9957,
          0.9897, -0.9983, -0.9926,  0.6151,  0.3218,  0.9954, -0.6164, -0.9993,
         -0.9996,  0.1665, -0.4728,  0.9966, -0.2522,  0.9994,  0.9168,  0.3018,
          0.4987, -0.2918,  0.2973, -0.1120, -0.2150,  0.9993, -0.6752,  0.9948]
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
            Structure(s, [List(['g', space2, 'e', space2, 'r', space2, 'a', space2, 'd',space2,'l',space2,'i',space2,'n',space2,'e']),
                          List(['g','e','r','a','d','l','i','n','e'])])]),
               Atom(b45, [
                   Structure(s, [List(['b',space2,'o',space2,'b']),
                                 List(['b','o','b'])])]),
               Atom(b45, [
                   Structure(s, [List(['c',space2,'a',space2,'r',space2,'o',space2,'l']),
                                 List(['c','a','r','o','l'])])]),
               Atom(b45, [
                   Structure(s, [List(['a',space2,'l',space2,'i',space2,'e']),
                                 List(['a','l','i','c','e'])])]),
               Atom(b45, [
                   Structure(s, [List(['h',space2,'e',space2,'n',space2,'r',space2,'y']),
                                 List(['h','e','n','r','y'])])]),

               ]

    learner.assert_knowledge(background)
    head = Atom(b45, [A])
    body1 = Atom(copyskip1, [A,B])
    body2 = Atom(is_lowercase, [A])

    clause1 = Clause(head, Body(body1, body2))

    print('coverage clause 1 : ', learner.evaluateClause(covered, clause1))
    network1 = Network.Network.LoadNetwork('network1.txt')
    network2 = Network.Network.LoadNetwork('network2.txt')

    program = learner.learn(task, background, hs,network1,network2,encodingExamples,decoder,primitives)

    print(program)


