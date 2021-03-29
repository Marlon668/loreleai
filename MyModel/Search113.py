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
# from docs.SWIProlog2 import SWIProlog
from loreleai.reasoning.lp.prolog import SWIProlog, Prolog
import heapq as has
from MyModel import encoding2,decoding, Network
import numpy as np
import random
import time


class MyOwnLeaner:

    def __init__(self, solver_instance: Prolog,encoder:encoding2,max_body_literals=10):
        self._solver = solver_instance
        self._candidate_pool = []
        self._encoder = encoder
        self._max_body_literals = max_body_literals

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
            print('value : ' ,1-current_cand._value) #because of heapq (ordered from minimum to maximum)
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
        #if max_var_occurrences(head,body,2):
        #    return True
        #if max_pred_occurrences(Atom,Body,copy1,2)
        return False

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
    print('Search for b113')
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
            Structure(s, [List(['o','x',one,space2,three,'x','w']),
                          List([O2, X2,one,three,X2,W2])])]),
        Atom(b45, [
            Structure(s, [List(['o', 'x', one, space2, three, 'b', 'n']),
                          List([O2, X2, one, three, B2, N2])])]),

        Atom(b45, [
            Structure(s, [List(['o', 'x', one, space2, four, 'b', 'h']),
                          List([O2, X2, one, four, B2, H2])])]),
        Atom(b45, [
            Structure(s, [List(['o', 'x', one, space2, three, 'b', 'w']),
                          List([O2, X2, one, three, B2, W2])])]),
        Atom(b45, [
            Structure(s, [List(['o', 'x', one, space2, three, 'l', 'z']),
                          List([O2, X2, one, three, L2, Z2])])])}

    # negative examples
    neg = {Atom(b45, [
        Structure(s, [List(['z', 'm', 't', 'b','b','v','u','g','a','r','n','h','b']),
                      List(['m','b','v','g','r','h'])])]),
            Atom(b45, [Structure(s, [List(
                [one,two,five,K2,space2,nine,space2,M2,'a','r',space2,one,one,komma2,five,four,space2,'m','e','t','a','p','r','o','b',punt2,'p','d','f']),
                List([nine,space2,M2,'a'])])]),
            Atom(b45, [Structure(s, [List(
                [P2,seven,I2,'e',F2,'g','q','j',F2,'m',Y2,'v',S2,'e']),
                                     List(['p',seven,'i','e','f','g','q','j','f','m','y','v','s','e'])])]),
            Atom(b45, [
                Structure(s, [List(
                    ['y','o','k','h','i','g','i','x','k','y','y','i','k']),
                              List(['o','k','h','g','i','x','y','y','i'])])]),
            Atom(b45, [Structure(s, [List(
                ['l','a','u','r','a']),
                                     List(['l','a',U2,R2,'a'])])])}
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
            Structure(s, [List(['o', 'x', one, space2, three, 'x', 'w']),
                          List([O2, X2, one, three, X2, W2])])])
        : [-0.3778,  0.6793,  0.9981, -0.9825,  0.7940, -0.5621,  0.9551,  0.9395,
         -0.9511, -0.7339,  0.9679,  0.9931,  0.6635, -0.9972, -0.7205, -0.9704,
          0.9509, -0.1931, -0.9989,  0.7961,  0.2423, -0.9972,  0.5454,  0.3604,
          0.9381,  0.2521,  0.9834,  0.9979,  0.7687,  0.9801,  0.2129, -0.9670,
          0.5267, -0.9833,  0.1936, -0.6507, -0.1722, -0.2347,  0.0518,  0.2725,
         -0.7573,  0.9325, -0.7277, -0.5169, -0.6993,  0.3765,  0.4990, -0.0981,
         -0.4921,  0.9998, -0.9069,  1.0000,  0.7594,  0.9998,  0.9950,  0.8292,
          0.9847,  0.3372,  0.6304,  0.8966,  0.9745, -0.4765,  0.8410, -0.5650,
         -0.7177, -0.8973,  0.4041,  0.2011, -0.6115,  0.8786,  0.4044,  0.6167,
          0.9734, -0.9314,  0.2136, -0.9079,  0.7549, -0.9990,  0.9422,  0.9979,
         -0.9123, -0.9924,  0.9898, -0.3197,  0.7065, -0.9639,  0.3329, -0.9947,
          0.3404, -0.7340, -0.8758, -0.9591, -0.3892,  0.8081,  0.9982,  0.8429,
         -0.3738,  0.3720, -0.0399,  0.8003, -0.7773, -0.6535, -0.6404,  0.8396,
         -0.5535, -0.8228, -0.9387,  0.0781, -0.9338, -0.0042,  0.9837, -0.9557,
         -0.2704,  0.3337, -0.1136,  0.2980,  0.9787, -0.3160, -0.3577,  0.9983,
         -0.2668,  0.2937,  0.9865, -0.6966,  0.9173, -0.5267,  0.7453, -0.3032,
          0.6954, -0.8646,  0.6688, -0.9673,  0.4667,  0.9929, -0.3959,  0.9977,
         -0.9951, -0.1199, -0.9996,  0.6812,  0.8863, -0.1984,  0.1873,  0.8793,
          0.9761,  0.4170,  0.5808,  0.8641, -0.2462,  0.6562,  0.7595,  0.8145,
         -0.9451,  1.0000, -0.6657, -0.2326,  0.1020,  0.0362,  0.2153, -0.6563,
          0.9745, -0.9905, -0.8902,  0.6352,  0.9964,  0.9670, -0.6063,  0.7543,
          0.9911,  0.9459,  0.6629, -0.8814, -0.8374,  0.6488,  0.4039,  0.4306,
          0.9594,  0.9931, -0.9515,  0.9976,  1.0000, -0.7156,  0.9672, -0.8137,
         -0.9899, -0.9729, -0.9609,  0.7553, -0.9541, -0.8262, -0.4599,  0.8051,
         -0.7052,  0.7560, -0.9825, -0.0833,  0.9644, -0.3225,  0.9986,  0.9640,
         -0.9944,  0.4737, -0.9091,  0.8721, -0.4562,  0.9599,  0.6512,  0.5780,
          0.6490, -1.0000, -0.7988, -0.8444, -0.9068,  0.6202,  0.9737,  0.7728,
         -0.5157, -0.1341,  0.4285,  0.9989, -0.9919, -0.3937,  0.3050, -0.9801,
         -0.9930,  0.9367, -0.1534,  0.2379, -0.5638, -0.9863,  0.0334, -0.3362,
          0.9725,  0.6759,  0.9651, -0.9970,  0.8717, -0.8919,  0.6547,  0.4839,
          0.6562, -0.5318, -0.8180,  0.6084,  0.8810, -0.8743,  0.8007, -0.7584,
          0.9746,  0.6423, -0.9394,  0.4501, -0.9774,  0.9927,  0.9287, -0.4736,
          0.9899, -0.9803, -0.3137,  0.3890, -0.0153, -1.0000, -0.6428, -0.9736,
          0.8991, -0.4854,  0.9764, -0.8675, -0.2135, -0.1052, -0.9979,  0.8675,
         -0.3638,  0.9879, -0.9207,  0.5728,  0.9884,  0.0919, -0.9671, -0.9984,
         -0.2435,  1.0000, -0.9346, -0.3508,  0.9981,  0.7090, -0.9295, -0.8963,
         -0.9976, -0.9931, -0.7324,  0.7318, -0.4047,  0.9805, -0.4540, -0.4051,
          0.9712,  1.0000, -0.5430,  0.3796,  0.2101, -0.9040, -1.0000,  0.4070,
          0.1272, -0.9988,  0.9955, -0.9858,  1.0000, -0.9096,  0.8335,  0.2317,
         -0.2107,  0.6525,  0.3536,  0.9982,  0.9370, -0.7416,  0.2945, -0.2087,
          0.5162, -0.8446,  0.9316,  0.5685,  0.0820, -0.7908, -0.8124, -0.7442,
         -0.9877, -0.7164,  0.3540,  0.7880,  0.6127,  0.8848,  0.9615, -0.2971,
          0.9136, -0.1599, -0.9998,  0.6157,  0.4437,  0.3679,  0.8107,  0.4514,
          0.9766, -0.9339,  1.0000, -0.3831, -0.1368,  0.5558,  0.9991, -0.9994,
          0.3678, -0.7886,  0.7975,  0.9052,  0.9717, -0.7457, -0.5124,  0.6675,
          0.7134,  0.6602,  0.9683,  0.3375, -0.2292,  0.8684,  0.8761,  0.9649,
         -0.9076, -0.2694, -0.8348,  0.8890,  0.8631, -0.1530,  0.7242, -0.3984,
          0.6348, -0.6270, -0.5762,  0.8162,  0.1849,  0.4100, -0.9814, -0.8431,
         -0.1151,  0.9945, -0.9498, -0.5362,  0.9793, -0.5146,  0.2846,  0.7814,
         -0.5114, -0.7700, -0.3123, -0.9960,  0.0932, -0.7506,  0.7708,  0.9288,
          0.0691,  0.4211, -0.4573, -0.8924,  0.9073,  0.8377,  0.9511,  0.9047,
         -0.4954, -0.3610,  0.7763,  0.4579, -0.4743,  0.9857, -0.9680,  0.9993,
         -0.2967, -0.9976,  0.5434,  0.9663, -0.9972, -0.4674, -0.9998,  0.9659,
         -0.9277,  0.6636,  0.5900, -0.9998, -1.0000, -0.7318, -0.5574, -0.5425,
         -0.9207,  0.9995, -0.1576,  0.7345,  0.2550, -0.9008,  0.7767,  0.6471,
         -0.8033, -0.9982,  0.7331, -0.1894,  0.9122,  0.4470, -0.9302,  0.8568,
         -0.8804,  0.6243,  0.9064,  0.2153,  0.8973, -0.9964,  0.9828,  0.2207,
          0.4575, -0.6610, -0.9599,  0.9971, -0.7572,  0.2322, -0.4515, -0.9920,
         -0.4615,  0.3935, -0.1646, -0.9843,  0.6321, -0.9842, -0.9989,  0.5810,
         -0.4745,  0.9995,  0.9591,  0.7341, -0.3040, -0.8453,  0.3064, -0.9989,
         -0.8910, -0.2524, -0.9728,  0.9277,  0.9884,  0.9839,  0.3389, -0.0299,
         -0.9376,  0.6970,  0.7957,  0.9271, -0.6979,  0.2002, -0.1407, -0.9735,
         -0.8303,  0.9914,  0.5403,  0.9859, -0.0545, -0.3352,  0.8629, -0.1636,
          0.4314, -1.0000, -0.6979, -0.3586, -0.9972,  0.9964, -0.9983, -0.5653,
          0.9441, -0.6441,  0.9638,  0.6348, -0.9978, -0.9981, -0.8158, -0.6989,
          0.9814,  0.0357,  0.7090,  0.9374,  0.3080,  0.9979,  0.3778,  0.9414,
          0.6511,  0.9946,  0.4713, -0.9884,  0.0282, -0.9966, -0.1526,  0.9058,
          0.9459,  0.9628,  0.3477,  0.9980, -0.9985,  0.9998, -0.9987,  0.5936,
          0.9981, -0.9689,  0.7771, -0.9914,  0.3821, -0.7983,  0.2349, -0.6370,
          0.9832, -0.9889, -0.9924,  0.8524,  0.8524,  0.8667, -0.3631,  0.7163,
          0.9915,  0.3464,  0.9267, -0.6312, -0.4008,  1.0000, -0.0727,  0.9327,
         -0.9821,  0.9273,  0.6185,  0.3881,  0.8889, -0.4183,  0.8980,  0.8870,
         -0.9927, -0.4299, -0.9955, -0.6406, -0.8792,  0.1801,  0.4077,  0.2935,
          0.2218, -0.9919,  0.4131, -0.9910,  0.9705,  0.1874,  0.3589, -0.0537,
          0.8638, -0.7456,  0.9936,  0.9942, -1.0000,  0.4812,  0.9716,  0.8353,
          0.9594, -0.9646, -0.4086,  0.6805, -0.3188,  0.9644,  0.6354, -0.3051,
          0.9695, -0.9875,  0.4271, -0.9119, -0.1420,  0.8486, -0.9565,  0.4177,
         -0.5177,  0.6606, -0.9964, -0.8499, -0.9971, -0.3644,  0.9869,  0.9334,
          0.9975,  0.8790, -0.6039, -0.1585, -0.9947,  0.6125,  0.4938, -0.2506,
          0.8094, -0.7117,  0.3099, -0.8119, -0.9976,  0.4067, -0.4353,  0.2698,
          0.6876,  0.8605, -0.9536, -0.3196, -0.8081,  0.5954, -0.7420, -0.8664,
          0.7851,  0.3609,  0.9983, -0.9855, -0.9254, -0.8957, -0.8211, -0.1927,
          0.4257,  0.3830,  0.5810, -0.6518, -0.4278,  0.9545, -0.9767, -0.9737,
          0.9910, -0.6984,  0.7040, -0.3952, -0.6912, -0.6140, -0.3534,  0.0026,
         -0.5246, -0.5349, -0.9999, -0.8961,  0.8055, -0.9859, -0.8768, -0.6306,
         -1.0000,  0.9866,  0.8189,  0.9957, -0.9977,  0.8856,  0.4903,  0.9932,
         -0.2257, -0.7024,  0.1316,  0.9922,  0.7364, -0.9241,  0.2736, -0.4721,
         -0.1309, -0.8323, -0.8139,  0.7125,  0.9029, -0.9617, -0.9968,  0.9963,
         -0.3326,  0.9776,  0.5844,  0.0899, -0.9078,  0.1729,  0.7338, -0.8322,
         -0.9979,  0.8984, -1.0000, -0.9748,  0.4332,  0.9864, -0.9889, -0.9447,
          0.1341, -0.9990, -0.5234,  0.4281,  0.8372, -0.9739, -0.4307, -0.6880,
          0.7381,  0.9601, -0.9502, -0.5478, -0.6897,  0.9990, -0.0120,  0.5564,
          0.2037,  0.6899,  0.7748, -0.8827, -0.6187, -0.9782, -0.7992,  0.9831,
          0.9569, -0.9939, -0.9818, -0.6508, -0.1409,  0.9729, -0.5620, -0.9980,
         -0.9982,  0.0628, -0.7639,  0.9887, -0.3469,  1.0000,  0.7828,  0.6802,
         -0.7260,  0.8753,  0.4839,  0.8005, -0.4476,  0.9991,  0.7642,  0.9830],
        Atom(b45, [
            Structure(s, [List(['o', 'x', one, space2, three, 'b', 'n']),
                          List([O2, X2, one, three, B2, N2])])])
        : [-0.3832,  0.6167,  0.9990, -0.9929,  0.9180, -0.4415,  0.9626,  0.9194,
         -0.9619, -0.8803,  0.9784,  0.9966,  0.2450, -0.9984, -0.6423, -0.9725,
          0.9747, -0.1456, -0.9997,  0.6466,  0.3723, -0.9982,  0.3759,  0.6103,
          0.9722,  0.4972,  0.9909,  0.9991,  0.9099,  0.9721, -0.0295, -0.9842,
          0.5468, -0.9920, -0.2350, -0.7580, -0.3232, -0.0974,  0.2658, -0.1144,
         -0.9384,  0.9200, -0.8118, -0.3737, -0.6877,  0.5547,  0.6405, -0.2056,
         -0.2353,  0.9998, -0.9575,  1.0000,  0.7984,  0.9999,  0.9980,  0.8606,
          0.9926,  0.3268,  0.3963,  0.9331,  0.9918, -0.3980,  0.9185, -0.6458,
         -0.5178, -0.9446,  0.7590,  0.1443, -0.7719,  0.8750,  0.7774,  0.5043,
          0.9920, -0.9477,  0.4519, -0.9275,  0.7950, -0.9993,  0.9741,  0.9992,
         -0.9091, -0.9965,  0.9952, -0.4192,  0.8842, -0.9718,  0.2429, -0.9961,
          0.3157, -0.9129, -0.8029, -0.9726, -0.2505,  0.8703,  0.9994,  0.6526,
         -0.1955,  0.2795,  0.0026,  0.8851, -0.9031, -0.6449, -0.4979,  0.7486,
         -0.1125, -0.8576, -0.9253,  0.2195, -0.8535, -0.2270,  0.9947, -0.9832,
         -0.4135,  0.3216, -0.1988,  0.2656,  0.9894,  0.0638,  0.0084,  0.9993,
         -0.1575,  0.6330,  0.9961, -0.4259,  0.8932, -0.4414,  0.6639, -0.4048,
          0.5722, -0.9316,  0.6777, -0.9856,  0.1387,  0.9965, -0.3742,  0.9993,
         -0.9981,  0.4317, -0.9999,  0.8688,  0.8549, -0.0781, -0.4029,  0.8891,
          0.9916,  0.2875,  0.1547,  0.8511, -0.6432,  0.5752,  0.7460,  0.9261,
         -0.9780,  1.0000, -0.2540,  0.1915,  0.3491,  0.0562, -0.1342, -0.4924,
          0.9777, -0.9966, -0.6095,  0.1214,  0.9983,  0.9875, -0.7473,  0.6212,
          0.9956,  0.8755,  0.7709, -0.8143, -0.7783,  0.3849,  0.1459,  0.4075,
          0.9840,  0.9959, -0.9817,  0.9986,  1.0000, -0.8285,  0.9633, -0.5063,
         -0.9961, -0.9853, -0.9844,  0.8321, -0.9666, -0.7688, -0.5848,  0.9230,
         -0.4383,  0.9030, -0.9923,  0.0673,  0.9694, -0.1340,  0.9993,  0.7785,
         -0.9978, -0.0817, -0.6441,  0.9606, -0.3933,  0.9707,  0.7299,  0.6665,
          0.6078, -1.0000, -0.6162, -0.7955, -0.9222,  0.8110,  0.9937,  0.8283,
         -0.6281, -0.0970,  0.3327,  0.9994, -0.9948, -0.3834,  0.5120, -0.9928,
         -0.9975,  0.9754, -0.0847, -0.1204, -0.5423, -0.9917,  0.0161, -0.3443,
          0.9849,  0.7318,  0.9629, -0.9982,  0.7647, -0.9775,  0.2527,  0.4324,
          0.6403, -0.5783, -0.9201,  0.3430,  0.9038, -0.8415,  0.3228, -0.7020,
          0.9815,  0.4410, -0.9433,  0.6281, -0.9820,  0.9947,  0.9075, -0.0617,
          0.9971, -0.9919, -0.3939, -0.0038, -0.1509, -1.0000, -0.5834, -0.9596,
          0.8715, -0.4539,  0.9915, -0.9530, -0.5949, -0.5675, -0.9985,  0.9297,
         -0.3518,  0.9949, -0.8983,  0.5877,  0.9956,  0.0081, -0.9857, -0.9991,
         -0.3152,  1.0000, -0.9719, -0.1657,  0.9991,  0.2444, -0.9723, -0.8912,
         -0.9993, -0.9972, -0.7066,  0.7808, -0.3228,  0.9799, -0.3549, -0.3177,
          0.9847,  1.0000, -0.5479,  0.2779,  0.1826, -0.9172, -1.0000,  0.2769,
         -0.0806, -0.9995,  0.9980, -0.9942,  1.0000, -0.7733,  0.6855,  0.6792,
         -0.0562,  0.3156,  0.1171,  0.9994,  0.9463, -0.6072,  0.2564, -0.0226,
          0.3332, -0.8368,  0.9479,  0.8081, -0.0056, -0.8712, -0.6022, -0.5020,
         -0.9959, -0.2745,  0.4618,  0.9167,  0.6099,  0.9315,  0.9785, -0.1292,
          0.9382, -0.3682, -0.9999,  0.6417,  0.5291, -0.1098,  0.7324,  0.5387,
          0.9916, -0.9546,  1.0000, -0.2513, -0.1548,  0.4265,  0.9996, -0.9998,
          0.2400, -0.7683,  0.5260,  0.8593,  0.9893, -0.7346, -0.3587,  0.3575,
          0.7868,  0.8934,  0.9871, -0.1677, -0.3319,  0.7793,  0.9074,  0.9656,
         -0.7434, -0.3537, -0.7014,  0.6776,  0.9349, -0.5197,  0.7313, -0.2912,
          0.6651, -0.6009, -0.1641,  0.9061,  0.0073,  0.3489, -0.9877, -0.7800,
         -0.2307,  0.9984, -0.9847, -0.7265,  0.9935, -0.4306,  0.3852,  0.8331,
          0.0235, -0.9022, -0.1142, -0.9989,  0.1307, -0.7013,  0.7757,  0.9358,
         -0.0480,  0.5900, -0.0336, -0.9333,  0.9642,  0.7089,  0.9895,  0.9084,
         -0.5383, -0.1836,  0.6962,  0.4275, -0.4149,  0.9924, -0.9885,  0.9996,
         -0.5268, -0.9989,  0.4167,  0.9592, -0.9986, -0.5774, -0.9999,  0.9825,
         -0.7980,  0.5204,  0.2339, -1.0000, -1.0000, -0.5644, -0.6108, -0.5143,
         -0.8623,  0.9996, -0.2932,  0.8119,  0.2251, -0.9597,  0.7480,  0.4654,
         -0.7194, -0.9993,  0.8231,  0.1376,  0.8370,  0.0192, -0.9552,  0.6926,
         -0.9605,  0.7169,  0.9522,  0.4454,  0.9140, -0.9973,  0.9886, -0.1023,
          0.5357, -0.5577, -0.9793,  0.9988, -0.4416,  0.0462, -0.3809, -0.9947,
         -0.0451,  0.2613, -0.2861, -0.9885,  0.5452, -0.9947, -0.9993,  0.4335,
          0.0277,  0.9999,  0.9549,  0.8393, -0.4451, -0.9308,  0.2423, -0.9996,
         -0.7185, -0.6384, -0.9903,  0.9411,  0.9934,  0.9941,  0.2418, -0.5063,
         -0.7749,  0.6827,  0.8763,  0.8957, -0.7502, -0.0056, -0.3560, -0.9853,
         -0.9079,  0.9968,  0.3078,  0.9943,  0.2178, -0.0140,  0.7394, -0.3259,
          0.1553, -1.0000, -0.8822, -0.1011, -0.9990,  0.9983, -0.9989, -0.7963,
          0.8895, -0.4700,  0.9730,  0.6385, -0.9991, -0.9993, -0.8035, -0.6948,
          0.9926,  0.0182,  0.6886,  0.8995,  0.7757,  0.9992, -0.0520,  0.9090,
          0.4564,  0.9980,  0.5725, -0.9959,  0.4878, -0.9984, -0.2151,  0.9359,
          0.9783,  0.9839,  0.0411,  0.9990, -0.9995,  0.9999, -0.9996,  0.3081,
          0.9993, -0.9801,  0.7869, -0.9963, -0.0605, -0.5825, -0.0578, -0.7214,
          0.9944, -0.9924, -0.9974,  0.5828,  0.7146,  0.9166,  0.0524,  0.6299,
          0.9976,  0.4924,  0.9578, -0.5756, -0.1693,  1.0000, -0.4434,  0.8111,
         -0.9920,  0.9719,  0.7431,  0.3954,  0.9125, -0.4465,  0.8423,  0.9103,
         -0.9968, -0.4068, -0.9978, -0.3838, -0.8273,  0.4356,  0.4022,  0.1838,
          0.2872, -0.9950,  0.4096, -0.9938,  0.9829, -0.2062,  0.1475, -0.1175,
          0.8196, -0.8101,  0.9948,  0.9987, -1.0000,  0.3579,  0.9851,  0.8550,
          0.9776, -0.9889, -0.3175,  0.6726, -0.6998,  0.9830,  0.7040, -0.0720,
          0.9495, -0.9931,  0.3620, -0.9195, -0.1854,  0.9564, -0.9873,  0.4305,
         -0.3507,  0.6965, -0.9987, -0.5150, -0.9987, -0.2997,  0.9936,  0.8736,
          0.9992,  0.9206, -0.6220,  0.1909, -0.9984,  0.3915,  0.4493, -0.3549,
          0.5942, -0.3200,  0.2371, -0.8439, -0.9986,  0.3348,  0.0543,  0.1393,
          0.9262,  0.9194, -0.9889, -0.6164, -0.8669,  0.4806, -0.7364, -0.8960,
          0.5191,  0.3856,  0.9995, -0.9940, -0.8817, -0.9302, -0.7646,  0.1057,
          0.4868,  0.5227,  0.5046, -0.5966, -0.6672,  0.9814, -0.9910, -0.9865,
          0.9976, -0.4179,  0.4497, -0.3445, -0.6912, -0.7156, -0.1894, -0.0851,
         -0.7653, -0.4215, -1.0000, -0.8412,  0.8468, -0.9927, -0.9320, -0.6886,
         -1.0000,  0.9957,  0.8901,  0.9985, -0.9993,  0.9354,  0.4183,  0.9973,
         -0.1771, -0.6953,  0.7091,  0.9966,  0.7351, -0.9153,  0.2317, -0.4367,
         -0.4425, -0.8678, -0.6022,  0.1797,  0.8174, -0.9830, -0.9971,  0.9985,
         -0.2070,  0.9884,  0.4751,  0.2154, -0.9771,  0.4238,  0.1735, -0.9298,
         -0.9989,  0.9361, -1.0000, -0.9887,  0.5260,  0.9939, -0.9933, -0.9740,
          0.0067, -0.9995, -0.6616,  0.0105,  0.8982, -0.9932, -0.2474, -0.7657,
          0.8270,  0.9823, -0.9635, -0.2004, -0.2678,  0.9993, -0.0250,  0.6435,
          0.5514,  0.1281,  0.2352, -0.9270, -0.5007, -0.9909, -0.9601,  0.9927,
          0.9811, -0.9981, -0.9930, -0.1993,  0.2570,  0.9907, -0.7898, -0.9991,
         -0.9992, -0.1981, -0.8458,  0.9954, -0.2216,  1.0000,  0.9358,  0.6133,
         -0.8617,  0.9403,  0.6954,  0.8612, -0.6313,  0.9997,  0.8464,  0.9948],
        Atom(b45, [
            Structure(s, [List(['o', 'x', one, space2, four, 'b', 'h']),
                          List([O2, X2, one, four, B2, H2])])])
        : [-0.4771,  0.6949,  0.9995, -0.9938,  0.9429, -0.4855,  0.9840,  0.8960,
         -0.9805, -0.8889,  0.9850,  0.9974,  0.3947, -0.9993, -0.5628, -0.9895,
          0.9807, -0.2320, -0.9998,  0.5795,  0.3867, -0.9991,  0.4897,  0.6114,
          0.9786,  0.5109,  0.9937,  0.9996,  0.9504,  0.9685,  0.3060, -0.9894,
          0.7184, -0.9966, -0.2154, -0.6634, -0.2754, -0.1737,  0.4400, -0.2014,
         -0.9390,  0.7993, -0.8456, -0.4042, -0.3416,  0.6110,  0.7816, -0.0821,
         -0.2753,  0.9999, -0.9683,  1.0000,  0.7960,  1.0000,  0.9992,  0.8730,
          0.9957,  0.4024,  0.6591,  0.9432,  0.9908, -0.5077,  0.9483, -0.7177,
         -0.7296, -0.9372,  0.7162,  0.1732, -0.7163,  0.8646,  0.8600,  0.5293,
          0.9956, -0.9678,  0.3058, -0.9548,  0.8567, -0.9997,  0.9821,  0.9995,
         -0.9184, -0.9981,  0.9969, -0.3164,  0.8913, -0.9675,  0.4851, -0.9981,
          0.4029, -0.9271, -0.7993, -0.9844, -0.1919,  0.8656,  0.9996,  0.4961,
         -0.1277,  0.2841,  0.0144,  0.8439, -0.9359, -0.5891, -0.7640,  0.8607,
         -0.2143, -0.8649, -0.9400,  0.3044, -0.8421, -0.2085,  0.9978, -0.9919,
         -0.2976,  0.2980, -0.2879,  0.4223,  0.9945, -0.0045, -0.0641,  0.9997,
         -0.2758,  0.4813,  0.9968, -0.4535,  0.8932, -0.4005,  0.6874, -0.0517,
          0.6075, -0.9138,  0.6527, -0.9924,  0.3178,  0.9978, -0.4409,  0.9996,
         -0.9989,  0.1546, -0.9999,  0.8063,  0.8175, -0.0327, -0.1743,  0.9297,
          0.9955,  0.3350,  0.2029,  0.8417, -0.5187,  0.7078,  0.8195,  0.8935,
         -0.9836,  1.0000, -0.1093,  0.0963,  0.5050,  0.0643,  0.0145, -0.4372,
          0.9836, -0.9984, -0.0906,  0.3563,  0.9989,  0.9910, -0.8097,  0.6577,
          0.9969,  0.8999,  0.8462, -0.8497, -0.7546,  0.5441,  0.2123,  0.4216,
          0.9857,  0.9985, -0.9853,  0.9993,  1.0000, -0.7950,  0.9685, -0.6619,
         -0.9976, -0.9894, -0.9896,  0.8587, -0.9700, -0.5924, -0.6720,  0.9605,
         -0.6964,  0.9246, -0.9939, -0.0529,  0.9871, -0.1871,  0.9997,  0.7472,
         -0.9991, -0.3538, -0.7081,  0.9820, -0.4848,  0.9857,  0.7918,  0.6491,
          0.6004, -1.0000, -0.8340, -0.7075, -0.8988,  0.8289,  0.9976,  0.8063,
         -0.5134, -0.0749,  0.4439,  0.9997, -0.9973, -0.4241,  0.6186, -0.9942,
         -0.9986,  0.9851, -0.2005, -0.3395, -0.6012, -0.9894,  0.0717, -0.0980,
          0.9890,  0.7288,  0.9779, -0.9991,  0.8923, -0.9869,  0.3850,  0.4243,
          0.6393, -0.5961, -0.9066,  0.5572,  0.9331, -0.8979,  0.3879, -0.6338,
          0.9836,  0.7114, -0.9255,  0.5215, -0.9925,  0.9974,  0.8736, -0.2941,
          0.9978, -0.9961, -0.3077,  0.3779, -0.1213, -1.0000, -0.5416, -0.9659,
          0.9291, -0.3337,  0.9947, -0.9647, -0.4221, -0.5229, -0.9991,  0.9566,
         -0.4364,  0.9969, -0.8849,  0.6204,  0.9980, -0.1927, -0.9936, -0.9996,
         -0.1924,  1.0000, -0.9884, -0.1715,  0.9996,  0.7140, -0.9826, -0.9370,
         -0.9995, -0.9981, -0.7707,  0.7568, -0.3544,  0.9866, -0.5320, -0.2313,
          0.9904,  1.0000, -0.6178,  0.1883,  0.0753, -0.9242, -1.0000,  0.4199,
          0.0889, -0.9998,  0.9985, -0.9963,  1.0000, -0.7440,  0.8248,  0.8178,
         -0.1527,  0.1556,  0.1173,  0.9998,  0.9660, -0.6570,  0.1672, -0.2885,
          0.4640, -0.8451,  0.9181,  0.8608,  0.0603, -0.9071, -0.6730, -0.6300,
         -0.9973, -0.4462,  0.5267,  0.9354,  0.4983,  0.9300,  0.9876, -0.1086,
          0.9337, -0.5309, -0.9999,  0.5899,  0.5178, -0.0310,  0.8122,  0.6896,
          0.9965, -0.9703,  1.0000, -0.4267, -0.2596,  0.5891,  0.9999, -1.0000,
          0.2614, -0.7664,  0.2648,  0.8829,  0.9946, -0.6849, -0.3973, -0.1981,
          0.5995,  0.9578,  0.9932, -0.0236, -0.3261,  0.9186,  0.8750,  0.9836,
         -0.6480, -0.3013, -0.8438,  0.7865,  0.9604, -0.4540,  0.7124, -0.4362,
          0.6363, -0.6826, -0.5477,  0.8981, -0.0830,  0.3945, -0.9934, -0.8769,
         -0.1826,  0.9990, -0.9898, -0.7383,  0.9969, -0.4115,  0.3442,  0.8648,
         -0.2295, -0.9319, -0.2024, -0.9995,  0.2525, -0.6906,  0.8299,  0.9376,
          0.0126,  0.5813,  0.1189, -0.9376,  0.9567,  0.5953,  0.9928,  0.8656,
         -0.4950, -0.1615,  0.6949,  0.3716, -0.4281,  0.9961, -0.9918,  0.9998,
         -0.5790, -0.9995,  0.7183,  0.9420, -0.9994, -0.6775, -1.0000,  0.9909,
         -0.7728,  0.7530,  0.6431, -1.0000, -1.0000, -0.6531, -0.6207, -0.5794,
         -0.8818,  0.9998, -0.0574,  0.8408,  0.1457, -0.9672,  0.6921,  0.7484,
         -0.6450, -0.9997,  0.7505,  0.2012,  0.8404,  0.0052, -0.9603,  0.8041,
         -0.9689,  0.7646,  0.9724,  0.5641,  0.8736, -0.9989,  0.9944,  0.0157,
          0.5294, -0.3918, -0.9887,  0.9991, -0.3521,  0.0363, -0.2816, -0.9971,
         -0.1020,  0.2252, -0.0762, -0.9910,  0.6260, -0.9961, -0.9996,  0.4414,
         -0.3935,  1.0000,  0.9783,  0.8163, -0.4492, -0.9387,  0.2486, -0.9998,
         -0.7594, -0.6077, -0.9932,  0.9315,  0.9968,  0.9953,  0.2652, -0.4416,
         -0.7778,  0.6826,  0.9370,  0.8587, -0.8329,  0.1220, -0.3000, -0.9927,
         -0.9080,  0.9983,  0.6837,  0.9972, -0.1351, -0.2553,  0.6958, -0.2588,
          0.2684, -1.0000, -0.8699, -0.1994, -0.9993,  0.9993, -0.9995, -0.7171,
          0.9036, -0.4077,  0.9834,  0.6814, -0.9996, -0.9997, -0.8754, -0.6908,
          0.9963,  0.0272,  0.6784,  0.9277,  0.6482,  0.9996, -0.3507,  0.8857,
          0.7064,  0.9990,  0.6772, -0.9975,  0.3907, -0.9991, -0.1866,  0.9688,
          0.9873,  0.9897,  0.5675,  0.9995, -0.9998,  1.0000, -0.9998,  0.6319,
          0.9996, -0.9875,  0.8352, -0.9985,  0.2773, -0.7407,  0.0406, -0.8095,
          0.9961, -0.9967, -0.9987,  0.6901,  0.6425,  0.8659, -0.1437,  0.7037,
          0.9986,  0.4430,  0.9696, -0.6230, -0.3478,  1.0000, -0.2511,  0.5944,
         -0.9953,  0.9832,  0.7827,  0.4331,  0.9341, -0.5360,  0.7057,  0.9302,
         -0.9977, -0.5856, -0.9981, -0.3230, -0.7559,  0.5924,  0.2974,  0.1529,
          0.2494, -0.9974,  0.3660, -0.9963,  0.9871,  0.0665,  0.0750, -0.3288,
          0.8781, -0.8688,  0.9972,  0.9993, -1.0000,  0.4566,  0.9933,  0.8609,
          0.9740, -0.9944, -0.2756,  0.7034, -0.7626,  0.9923,  0.7152, -0.2505,
          0.9609, -0.9965,  0.5946, -0.9141, -0.1351,  0.9327, -0.9895,  0.4083,
         -0.3568,  0.7080, -0.9992, -0.3418, -0.9993, -0.2689,  0.9957,  0.9258,
          0.9995,  0.9534, -0.6512,  0.1471, -0.9990,  0.5724,  0.5223, -0.3661,
          0.5272, -0.7176,  0.0215, -0.8807, -0.9994,  0.4291, -0.3479,  0.3242,
          0.9352,  0.9539, -0.9942, -0.5262, -0.9135,  0.5466, -0.8568, -0.9386,
          0.6577,  0.3566,  0.9998, -0.9966, -0.9233, -0.9198, -0.8089, -0.0670,
          0.5041,  0.5304,  0.4129, -0.7442, -0.5686,  0.9876, -0.9942, -0.9920,
          0.9985, -0.2640,  0.5630, -0.3506, -0.7551, -0.5771, -0.2760, -0.0978,
         -0.8830, -0.5903, -1.0000, -0.7630,  0.8385, -0.9961, -0.9243, -0.6487,
         -1.0000,  0.9976,  0.9077,  0.9991, -0.9994,  0.9152,  0.3536,  0.9982,
         -0.1577, -0.7021,  0.6070,  0.9983,  0.6685, -0.9091,  0.2427, -0.3658,
         -0.3485, -0.9153, -0.7016,  0.0159,  0.8715, -0.9894, -0.9987,  0.9993,
         -0.2787,  0.9928,  0.4453,  0.2238, -0.9737,  0.3305,  0.1831, -0.9179,
         -0.9994,  0.9262, -1.0000, -0.9940,  0.5755,  0.9937, -0.9966, -0.9884,
          0.0955, -0.9997, -0.7916, -0.0071,  0.8227, -0.9944, -0.5796, -0.7883,
          0.9025,  0.9871, -0.9822, -0.1816, -0.4575,  0.9995,  0.0238,  0.6092,
          0.4973,  0.3628,  0.0686, -0.9470, -0.4484, -0.9952, -0.9729,  0.9964,
          0.9886, -0.9990, -0.9957, -0.2975,  0.0306,  0.9955, -0.6969, -0.9995,
         -0.9995, -0.1178, -0.8568,  0.9982, -0.2922,  1.0000,  0.9297,  0.5563,
         -0.8977,  0.9438,  0.6815,  0.7830, -0.5227,  0.9998,  0.8191,  0.9967],
        Atom(b45, [
            Structure(s, [List(['o', 'x', one, space2, three, 'b', 'w']),
                          List([O2, X2, one, three, B2, W2])])])
        : [-3.2103e-01,  7.0177e-01,  9.9812e-01, -9.8188e-01,  8.4108e-01,
         -2.5596e-01,  9.5828e-01,  8.6480e-01, -9.4336e-01, -7.3738e-01,
          9.5359e-01,  9.9426e-01,  1.9862e-01, -9.9658e-01, -5.9749e-01,
         -9.6415e-01,  9.4551e-01, -5.1920e-02, -9.9923e-01,  7.9485e-01,
          3.4629e-01, -9.9771e-01,  4.7084e-01,  6.0533e-01,  9.2660e-01,
          3.5052e-01,  9.8059e-01,  9.9868e-01,  8.4568e-01,  9.6597e-01,
          1.4378e-01, -9.6883e-01,  6.3550e-01, -9.8439e-01, -1.5462e-01,
         -5.6085e-01, -2.7323e-01, -1.3532e-01, -6.5194e-02,  3.2654e-03,
         -8.5658e-01,  9.3048e-01, -7.7205e-01, -3.9772e-01, -6.0164e-01,
          4.8625e-01,  6.1113e-01, -2.5770e-01, -4.2017e-01,  9.9976e-01,
         -9.2514e-01,  1.0000e+00,  7.6574e-01,  9.9978e-01,  9.9579e-01,
          7.4142e-01,  9.8707e-01,  3.4944e-01,  3.3241e-01,  8.5546e-01,
          9.8487e-01, -5.0547e-01,  8.5985e-01, -5.6204e-01, -5.8104e-01,
         -9.3183e-01,  7.3290e-01,  1.8143e-01, -6.8577e-01,  8.6363e-01,
          6.1450e-01,  5.7866e-01,  9.8030e-01, -9.3372e-01,  2.7285e-01,
         -8.8111e-01,  8.3864e-01, -9.9865e-01,  9.4263e-01,  9.9868e-01,
         -8.8270e-01, -9.9409e-01,  9.8626e-01, -3.7458e-01,  8.7057e-01,
         -9.6453e-01,  1.7986e-01, -9.9492e-01,  3.8073e-01, -8.0941e-01,
         -8.0715e-01, -9.3942e-01, -3.6658e-01,  8.5022e-01,  9.9879e-01,
          7.1673e-01, -2.7672e-01,  3.3562e-01,  2.1624e-01,  7.8492e-01,
         -8.0668e-01, -6.0987e-01, -3.7213e-01,  7.2645e-01, -2.2430e-01,
         -8.0414e-01, -8.9311e-01,  6.6076e-02, -8.0509e-01, -1.5963e-01,
          9.8860e-01, -9.6177e-01, -3.5406e-01,  2.6703e-01, -1.7358e-01,
          5.6831e-01,  9.7963e-01, -1.6021e-01, -1.1053e-01,  9.9873e-01,
         -1.8563e-01,  5.3422e-01,  9.8968e-01, -4.8954e-01,  8.9946e-01,
         -4.3556e-01,  7.5124e-01, -3.1500e-01,  5.9993e-01, -8.1074e-01,
          6.5921e-01, -9.6611e-01,  1.6641e-02,  9.9306e-01, -4.0454e-01,
          9.9834e-01, -9.9510e-01,  4.2360e-01, -9.9966e-01,  7.4263e-01,
          8.6341e-01, -1.1441e-01, -2.9153e-01,  8.5681e-01,  9.7878e-01,
          3.2724e-01,  1.2936e-01,  8.6568e-01, -3.3572e-01,  4.9202e-01,
          7.6250e-01,  8.4565e-01, -9.4228e-01,  1.0000e+00, -3.2622e-01,
          1.9432e-01,  3.2017e-01,  5.2906e-02, -1.4394e-01, -5.3216e-01,
          9.6838e-01, -9.9394e-01, -7.7518e-01,  2.1086e-01,  9.9598e-01,
          9.6457e-01, -7.1267e-01,  4.7877e-01,  9.9215e-01,  8.9403e-01,
          8.0095e-01, -7.8551e-01, -7.9392e-01,  2.7163e-01,  2.5308e-01,
          4.1577e-01,  9.6231e-01,  9.9139e-01, -9.5568e-01,  9.9738e-01,
          9.9997e-01, -7.6116e-01,  9.4853e-01, -5.1147e-01, -9.9407e-01,
         -9.6872e-01, -9.6058e-01,  7.3600e-01, -9.7144e-01, -8.5005e-01,
         -4.7326e-01,  8.3460e-01, -4.3085e-01,  8.2058e-01, -9.8892e-01,
         -3.2519e-02,  9.6791e-01, -1.8051e-01,  9.9865e-01,  8.0091e-01,
         -9.9622e-01,  8.5235e-02, -7.4423e-01,  9.5594e-01, -4.4680e-01,
          9.7042e-01,  6.6717e-01,  5.3039e-01,  7.9344e-01, -1.0000e+00,
         -6.2260e-01, -8.5597e-01, -8.8346e-01,  6.6213e-01,  9.8120e-01,
          8.2922e-01, -4.9337e-01, -1.1713e-01,  3.0851e-01,  9.9897e-01,
         -9.9157e-01, -4.5904e-01,  3.6395e-01, -9.7623e-01, -9.9546e-01,
          9.2914e-01, -5.9737e-02, -1.3186e-01, -5.6671e-01, -9.7951e-01,
         -3.1978e-02, -1.9426e-01,  9.7320e-01,  6.8251e-01,  9.6366e-01,
         -9.9692e-01,  7.0931e-01, -9.3518e-01,  3.8993e-01,  5.3098e-01,
          6.5294e-01, -5.5945e-01, -8.2407e-01,  2.5743e-01,  8.8606e-01,
         -9.2625e-01,  5.2883e-01, -6.8271e-01,  9.6566e-01,  4.0019e-01,
         -9.4268e-01,  3.6579e-01, -9.7778e-01,  9.9075e-01,  9.2130e-01,
          1.1096e-02,  9.9205e-01, -9.8311e-01, -3.8072e-01,  1.7588e-02,
         -7.8782e-02, -1.0000e+00, -5.7212e-01, -9.6962e-01,  9.1626e-01,
         -4.6427e-01,  9.8295e-01, -8.8938e-01, -5.1830e-01, -3.1417e-01,
         -9.9763e-01,  8.6589e-01, -3.0350e-01,  9.9275e-01, -8.9156e-01,
          4.7096e-01,  9.8986e-01,  2.2371e-01, -9.7230e-01, -9.9866e-01,
         -2.1644e-01,  9.9999e-01, -9.5194e-01, -1.6720e-01,  9.9833e-01,
          4.4628e-01, -9.1371e-01, -8.4491e-01, -9.9817e-01, -9.9372e-01,
         -7.7929e-01,  7.2473e-01, -2.6061e-01,  9.6549e-01, -4.1364e-01,
         -3.3484e-01,  9.6506e-01,  9.9997e-01, -5.5557e-01,  3.0508e-01,
          2.5703e-01, -9.0255e-01, -1.0000e+00,  2.8586e-01,  1.0007e-01,
         -9.9896e-01,  9.9678e-01, -9.9025e-01,  1.0000e+00, -7.7229e-01,
          7.0490e-01,  4.0055e-01, -1.7191e-01,  3.7641e-01,  1.8299e-01,
          9.9872e-01,  9.0641e-01, -7.1264e-01,  2.4002e-01, -1.3604e-01,
          4.3117e-01, -7.9883e-01,  8.6579e-01,  7.5893e-01,  2.9811e-02,
         -8.0524e-01, -6.6499e-01, -6.9537e-01, -9.8947e-01, -3.3042e-01,
          4.3933e-01,  7.7576e-01,  3.8784e-01,  8.3762e-01,  9.5521e-01,
         -1.9004e-01,  9.2374e-01, -4.5304e-01, -9.9971e-01,  4.7458e-01,
          3.9651e-01, -3.5551e-02,  7.6071e-01,  4.4238e-01,  9.8172e-01,
         -9.4636e-01,  9.9994e-01, -3.0790e-01, -3.7391e-02,  3.2913e-01,
          9.9944e-01, -9.9942e-01,  2.7112e-01, -7.4954e-01,  5.0199e-01,
          8.7535e-01,  9.7853e-01, -6.7640e-01, -3.8013e-01,  3.3398e-01,
          7.0454e-01,  8.3245e-01,  9.7294e-01,  7.9212e-02, -2.6529e-01,
          7.8871e-01,  8.6786e-01,  9.4053e-01, -7.0095e-01, -3.0060e-01,
         -7.5851e-01,  6.9017e-01,  8.7472e-01, -3.4388e-01,  7.3689e-01,
         -3.5747e-01,  5.9923e-01, -7.2205e-01, -2.3061e-01,  8.7991e-01,
          1.0714e-01,  3.0321e-01, -9.7453e-01, -7.8432e-01, -1.1791e-01,
          9.9615e-01, -9.6387e-01, -5.7347e-01,  9.8293e-01, -4.4668e-01,
          3.5105e-01,  7.7942e-01,  1.2009e-01, -7.9507e-01, -2.4365e-01,
         -9.9668e-01,  2.4649e-01, -7.2106e-01,  8.3978e-01,  9.3624e-01,
         -3.9950e-04,  4.2923e-01, -9.8484e-02, -8.3539e-01,  9.0447e-01,
          8.0184e-01,  9.5477e-01,  7.9382e-01, -4.5464e-01, -1.7763e-01,
          6.1225e-01,  4.2932e-01, -3.3037e-01,  9.8507e-01, -9.7384e-01,
          9.9928e-01, -3.8958e-01, -9.9858e-01,  3.4775e-01,  9.4594e-01,
         -9.9660e-01, -5.5604e-01, -9.9979e-01,  9.6688e-01, -8.5781e-01,
          3.3510e-01,  2.7752e-01, -9.9976e-01, -1.0000e+00, -6.1786e-01,
         -7.1510e-01, -5.9845e-01, -8.8065e-01,  9.9931e-01, -1.5013e-01,
          6.9268e-01,  1.6757e-01, -9.0341e-01,  7.2353e-01,  3.6488e-01,
         -7.4120e-01, -9.9837e-01,  8.1940e-01,  1.7401e-01,  8.2875e-01,
          4.7737e-02, -8.9452e-01,  7.9869e-01, -9.2593e-01,  7.2797e-01,
          8.8470e-01,  3.1619e-01,  9.1519e-01, -9.9573e-01,  9.7704e-01,
          4.8581e-04,  5.0231e-01, -6.3242e-01, -9.5180e-01,  9.9725e-01,
         -4.5625e-01,  3.1912e-02, -2.8593e-01, -9.9169e-01, -1.8685e-01,
          4.3439e-01, -1.9081e-01, -9.7653e-01,  6.5307e-01, -9.8700e-01,
         -9.9863e-01,  6.1528e-01, -5.1922e-02,  9.9961e-01,  9.3421e-01,
          7.4328e-01, -3.7321e-01, -8.4068e-01,  3.7956e-01, -9.9908e-01,
         -8.2668e-01, -3.5086e-01, -9.8303e-01,  9.5587e-01,  9.8434e-01,
          9.8534e-01,  1.0319e-01, -2.8329e-01, -8.4137e-01,  6.5210e-01,
          8.2347e-01,  8.9755e-01, -7.3321e-01,  1.1672e-01, -2.4463e-01,
         -9.6927e-01, -7.7289e-01,  9.9372e-01,  2.6576e-01,  9.9222e-01,
          2.0846e-01,  5.5293e-02,  7.8389e-01, -1.9877e-01,  1.7270e-01,
         -9.9999e-01, -7.1305e-01, -1.6109e-01, -9.9763e-01,  9.9687e-01,
         -9.9821e-01, -6.0273e-01,  9.0868e-01, -3.7720e-01,  9.6717e-01,
          5.0573e-01, -9.9838e-01, -9.9854e-01, -8.1627e-01, -7.2943e-01,
          9.8398e-01,  1.0566e-01,  7.2732e-01,  8.9038e-01,  4.5581e-01,
          9.9855e-01,  7.3846e-02,  9.1206e-01,  2.7309e-01,  9.9602e-01,
          5.8060e-01, -9.9274e-01,  4.1190e-01, -9.9665e-01, -5.9379e-02,
          8.7270e-01,  9.5225e-01,  9.6707e-01,  6.4702e-02,  9.9786e-01,
         -9.9881e-01,  9.9980e-01, -9.9901e-01,  3.2120e-01,  9.9833e-01,
         -9.6491e-01,  7.7830e-01, -9.9351e-01, -1.3290e-01, -6.9941e-01,
          1.0131e-01, -6.5945e-01,  9.8204e-01, -9.9136e-01, -9.9571e-01,
          7.5045e-01,  7.0624e-01,  8.5628e-01,  2.5740e-02,  6.5194e-01,
          9.9302e-01,  3.4794e-01,  9.0977e-01, -6.6819e-01,  1.1719e-01,
          1.0000e+00, -3.0428e-01,  8.6897e-01, -9.8649e-01,  9.1785e-01,
          7.7089e-01,  2.4319e-01,  8.3687e-01, -4.0462e-01,  8.2663e-01,
          9.0525e-01, -9.9394e-01, -5.3716e-01, -9.9525e-01, -4.5950e-01,
         -8.3593e-01,  2.5181e-01,  3.8147e-01,  1.1028e-01,  1.4456e-01,
         -9.9033e-01,  5.2838e-01, -9.8689e-01,  9.6708e-01, -2.4787e-01,
          2.2549e-01, -1.0193e-01,  8.0896e-01, -8.1674e-01,  9.9353e-01,
          9.9624e-01, -1.0000e+00,  3.5136e-01,  9.6999e-01,  7.7362e-01,
          9.3969e-01, -9.7267e-01, -2.8364e-01,  7.2528e-01, -4.2767e-01,
          9.5374e-01,  6.0657e-01, -1.6370e-01,  9.2809e-01, -9.8946e-01,
          3.6309e-01, -8.7984e-01, -1.8775e-01,  8.6614e-01, -9.6848e-01,
          4.3423e-01, -2.4693e-01,  6.8089e-01, -9.9728e-01, -5.9312e-01,
         -9.9801e-01, -3.4307e-01,  9.8755e-01,  8.9210e-01,  9.9812e-01,
          8.3189e-01, -6.5794e-01,  1.3123e-01, -9.9729e-01,  3.4493e-01,
          4.5963e-01, -3.8609e-01,  6.9156e-01, -4.1653e-01,  1.3413e-01,
         -8.3398e-01, -9.9797e-01,  4.1617e-01,  3.7179e-02,  3.6552e-01,
          7.8557e-01,  8.9680e-01, -9.6292e-01, -5.3080e-01, -8.3800e-01,
          5.0900e-01, -8.0170e-01, -9.1937e-01,  7.0140e-01,  3.2181e-01,
          9.9888e-01, -9.8983e-01, -8.5632e-01, -8.1569e-01, -7.9556e-01,
         -1.9110e-01,  5.1877e-01,  4.5526e-01,  3.9040e-01, -4.7427e-01,
         -4.8767e-01,  9.5301e-01, -9.8456e-01, -9.7645e-01,  9.9395e-01,
         -5.7962e-01,  4.5322e-01, -3.3631e-01, -6.6802e-01, -6.6034e-01,
         -3.2427e-01,  9.7156e-02, -7.2266e-01, -5.3495e-01, -9.9992e-01,
         -7.9082e-01,  8.0444e-01, -9.8494e-01, -8.3945e-01, -6.3249e-01,
         -1.0000e+00,  9.8912e-01,  7.9101e-01,  9.9693e-01, -9.9844e-01,
          8.8868e-01,  3.6596e-01,  9.9451e-01, -2.2086e-01, -7.1058e-01,
          5.2474e-01,  9.9276e-01,  7.5135e-01, -9.4949e-01,  2.5712e-01,
         -4.3321e-01, -3.6153e-01, -8.7408e-01, -5.7381e-01,  3.3346e-01,
          8.7240e-01, -9.6141e-01, -9.9465e-01,  9.9710e-01, -3.0994e-01,
          9.7845e-01,  5.0702e-01, -1.9111e-01, -9.0967e-01,  2.6708e-01,
          2.1764e-01, -8.1679e-01, -9.9814e-01,  9.1552e-01, -1.0000e+00,
         -9.7636e-01,  4.9271e-01,  9.7815e-01, -9.8669e-01, -9.5185e-01,
          9.1136e-02, -9.9923e-01, -5.8831e-01,  2.8123e-02,  8.1246e-01,
         -9.8216e-01, -1.2316e-01, -6.7497e-01,  8.0479e-01,  9.7376e-01,
         -9.3756e-01, -3.6583e-01, -4.2700e-01,  9.9874e-01, -3.1676e-02,
          6.0804e-01,  3.1509e-01,  4.6395e-01,  4.2325e-01, -8.5924e-01,
         -5.0233e-01, -9.8136e-01, -8.6662e-01,  9.8571e-01,  9.4677e-01,
         -9.9560e-01, -9.8801e-01, -2.1241e-01,  7.8756e-02,  9.8048e-01,
         -5.7331e-01, -9.9814e-01, -9.9842e-01, -8.0677e-02, -7.2799e-01,
          9.8951e-01, -2.2606e-01,  1.0000e+00,  8.5388e-01,  5.3913e-01,
         -7.6908e-01,  9.2467e-01,  5.8919e-01,  7.9964e-01, -5.8185e-01,
          9.9935e-01,  7.9794e-01,  9.9035e-01],
        Atom(b45, [
            Structure(s, [List(['o', 'x', one, space2, three, 'l', 'z']),
                          List([O2, X2, one, three, L2, Z2])])])
        : [-5.3516e-01,  7.2749e-01,  9.9933e-01, -9.9390e-01,  9.0046e-01,
         -8.0194e-01,  9.8000e-01,  9.4790e-01, -9.7228e-01, -9.4125e-01,
          9.8726e-01,  9.9614e-01,  7.2579e-01, -9.9862e-01, -8.6589e-01,
         -9.8480e-01,  9.7627e-01, -4.8602e-01, -9.9978e-01,  5.6692e-01,
          3.7754e-01, -9.9850e-01,  4.8255e-01,  3.3425e-01,  9.6429e-01,
          4.2530e-01,  9.9373e-01,  9.9899e-01,  9.2589e-01,  9.8441e-01,
          2.7945e-01, -9.8957e-01,  3.7020e-01, -9.9425e-01,  7.4784e-02,
         -8.2617e-01, -1.5299e-01, -1.0387e-01, -5.7503e-02,  4.9123e-01,
         -9.4198e-01,  9.3287e-01, -8.4583e-01, -3.0859e-01, -9.1071e-01,
          4.6868e-01,  7.5048e-01, -6.3254e-03, -3.4893e-01,  9.9992e-01,
         -9.7783e-01,  1.0000e+00,  8.4565e-01,  9.9998e-01,  9.9642e-01,
          8.6419e-01,  9.9363e-01,  3.6637e-01,  7.4974e-01,  9.2986e-01,
          9.8813e-01, -5.7905e-01,  9.2045e-01, -6.5496e-01, -8.1403e-01,
         -9.0461e-01,  5.8311e-01,  4.1764e-01, -8.3992e-01,  9.2052e-01,
          7.8434e-01,  5.2441e-01,  9.9476e-01, -9.5890e-01,  3.5283e-01,
         -9.5242e-01,  6.9643e-01, -9.9949e-01,  9.7435e-01,  9.9888e-01,
         -9.4491e-01, -9.9636e-01,  9.9455e-01, -3.9531e-01,  7.1699e-01,
         -9.8561e-01,  5.8920e-01, -9.9663e-01,  4.0985e-01, -9.1997e-01,
         -9.4233e-01, -9.8109e-01, -3.2727e-01,  8.8683e-01,  9.9932e-01,
          9.0749e-01, -1.4043e-01,  4.7372e-01, -3.6595e-01,  9.2000e-01,
         -9.3021e-01, -8.1714e-01, -8.1229e-01,  9.1421e-01, -3.3115e-01,
         -7.4236e-01, -9.7361e-01,  4.4305e-01, -9.7941e-01,  1.6298e-02,
          9.9641e-01, -9.8795e-01, -1.4823e-01,  4.2929e-01, -8.0156e-01,
          4.7441e-01,  9.8368e-01, -2.3237e-01, -2.0803e-02,  9.9930e-01,
         -3.9814e-01,  1.7945e-01,  9.9292e-01, -6.9656e-01,  9.4512e-01,
         -5.5148e-01,  6.2975e-01, -5.9442e-01,  6.5176e-01, -9.1612e-01,
          7.2624e-01, -9.8532e-01,  7.2600e-01,  9.9563e-01, -3.3707e-01,
          9.9895e-01, -9.9822e-01, -1.0042e-01, -9.9986e-01,  7.1197e-01,
          9.2098e-01, -9.6429e-02,  8.3550e-02,  8.4720e-01,  9.8875e-01,
          4.1491e-01,  8.2425e-01,  9.5693e-01, -7.4418e-01,  7.2840e-01,
          9.0132e-01,  9.3298e-01, -9.8447e-01,  1.0000e+00, -7.1178e-01,
         -2.5900e-01, -1.7029e-01,  1.4333e-01,  2.8792e-01, -7.5388e-01,
          9.8094e-01, -9.9394e-01, -8.3062e-01,  7.3731e-01,  9.9758e-01,
          9.9080e-01, -8.2703e-01,  7.9915e-01,  9.9585e-01,  9.4197e-01,
          7.4722e-01, -8.4234e-01, -8.7889e-01,  6.9425e-01,  4.4807e-01,
          4.2108e-01,  9.7351e-01,  9.9857e-01, -9.8372e-01,  9.9884e-01,
          1.0000e+00, -8.8500e-01,  9.6680e-01, -7.7978e-01, -9.9616e-01,
         -9.9015e-01, -9.8437e-01,  7.9334e-01, -9.8029e-01, -8.8622e-01,
         -7.8075e-01,  9.5776e-01, -8.1884e-01,  9.0008e-01, -9.9534e-01,
         -9.2298e-02,  9.7828e-01, -2.2685e-01,  9.9938e-01,  8.4307e-01,
         -9.9749e-01,  4.5450e-01, -8.8517e-01,  9.4072e-01, -4.5405e-01,
          9.8058e-01,  7.9554e-01,  7.0406e-01,  1.1410e-01, -1.0000e+00,
         -8.0974e-01, -7.5899e-01, -9.6177e-01,  8.3374e-01,  9.9292e-01,
          8.7205e-01, -5.7568e-01, -7.7216e-02,  6.1217e-01,  9.9938e-01,
         -9.9311e-01, -4.3382e-01,  7.3796e-01, -9.9385e-01, -9.9620e-01,
          9.8040e-01, -1.7401e-01,  7.0152e-01, -5.6988e-01, -9.9535e-01,
          3.7152e-02, -6.2336e-01,  9.8792e-01,  7.6320e-01,  9.8642e-01,
         -9.9912e-01,  9.0580e-01, -9.7862e-01,  5.5019e-01,  4.0452e-01,
          6.5110e-01, -6.1425e-01, -9.0599e-01,  7.7398e-01,  9.3817e-01,
         -8.9750e-01,  7.4334e-01, -8.5058e-01,  9.7747e-01,  8.3103e-01,
         -9.0232e-01,  5.8063e-01, -9.8762e-01,  9.9514e-01,  9.3485e-01,
         -5.1548e-01,  9.9587e-01, -9.9412e-01, -4.0574e-01,  7.1357e-01,
          1.1177e-02, -1.0000e+00, -7.2754e-01, -9.8125e-01,  9.5433e-01,
         -4.3344e-01,  9.8865e-01, -9.6546e-01,  1.0115e-01, -6.5078e-01,
         -9.9889e-01,  9.6590e-01, -5.3241e-01,  9.9365e-01, -8.4721e-01,
          6.4123e-01,  9.9723e-01, -1.0951e-01, -9.8829e-01, -9.9940e-01,
         -4.8935e-01,  1.0000e+00, -9.8468e-01, -1.8205e-01,  9.9916e-01,
          8.3661e-01, -9.5446e-01, -9.7320e-01, -9.9846e-01, -9.9654e-01,
         -6.8142e-01,  5.9447e-01, -2.3271e-01,  9.9285e-01, -5.3163e-01,
         -3.4931e-01,  9.7961e-01,  1.0000e+00, -5.7180e-01,  5.1191e-01,
          3.0558e-01, -9.4497e-01, -1.0000e+00,  6.5496e-02,  8.7673e-02,
         -9.9967e-01,  9.9741e-01, -9.8167e-01,  1.0000e+00, -9.4632e-01,
          8.6488e-01,  8.2595e-01, -2.4272e-01,  4.3508e-01,  3.4097e-01,
          9.9927e-01,  9.7474e-01, -6.3608e-01,  2.7131e-01, -3.7814e-01,
          4.8418e-01, -9.2453e-01,  9.6660e-01,  7.8984e-01,  3.4849e-01,
         -8.8369e-01, -7.3760e-01, -8.4418e-01, -9.9412e-01, -6.4754e-01,
          4.1516e-01,  9.4503e-01,  7.8384e-01,  9.1473e-01,  9.8936e-01,
         -4.5677e-01,  8.9626e-01, -4.9737e-01, -9.9996e-01,  8.1946e-01,
          5.8483e-01,  3.4501e-01,  7.7924e-01,  6.4394e-01,  9.9313e-01,
         -9.5143e-01,  1.0000e+00, -3.9005e-01, -1.4627e-01,  6.8429e-01,
          9.9972e-01, -9.9996e-01,  5.1937e-01, -7.9396e-01,  7.6683e-01,
          9.2058e-01,  9.9147e-01, -7.6313e-01, -5.3178e-01,  5.0187e-01,
          6.9289e-01,  9.3863e-01,  9.9167e-01,  2.1876e-01, -2.0637e-01,
          9.3216e-01,  9.2116e-01,  9.7995e-01, -8.9976e-01, -3.8439e-01,
         -8.0406e-01,  8.6134e-01,  9.4424e-01,  8.5297e-02,  7.7088e-01,
         -2.9775e-01,  7.0182e-01, -7.1355e-01, -6.9432e-01,  9.0042e-01,
         -1.4332e-01,  4.6102e-01, -9.9398e-01, -9.3565e-01, -3.9850e-01,
          9.9821e-01, -9.8066e-01, -7.8338e-01,  9.9005e-01, -5.7054e-01,
          5.1496e-01,  8.8328e-01, -5.1863e-01, -9.0392e-01, -2.5887e-01,
         -9.9855e-01,  5.9712e-02, -6.0855e-01,  6.7467e-01,  9.7199e-01,
          8.0755e-02,  5.0575e-01, -3.4761e-01, -9.4982e-01,  9.2930e-01,
          8.0668e-01,  9.7710e-01,  9.5923e-01, -5.0938e-01, -4.5578e-01,
          7.8884e-01,  5.6574e-01, -5.4427e-01,  9.9515e-01, -9.8977e-01,
          9.9963e-01, -7.2025e-01, -9.9841e-01,  8.8110e-01,  9.5210e-01,
         -9.9960e-01, -3.5803e-01, -9.9996e-01,  9.8280e-01, -9.4848e-01,
          8.6624e-01,  7.4789e-01, -9.9999e-01, -1.0000e+00, -7.3552e-01,
         -7.3074e-01, -4.0319e-01, -9.1289e-01,  9.9990e-01, -6.4810e-02,
          8.4845e-01, -7.2995e-04, -9.7051e-01,  8.1983e-01,  7.3490e-01,
         -7.1329e-01, -9.9913e-01,  8.3491e-01, -1.2811e-01,  9.3642e-01,
          2.7165e-01, -9.4612e-01,  9.3485e-01, -9.5950e-01,  7.4195e-01,
          9.7273e-01,  6.4216e-01,  9.5573e-01, -9.9781e-01,  9.8941e-01,
          1.5081e-01,  5.2339e-01, -3.1729e-01, -9.8557e-01,  9.9913e-01,
         -9.1871e-01,  2.6095e-01, -4.8561e-01, -9.9494e-01, -3.1005e-01,
          4.8372e-01,  8.7664e-02, -9.8689e-01,  4.4127e-01, -9.9216e-01,
         -9.9962e-01,  5.2050e-01, -5.4509e-01,  9.9995e-01,  9.7801e-01,
          8.5164e-01, -4.5214e-01, -9.2106e-01,  3.3804e-01, -9.9958e-01,
         -8.7476e-01, -5.7244e-01, -9.8805e-01,  9.6388e-01,  9.9157e-01,
          9.9465e-01,  7.2249e-01,  3.5601e-02, -9.6970e-01,  8.1288e-01,
          9.0272e-01,  9.5745e-01, -7.8654e-01,  2.3179e-01, -3.1333e-01,
         -9.8806e-01, -9.0298e-01,  9.9576e-01,  8.2529e-01,  9.9016e-01,
         -5.1074e-01, -3.7299e-01,  7.3469e-01, -3.6486e-01,  4.8374e-01,
         -1.0000e+00, -9.1509e-01, -4.7654e-01, -9.9858e-01,  9.9857e-01,
         -9.9940e-01, -7.8037e-01,  9.6340e-01, -6.2914e-01,  9.7904e-01,
          7.2844e-01, -9.9944e-01, -9.9925e-01, -8.7675e-01, -6.6773e-01,
          9.8876e-01, -1.7115e-02,  6.2493e-01,  9.1593e-01,  8.6319e-01,
          9.9851e-01, -2.1772e-02,  9.2898e-01,  8.9093e-01,  9.9838e-01,
          6.8627e-01, -9.9404e-01,  1.0605e-01, -9.9841e-01, -3.9460e-01,
          9.6180e-01,  9.8085e-01,  9.8893e-01,  7.3851e-01,  9.9936e-01,
         -9.9937e-01,  9.9997e-01, -9.9948e-01,  7.8982e-01,  9.9919e-01,
         -9.8429e-01,  8.5638e-01, -9.9543e-01,  4.8588e-01, -8.9237e-01,
          1.8537e-01, -8.1776e-01,  9.9413e-01, -9.9249e-01, -9.9423e-01,
          6.6900e-01,  7.4761e-01,  9.5457e-01, -4.7911e-01,  8.1381e-01,
          9.9670e-01,  3.1395e-01,  9.7036e-01, -5.9206e-01, -5.2870e-01,
          1.0000e+00,  2.5627e-01,  8.4325e-01, -9.9508e-01,  9.7943e-01,
          8.0568e-01,  4.5076e-01,  9.3196e-01, -4.0329e-01,  9.2888e-01,
          8.7647e-01, -9.9269e-01, -5.0953e-01, -9.9852e-01, -7.4998e-01,
         -7.8367e-01,  6.3886e-01,  4.8042e-01,  3.5978e-01,  5.2394e-01,
         -9.9597e-01,  3.2676e-01, -9.9644e-01,  9.9115e-01,  4.3144e-01,
          2.0746e-01, -3.1052e-01,  8.7725e-01, -8.8179e-01,  9.9570e-01,
          9.9819e-01, -1.0000e+00,  5.4636e-01,  9.8908e-01,  9.2665e-01,
          9.8052e-01, -9.8754e-01, -4.2056e-01,  4.2014e-01, -5.9263e-01,
          9.8882e-01,  7.7581e-01, -3.0311e-01,  9.7614e-01, -9.9411e-01,
          6.8095e-01, -9.3772e-01, -4.0522e-01,  9.6375e-01, -9.8406e-01,
          4.5761e-01, -7.2925e-01,  7.4093e-01, -9.9838e-01, -8.8475e-01,
         -9.9844e-01, -4.8099e-01,  9.9446e-01,  8.0408e-01,  9.9925e-01,
          8.6579e-01, -5.9896e-01, -4.6222e-02, -9.9800e-01,  7.5824e-01,
          4.3940e-01, -3.3860e-01,  7.5622e-01, -8.0663e-01,  5.2719e-01,
         -9.3307e-01, -9.9924e-01,  4.6475e-01, -4.9860e-01,  1.7627e-01,
          9.4798e-01,  9.0635e-01, -9.8816e-01, -2.5411e-01, -8.9890e-01,
          6.6353e-01, -8.4700e-01, -9.4058e-01,  7.2015e-01,  8.1928e-02,
          9.9956e-01, -9.9408e-01, -9.3867e-01, -9.2399e-01, -8.4149e-01,
         -3.0403e-01,  4.3427e-01,  2.6088e-01,  8.0798e-01, -8.1363e-01,
         -2.1804e-01,  9.8694e-01, -9.9049e-01, -9.8197e-01,  9.9701e-01,
         -6.3155e-01,  7.2044e-01, -4.3334e-01, -7.9246e-01, -6.9520e-01,
         -3.4019e-01, -3.5149e-02, -7.3135e-01, -5.9011e-01, -9.9999e-01,
         -8.4888e-01,  7.7273e-01, -9.8961e-01, -9.6015e-01, -7.5475e-01,
         -1.0000e+00,  9.9488e-01,  8.9388e-01,  9.9861e-01, -9.9918e-01,
          9.6377e-01,  6.1416e-01,  9.9654e-01, -1.6688e-01, -8.3547e-01,
          4.0321e-01,  9.9780e-01,  8.7008e-01, -8.7960e-01,  1.4634e-01,
         -4.6574e-01, -5.2255e-01, -8.1456e-01, -8.3869e-01,  5.2941e-01,
          8.9175e-01, -9.7651e-01, -9.9883e-01,  9.9867e-01, -2.4277e-01,
          9.9173e-01,  6.1860e-01, -3.8992e-02, -9.7193e-01,  2.2085e-01,
          5.5662e-01, -9.4907e-01, -9.9927e-01,  9.4115e-01, -1.0000e+00,
         -9.8768e-01,  6.2152e-01,  9.9248e-01, -9.9606e-01, -9.6831e-01,
          3.6667e-01, -9.9946e-01, -7.1688e-01,  5.4561e-01,  8.6243e-01,
         -9.9194e-01, -7.0635e-01, -8.5324e-01,  8.5258e-01,  9.8305e-01,
         -9.7949e-01, -4.5659e-01, -5.8726e-01,  9.9970e-01, -1.1044e-01,
          6.0605e-01,  4.7935e-01,  6.0106e-01,  5.9166e-01, -9.5002e-01,
         -8.2557e-01, -9.9319e-01, -9.5819e-01,  9.9286e-01,  9.8154e-01,
         -9.9778e-01, -9.9136e-01, -5.1146e-01, -2.0669e-01,  9.9333e-01,
         -7.5991e-01, -9.9932e-01, -9.9866e-01, -1.7530e-01, -8.4902e-01,
          9.9478e-01, -4.1331e-01,  1.0000e+00,  9.3143e-01,  7.5451e-01,
         -8.5792e-01,  8.9188e-01,  7.2253e-01,  8.7701e-01, -3.2193e-01,
          9.9971e-01,  8.6736e-01,  9.9525e-01]
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
    learner = MyOwnLeaner(prolog,encoding2)

    decoder = decoding.decoding(primitives)
    network1 = Network.Network.LoadNetwork('network1.txt')
    network2 = Network.Network.LoadNetwork('network2.txt')
    print(network1.GetNumOutputs())

    program = learner.learn(task, background, hs,network1,network2,encodingExamples,decoder,primitives)

    print(program)


