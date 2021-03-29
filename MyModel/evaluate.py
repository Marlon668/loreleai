import typing
from itertools import product, combinations_with_replacement
from abc import ABC, abstractmethod

from orderedset import OrderedSet
import time

from pylo.engines.prolog.SWIProlog import Pair
from loreleai.language.lp import c_var, c_pred, Clause, Atom, Body, Not, List, c_const, Structure,Variable, Type
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_singleton_vars, has_duplicated_literal
from loreleai.learning.language_manipulation import plain_extension
from loreleai.learning.task import Task, Knowledge
from loreleai.reasoning.lp.prolog import SWIProlog, Prolog
import random
from MyModel import encoding2
from MyModel import Network
import numpy as np

# Class for generating input/output examples
class evaluate:
    def __init__(self, solver_instance: Prolog,knowledge: Knowledge, negative_examples, hs:TopDownHypothesisSpace, primitives,encodingProblems):
        self._solver = solver_instance
        self._negative_examples = negative_examples
        self._primitives= primitives
        self._knowledge = knowledge
        self._assert_knowledge(knowledge)
        self._hs = hs
        self._encodingProblems = encodingProblems

    # generate best clauses
    def generateExamples(self, mostgenericclause,positive_examples,k,encoder:encoding2):
        states = self.generateStates()
        self._number = -1
        input = []
        output = []
        self._input2 = []
        self._output2 = []
        self._encoder = encoder
        numberState = 0
        self._numberOfEvaluations = 0
        network1 = Network.Network.LoadNetwork('network1.txt')
        network2 = Network.Network.LoadNetwork('network2.txt')

        print(states)
        for state in [1]:
            state = states[4]
            numberIO = 0
            print(state)
            self._encodingState = self.encodeState(state)
            trainingprograms = [mostgenericclause]
            positive_examples_state = []
            for i in range(0, len(state)):
                if state[i] == 1:
                    positive_examples_state.append(positive_examples[i])
            while (len(trainingprograms)!=0):
                self._number += 1
                clause = trainingprograms.pop()
                clause = clause[0]
                expansions = self.expand(clause,positive_examples_state)
                if len(clause.get_literals())<4:
                    bestk = self.pickKBest(sorted(expansions[1],key=lambda l: l[1],reverse=True),k)
                    for training in bestk:
                        trainingprograms.append(training)
                else:
                    if len(clause.get_literals()) < 5:
                        bestk = self.pickKBest(sorted(expansions[1], key=lambda l: l[1], reverse=True), 2)
                        for training in bestk:
                            trainingprograms.append(training)
                if len(clause.get_body().get_literals()) != 0:
                    encode = [*encoder.encode(clause),*self._encodingState]
                    input.append(encode)
                    output.append(expansions[0])

                if (self._number % 49 == 0 ) &(len(input)>0):
                    print('train 1')
                    network1.Train(input, output, 0.1)
                    network1.SaveNetwork('network1.txt')
                    toc = time.perf_counter()
                    input = []
                    output = []
                if (self._number % 500 == 0) &(len(input)>0):
                    print('train 1')
                    network1.Train(input, output, 0.1)
                    network1.SaveNetwork('network1.txt')
                    print('train 2')
                    network2.Train(self._input2, self._output2, 0.1)
                    network2.SaveNetwork('network2.txt')
                    print('train 2')
                    self._input2 = []
                    self._output2 = []
                    
                if len(self._input2)>10000:
                    print('train 2')
                    network2.Train(self._input2, self._output2, 0.1)
                    network2.SaveNetwork('network2.txt')
                    self._input2= []
                    self._output2 = []
                numberIO += 1
                print('number of I/O : ' , self._number)
                print('number of evaluations: ', len(self._input2))
            print(len(trainingprograms))
        network1.SaveNetwork('network1.txt')
        network2.SaveNetwork('network2.txt')

    #generate worst clauses
    def generateExamples2(self, mostgenericclause, positive_examples, k, encoder: encoding2):
        states = self.generateStates()
        self._number = -1
        input = []
        output = []
        self._input2 = []
        self._output2 = []
        self._encoder = encoder
        self._numberOfEvaluations = 0
        network1 = Network.Network.LoadNetwork('network1.txt')
        network2 = Network.Network.LoadNetwork('network2.txt')
        print(states)
        for state in states:
            numberIO = 0
            print(state)
            self._encodingState = self.encodeState(state)
            trainingprograms = [mostgenericclause]
            positive_examples_state = []
            for i in range(0, len(state)):
                if state[i] == 1:
                    positive_examples_state.append(positive_examples[i])
            while (len(trainingprograms) != 0):
                self._number += 1
                clause = trainingprograms.pop()
                value = clause[1]
                clause = clause[0]
                expansions = self.expand(clause, positive_examples_state)
                if len(clause.get_literals()) < 4:
                    worstk = self.pickKWorst(sorted(expansions[1], key=lambda l: l[1]), k)
                    for training in worstk:
                        trainingprograms.append(training)
                else:
                    if len(clause.get_literals()) < 5:
                        worstk = self.pickKWorst(sorted(expansions[1], key=lambda l: l[1]), 2)
                        for training in worstk:
                            trainingprograms.append(training)
                if len(clause.get_body().get_literals()) != 0:
                    encode = [*encoder.encode(clause), *self._encodingState]
                    input.append(encode)
                    output.append(expansions[0])
                if (self._number % 49 == 0) & (len(input) > 0):
                    print('train 1')
                    network1.Train(input, output, 0.1)
                    network1.SaveNetwork('network1.txt')
                    toc = time.perf_counter()
                    input = []
                    output = []
                if (self._number % 500 == 0) & (len(input) > 0):
                    print('train 1')
                    network1.Train(input, output, 0.1)
                    network1.SaveNetwork('network1.txt')
                    print('train 2')
                    network2.Train(self._input2, self._output2, 0.1)
                    network2.SaveNetwork('network2.txt')
                    print('train 2')
                    self._input2 = []
                    self._output2 = []

                if len(self._input2) > 10000:
                    print('train 2')
                    network2.Train(self._input2, self._output2, 0.1)
                    network2.SaveNetwork('network2.txt')
                    self._input2 = []
                    self._output2 = []
                numberIO += 1
                print('number of I/O : ', self._number)
                print('number of evaluations: ', len(self._input2))
            print(len(trainingprograms))
        network1.SaveNetwork('network1.txt')
        network2.SaveNetwork('network2.txt')

    def encodeState(self,state):
        number = 0
        encoding = []
        for i in range(0,len(state)):
            if(state[i]==1):
                if(number == 0):
                    encoding = self._encodingProblems[i+1]
                    print(self._encodingProblems[i+1])
                    print(encoding)
                    number += 1
                else:
                    encoding = [a + b for a, b in zip(encoding, self._encodingProblems[i+1])]
                    print(encoding)
                    print(self._encodingProblems[i+1])
                    number +=1
        print(np.concatenate((number,np.divide(encoding,number)),axis=None))
        return np.concatenate((number,np.divide(encoding,number)),axis=None)

    def pickKBest(self,expansions,k):
        bestk = []
        best = expansions[0][1]
        last = []
        if(best==0):
            return bestk
        for clause in expansions :
            if clause[1] == best:
                last.append(clause)
            else:
                if len(bestk) + len(last) <= k :
                    bestk = bestk + last
                    if (len(bestk)==k) | (clause[1]<=0 ):
                        return bestk
                    last = [clause]
                    best = clause[1]
                else:
                    l = k -len(bestk)
                    bestk = bestk + random.sample(last,l)
                    return bestk
        l = k-len(bestk)
        bestk = bestk + random.sample(last, l)
        print(bestk)
        return bestk

    def pickKWorst(self,expansions,k):
        worstk = []
        worst = expansions[0][1]
        if worst == 0:
            different = 1
        else:
            different = 0
        last = []
        for clause in expansions :
            if clause[1] == worst:
                last.append(clause)
            else:
                if different == 1:
                    worstk.append(random.sample(last,1))
                    different = 0
                else:
                    if len(worstk) + len(worstk) <= k :
                        worstk= worstk + last
                        if (len(worstk)==k) & different == 0 :
                            return worstk
                        last = [clause]
                        worst = clause[1]
                    else:
                        l = k -len(worstk)
                        worstk = worstk+ random.sample(last,l)
                        return worstk
        l = k-len(worstk)
        worstk = worstk+ random.sample(last, l)
        return worstk

    def generateStates(self):
        states = [[1,1,1,1,1]]
        newstate = [0,0,0,0,0]
        newstate[random.randint(0,4)] = 1
        states.append(newstate)
        for i in range(0,1):
            newstate = [0, 0, 0, 0, 0]
            index1 = random.randint(0, 4)
            index2 = random.randint(0, 4)
            while index2 == index1:
                index2 = random.randint(0, 4)
            newstate[index1] = 1
            newstate[index2] = 1
            states.append(newstate)
        for i in range(0,1):
            newstate = [1, 1, 1, 1, 1]
            index1 = random.randint(0,4)
            index2 = random.randint(0,4)
            while index2 == index1:
                index2 = random.randint(0, 4)
            newstate[index1] = 0
            newstate[index2] = 0
            states.append(newstate)
        for i in range(0,1):
            newstate = [1, 1, 1, 1, 1]
            newstate[random.randint(0, 4)] = 0
            states.append(newstate)
        return states

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

    def expand(self,clause,positive_examples):
        encoding = []
        expansions = []
        for atom in self._primitives:
            newEncoding = self.expandClause(clause, positive_examples, atom, 3)
            encoding= encoding + [newEncoding[0]]
            expansions = expansions + newEncoding[1]
        return [encoding,expansions]

    def _evaluateClause(self, clause,positive_examples):
        if self._numberOfEvaluations<90000:
            self._numberOfEvaluations +=1
            numberofpositivecoverance = 0
            self._solver.assertz(clause)
            for example in positive_examples:
                if self._solver.has_solution(example):
                    numberofpositivecoverance +=1
            numberofnegativecoverance = 0
            for example in self._negative_examples:
                if self._solver.has_solution(example):
                    numberofnegativecoverance += 1
            self._solver.retract(clause)
            if numberofnegativecoverance + numberofpositivecoverance == 0:
                y = 0
            else:
                y = numberofpositivecoverance/(numberofpositivecoverance+numberofnegativecoverance)*(numberofpositivecoverance)/len(positive_examples)
            return y
        else:
            self._numberOfEvaluations = 0
            self._solver.release()
            self._solver = SWIProlog()
            self._assert_knowledge(self._knowledge)
            self._numberOfEvaluations += 1
            numberofpositivecoverance = 0
            self._solver.assertz(clause)
            for example in positive_examples:
                if self._solver.has_solution(example):
                    numberofpositivecoverance += 1
            numberofnegativecoverance = 0
            for example in self._negative_examples:
                if self._solver.has_solution(example):
                    numberofnegativecoverance += 1
            self._solver.retract(clause)
            if numberofnegativecoverance + numberofpositivecoverance == 0:
                return 0
            else:
                return numberofpositivecoverance / (numberofpositivecoverance + numberofnegativecoverance) * (
                    numberofpositivecoverance) / len(positive_examples)

    def expandClause(self,clause,positive_examples,atom,numberBest):
        expansions = plain_extension(clause._body,atom)
        x = []
        minimum = 0
        bestexp = []
        for exp in expansions:
            addedLiteral = exp.get_literals()[-1]
            if (not has_duplicated_literal(clause.get_head(),exp)) & (len(addedLiteral.get_variables()) == len(set(addedLiteral.get_variables()))):
                y = self._evaluateClause(Clause(clause.get_head(), exp),positive_examples)
                if len(x)<numberBest:
                    x.append(y)
                    bestexp.append((Clause(clause.get_head(), exp),y))
                    if random.random()>0.6:
                        self._input2.append([*self._encoder.encode(Clause(clause.get_head(), exp)), *self._encodingState])
                        self._output2.append([y])
                else:
                    if minimum == 0:
                        minimum = min(range(len(x)), key=x.__getitem__)
                    if x[minimum] < y:
                        bestexp[minimum] = ((Clause(clause.get_head(), exp), y))
                        x[minimum] = y
                        minimum = 0
        if len(x)==0:
            y = 0
        else:
            y = sum(x)/len(x)
        for exp in bestexp:
            self._input2.append([*self._encoder.encode(exp[0]), *self._encodingState])
            self._output2.append([y])
        return [y,bestexp]

    def getIndexOfMinimum(self,x):
        index = [0]
        for i in range(1,len(x)):
            if x[i]<x[index[0]]:
                index = [i]
            else:
                if x[i] == x[index[0]]:
                    index.append(i)
        return random.choice(index)
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

    def has_duplicated_literal(head: Atom, body: Body) -> bool:
        """
        Returns True if there are duplicated literals in the body
        """
        return len(body) != len(set(body.get_literals()))

    def writeEncodingsToFile(self,encoding,file):
        np.savetxt(file,encoding,fmt='%s')

    def readEncodingsOfFile(self,file):
        encoding = np.loadtxt(file, dtype=np.object)
        encoding = [[float(y) for y in x] for x in encoding]
        return encoding

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

    # positive examples
    pos = {Atom(b45, [
        Structure(s, [List([F2, 'c', E2, Q2, E2,'h','c',F2, C2, 'q']),
                      List([F2, E2, Q2, E2, F2, C2])])]),
           Atom(b45, [Structure(s, [List(
               ['o', N2, A2, 'z', 'g', 'f']),
               List(
                   [N2,A2])])]),
           Atom(b45, [Structure(s, [List([T2,'r','g','y',T2,P2]),
                                    List([T2,T2,P2])])]),
           Atom(b45, [
               Structure(s, [List(['c',O2,'o','j','j',H2,F2,M2,'g',C2]),
                             List([O2, H2, F2, M2, C2])])]),
           Atom(b45, [Structure(s, [List(['x','u','k','l','w','f',Z2,L2,R2,'h',U2,'t']),
                                    List([Z2,L2,R2,U2])])])}

    # negative examples
    neg = {Atom(b45, [
        Structure(s, [List(['e', 'd', 'i', 't', 'h']),
                      List(['e','i'])])]),
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
        Structure(s, [List([Fz, 'c', Ez, Qz, Ez,'h','c',Fz, Cz, 'q']),
                      List([Fz, Ez, Qz, Ez, Fz, Cz])])]),
           Atom(b45, [Structure(s, [List(
               ['o', Nz, Az, 'z', 'g', 'f']),
               List(
                   [Nz,Az])])]),
           Atom(b45, [Structure(s, [List([Tz,'r','g','y',Tz,Pz]),
                                    List([Tz,Tz,Pz])])]),
           Atom(b45, [
               Structure(s, [List(['c',Oz,'o','j','j',Hz,Fz,Mz,'g',Cz]),
                             List([Oz, Hz, Fz, Mz, Cz])])]),
           Atom(b45, [Structure(s, [List(['x','u','k','l','w','f',Zz,Lz,Rz,'h',Uz,'t']),
                                    List([Zz,Lz,Rz,Uz])])]),
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

                }# positive examples
    pos = {Atom(b45, [
        Structure(s, [List([F2, 'c', E2, Q2, E2,'h','c',F2, C2, 'q']),
                      List([F2, E2, Q2, E2, F2, C2])])]),
           Atom(b45, [Structure(s, [List(
               ['o', N2, A2, 'z', 'g', 'f']),
               List(
                   [N2,A2])])]),
           Atom(b45, [Structure(s, [List([T2,'r','g','y',T2,P2]),
                                    List([T2,T2,P2])])]),
           Atom(b45, [
               Structure(s, [List(['c',O2,'o','j','j',H2,F2,M2,'g',C2]),
                             List([O2, H2, F2, M2, C2])])]),
           Atom(b45, [Structure(s, [List(['x','u','k','l','w','f',Z2,L2,R2,'h',U2,'t']),
                                    List([Z2,L2,R2,U2])])])}

    # negative examples
    neg = {Atom(b45, [
        Structure(s, [List(['e', 'd', 'i', 't', 'h']),
                      List(['e','i'])])]),
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
        Structure(s, [List([Fz, 'c', Ez, Qz, Ez,'h','c',Fz, Cz, 'q']),
                      List([Fz, Ez, Qz, Ez, Fz, Cz])])]),
           Atom(b45, [Structure(s, [List(
               ['o', Nz, Az, 'z', 'g', 'f']),
               List(
                   [Nz,Az])])]),
           Atom(b45, [Structure(s, [List([Tz,'r','g','y',Tz,Pz]),
                                    List([Tz,Tz,Pz])])]),
           Atom(b45, [
               Structure(s, [List(['c',Oz,'o','j','j',Hz,Fz,Mz,'g',Cz]),
                             List([Oz, Hz, Fz, Mz, Cz])])]),
           Atom(b45, [Structure(s, [List(['x','u','k','l','w','f',Zz,Lz,Rz,'h',Uz,'t']),
                                    List([Zz,Lz,Rz,Uz])])]),
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

                }# positive examples
    pos = {Atom(b45, [
        Structure(s, [List([F2, 'c', E2, Q2, E2,'h','c',F2, C2, 'q']),
                      List([F2, E2, Q2, E2, F2, C2])])]),
           Atom(b45, [Structure(s, [List(
               ['o', N2, A2, 'z', 'g', 'f']),
               List(
                   [N2,A2])])]),
           Atom(b45, [Structure(s, [List([T2,'r','g','y',T2,P2]),
                                    List([T2,T2,P2])])]),
           Atom(b45, [
               Structure(s, [List(['c',O2,'o','j','j',H2,F2,M2,'g',C2]),
                             List([O2, H2, F2, M2, C2])])]),
           Atom(b45, [Structure(s, [List(['x','u','k','l','w','f',Z2,L2,R2,'h',U2,'t']),
                                    List([Z2,L2,R2,U2])])])}

    # negative examples
    neg = {Atom(b45, [
        Structure(s, [List(['e', 'd', 'i', 't', 'h']),
                      List(['e','i'])])]),
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
        Structure(s, [List([Fz, 'c', Ez, Qz, Ez,'h','c',Fz, Cz, 'q']),
                      List([Fz, Ez, Qz, Ez, Fz, Cz])])]),
           Atom(b45, [Structure(s, [List(
               ['o', Nz, Az, 'z', 'g', 'f']),
               List(
                   [Nz,Az])])]),
           Atom(b45, [Structure(s, [List([Tz,'r','g','y',Tz,Pz]),
                                    List([Tz,Tz,Pz])])]),
           Atom(b45, [
               Structure(s, [List(['c',Oz,'o','j','j',Hz,Fz,Mz,'g',Cz]),
                             List([Oz, Hz, Fz, Mz, Cz])])]),
           Atom(b45, [Structure(s, [List(['x','u','k','l','w','f',Zz,Lz,Rz,'h',Uz,'t']),
                                    List([Zz,Lz,Rz,Uz])])]),
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


    # positive examples
    pos2 = {
        Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, one, zero, hook2a]),
                          List([one, zero])])]),
        Atom(b45, [
        Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, one, hook2a]),
                      List([one])])]),

            Atom(b45, [
                Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, three, hook2a ]),
                              List([three])])]),
            Atom(b45, [
                Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, five,zero ,hook2a ]),
                              List([five,zero])])]),
            Atom(b45, [
                Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, five, hook2a ]),
                              List([five])])])}

    # negative examples
    neg2 = {Atom(b45, [
        Structure(s, [List(['l', 'e', 't', 't', 'e', 'r', hook1a, 'd', hook2a, punt2]),
                      List(['d'])])]),
           Atom(b45, [Structure(s, [List(
               [three, two, punt2, six, space2, hook1z, nine, zero, punt2, seven, hook2a]),
               List([nine, zero, punt2, seven])])]),
           Atom(b45, [Structure(s, [List([four, space2, S2, 'c', 'i', 'e', 'n', 'c', 'e', space2, nine, komma2, three, six, nine, space2, one, komma2, one,two, five,komma2, zero,two,two]),
                                    List([four, space2, S2, 'c', 'i', 'e', 'n', 'c', 'e'])])]),
           Atom(b45, [
               Structure(s, [List([eight, space2, J2, 'o', 'h', 'n', space2, M2, 'a', 'j', 'o', 'r', space2, one, nine, nine, one, streep2, one, nine, nine, seven,space2, C2, 'o', 'n', 's', 'e', 'r', 'v', 'a', 't', 'i', 'v', 'e']),
                             List([J2, 'o', 'h', 'n', space2, M2, 'a', 'j', 'o', 'r'])])]),
           Atom(b45, [Structure(s, [List([three, space2, four, nine, six,space2, 'k', 'm', space2, hook1a, two, komma2, one, seven, two, space2, 'm', 'i', hook2z, space2, eight, seven, 'h', space2, three, four, space2,four,seven]),
                                    List([three, komma2, four, nine, six, komma2, space2, two, komma2, one, seven, two])])])}
    task2 = Task(positive_examples=pos2, negative_examples=neg2)

    examples2 = {
            Atom(b45, [
                Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1z, onez, zeroz, hook2z]),
                          List([onez, zeroz])])]),
            Atom(b45, [
                Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1z, onez, hook2z]),
                      List([onez])])]),

            Atom(b45, [
                Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1z, threez, hook2z ]),
                              List([threez])])]),
            Atom(b45, [
                Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1z , fivez,zeroz ,hook2z ]),
                              List([fivez,zeroz])])]),
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
        head_constructor=b291,
        expansion_hooks_reject=[lambda x, y: has_singleton_vars(x, y),
                                lambda x, y: has_duplicated_literal(x, y)],
        recursive_procedures=True)
    primitives = [not_space, mk_uppercase, mk_lowercase, is_empty, is_space, is_uppercase, not_uppercase, is_lowercase,
                  not_lowercase, is_letter, not_letter, is_number, not_number, skip1, copy1, write1, copyskip1]
    head = Atom(b45, [A])

    print(Clause(head,[]))
    body1 = Atom(write1, [A,A,A])
    body2 = Atom(mk_uppercase, [A, B])
    body3 = Atom(is_lowercase, [B])
    body4 = Atom(mk_lowercase, [B, C])
    body5 = Atom(mk_lowercase, [C, D])
    body6 = Atom(b45, [D])
    body7 = Atom(skip1, [D, E])
    clause = Clause(head, Body(body1))


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
    encoding2 = encoding2.encoding(primitives2)
    encodingExamples = {
        1: [-0.2781, 0.6987, 0.9962, -0.9904, 0.7330, 0.2802, 0.9755, 0.0176,
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
        2: [-6.0647e-01, 7.0167e-01, 9.9876e-01, -9.9364e-01, 8.8290e-01,
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
        3: [-2.0942e-01, 6.6699e-01, 9.9740e-01, -9.8807e-01, 7.2662e-01,
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
        4: [-0.3844, 0.7598, 0.9974, -0.9927, 0.7489, -0.1418, 0.9849, 0.6287,
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
        5: [-0.3852, 0.7345, 0.9975, -0.9941, 0.8059, 0.1372, 0.9769, 0.3380,
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
            -0.0145, 0.0737, 0.5049, 0.0341, -0.1828, 0.9989, -0.7702, 0.9887]}
    encodingExamples = {
        1: [-3.6467e-01,  7.2022e-01,  9.9840e-01, -9.9039e-01,  8.3420e-01,
          5.5713e-01,  9.8474e-01,  8.3312e-01, -9.6535e-01, -7.9329e-01,
          9.5382e-01,  9.9510e-01,  4.0992e-01, -9.9503e-01, -9.5377e-02,
         -9.8506e-01,  9.8066e-01, -3.7422e-01, -9.9849e-01, -4.7179e-01,
          4.7208e-01, -9.9781e-01,  5.5456e-01,  8.2827e-01,  9.6785e-01,
          1.1619e-01,  9.8834e-01,  9.9754e-01,  9.2737e-01,  8.9083e-01,
          3.5186e-01, -9.8579e-01,  6.2498e-01, -9.8635e-01,  1.5097e-01,
         -4.1594e-01, -1.5747e-01, -1.0054e-01,  6.0735e-01, -5.2962e-01,
         -9.0573e-01,  1.2193e-01, -5.5054e-01, -5.0213e-01, -7.1285e-01,
          6.9840e-01,  7.0002e-01, -2.8657e-01, -3.4797e-01,  9.9937e-01,
         -9.6649e-01,  1.0000e+00,  4.8330e-01,  9.9986e-01,  9.9640e-01,
          8.4478e-01,  9.9368e-01,  5.0361e-01,  5.6546e-01,  8.7371e-01,
          9.6266e-01, -5.2744e-01,  8.8764e-01, -6.6546e-01, -6.1750e-01,
         -9.2757e-01,  1.1076e-01,  5.3724e-01, -6.0024e-01,  8.7912e-01,
          7.1418e-01,  4.1665e-01,  9.9339e-01, -9.2335e-01,  1.1751e-01,
         -9.4106e-01,  6.5125e-01, -9.9860e-01,  9.7218e-01,  9.9782e-01,
         -7.7100e-01, -9.9456e-01,  9.9022e-01, -3.1527e-01,  2.9199e-01,
         -7.1087e-01,  7.3231e-01, -9.9483e-01,  2.2366e-01, -8.6273e-01,
         -3.6894e-01, -9.6951e-01,  1.2741e-01,  3.6664e-01,  9.9757e-01,
          6.4444e-01, -1.8288e-01,  2.2932e-01,  2.3575e-01,  8.0981e-01,
         -8.1989e-01, -4.2505e-01, -7.9627e-01,  8.6019e-01, -4.7185e-01,
         -3.8494e-01, -9.6987e-01, -2.8147e-01, -4.4234e-01,  1.0784e-01,
          9.9466e-01, -9.7669e-01, -6.6159e-01,  3.0513e-01, -4.2590e-03,
          2.8770e-01,  9.7433e-01,  8.5127e-01, -3.4781e-01,  9.9746e-01,
         -1.6392e-01,  5.5473e-02,  9.9290e-01, -5.1847e-01,  7.7579e-01,
         -4.1315e-01,  8.1691e-02,  1.6544e-03,  4.9338e-01, -9.0688e-01,
          6.7730e-01, -9.8294e-01,  7.3619e-01,  9.9229e-01, -3.9875e-01,
          9.9812e-01, -9.9562e-01,  1.4758e-01, -9.9939e-01,  4.5815e-01,
          9.0275e-01, -2.7708e-01,  2.1270e-01,  7.6177e-01,  9.9479e-01,
          1.8606e-01,  1.7964e-02,  4.5942e-01, -3.7773e-01,  1.7604e-01,
          6.7649e-01,  9.3257e-01, -9.6515e-01,  9.9999e-01, -8.3262e-02,
          2.4859e-01,  7.2922e-01, -2.6777e-02,  3.7276e-01, -3.5377e-01,
          9.7556e-01, -9.9397e-01, -6.8624e-01,  6.4995e-01,  9.9400e-01,
          9.8391e-01, -6.9883e-02,  5.4316e-01,  9.9597e-01,  6.6104e-01,
          3.8143e-01, -7.2665e-01, -6.7922e-01,  4.8303e-01,  2.7916e-01,
          2.6955e-01,  9.6980e-01,  9.9274e-01, -9.8062e-01,  9.9180e-01,
          9.9970e-01, -6.5482e-01,  9.6088e-01, -6.6280e-01, -9.9457e-01,
         -9.8396e-01, -9.7528e-01,  5.2064e-01, -9.0412e-01, -4.3723e-01,
         -3.9116e-01,  9.6043e-01, -8.1902e-01,  9.3267e-01, -9.7377e-01,
         -5.4776e-02,  9.9067e-01, -2.4789e-01,  9.9619e-01,  9.1358e-01,
         -9.9529e-01, -2.5520e-01, -5.6831e-01,  9.7236e-01, -3.1447e-01,
          9.8203e-01,  2.8774e-01,  4.9507e-01,  6.2686e-01, -1.0000e+00,
         -8.6591e-01, -6.2642e-01, -7.0219e-01,  8.0485e-01,  9.8562e-01,
          5.3985e-01, -4.4396e-01, -4.6622e-02,  4.8686e-01,  9.9873e-01,
         -9.8522e-01, -5.3539e-01,  7.2772e-01, -9.8906e-01, -9.9503e-01,
          9.4938e-01, -1.6706e-01, -4.0645e-02, -5.5613e-01, -8.5635e-01,
         -1.2307e-01,  3.8780e-01,  9.8724e-01,  2.7739e-01,  8.1338e-01,
         -9.9848e-01,  9.2932e-01, -9.7434e-01,  1.5176e-01,  4.1901e-01,
          5.9797e-01, -5.1924e-01, -9.3232e-01,  6.9774e-01,  9.1404e-01,
         -5.1095e-01,  6.6777e-01, -7.2099e-01,  7.6532e-01,  7.8509e-01,
         -5.7691e-01, -4.3890e-01, -9.8610e-01,  9.9378e-01,  7.8377e-01,
         -1.7584e-01,  9.8713e-01, -9.8723e-01, -3.8431e-02,  5.5588e-01,
         -6.5306e-02, -9.9998e-01, -1.1045e-01, -8.7917e-01,  8.4667e-01,
         -4.2011e-01,  9.8855e-01, -9.5579e-01, -1.2259e-01,  3.7354e-01,
         -9.9643e-01,  9.5086e-01, -5.3302e-01,  9.8977e-01, -8.3638e-01,
          6.2319e-01,  9.9647e-01,  4.5891e-01, -9.8970e-01, -9.9885e-01,
          7.5077e-01,  9.9983e-01, -9.7343e-01, -1.7714e-01,  9.9868e-01,
          8.2817e-01, -9.6074e-01, -9.4818e-01, -9.9504e-01, -9.9420e-01,
         -6.9759e-01, -9.1964e-02, -2.4149e-01,  9.9221e-01, -3.5998e-02,
         -1.3025e-01,  9.8604e-01,  9.9989e-01, -5.0721e-01,  2.9288e-01,
          3.4391e-01, -8.9459e-01, -9.9993e-01,  7.4551e-01,  1.7648e-01,
         -9.9855e-01,  9.9524e-01, -9.9178e-01,  9.9999e-01, -8.8402e-01,
          7.3901e-01,  6.9683e-01, -3.1562e-01,  4.1003e-01,  1.6442e-01,
          9.9804e-01,  9.4683e-01, -5.8621e-01,  3.0281e-01, -3.5627e-01,
          4.8122e-01, -5.7172e-01,  8.5881e-01,  3.2883e-01,  1.7825e-02,
         -9.0477e-01, -5.4145e-01, -5.2438e-01, -9.9275e-01, -1.0088e-01,
          4.2083e-01,  9.1224e-01, -3.9628e-02,  9.2188e-01,  9.8817e-01,
         -2.3303e-01,  7.7280e-01, -4.4116e-01, -9.9642e-01,  3.5327e-01,
          2.5967e-01, -3.1377e-01,  5.9854e-01,  9.1237e-01,  9.9053e-01,
         -9.4702e-01,  9.9982e-01, -3.6196e-01,  3.7609e-01,  8.0033e-01,
          9.9904e-01, -9.9947e-01,  2.8718e-01, -2.2450e-01,  7.1495e-01,
          7.3650e-01,  9.8598e-01, -2.6949e-01,  1.7506e-01,  4.5297e-02,
          3.1123e-01,  9.2281e-01,  9.8072e-01,  2.7674e-02, -1.8662e-01,
          9.4159e-01,  2.9692e-01,  9.9112e-01, -7.1030e-01, -3.6952e-02,
         -7.4586e-01,  6.6062e-01,  9.6682e-01, -6.5103e-02, -4.3717e-01,
         -3.2153e-01, -7.1926e-02, -5.4892e-02, -5.5772e-01,  7.1477e-01,
          1.6406e-01,  2.4687e-01, -9.9093e-01, -5.3837e-01,  5.1729e-01,
          9.9437e-01, -9.9155e-01, -1.1506e-01,  9.8615e-01, -2.6445e-01,
          1.4104e-01,  7.7587e-01, -2.8414e-01, -9.3303e-01, -3.2512e-01,
         -9.9690e-01, -7.4687e-04, -6.8585e-01,  7.0480e-01,  5.6853e-01,
          1.5994e-01,  8.2832e-02,  9.1427e-02, -8.4324e-01,  9.0821e-01,
          3.7069e-01,  9.8472e-01,  8.1723e-01, -2.5179e-01, -1.1686e-01,
          4.9422e-01,  2.7198e-01, -3.8019e-01,  9.9208e-01, -9.8564e-01,
          9.9877e-01, -3.6990e-01, -9.9558e-01,  8.6555e-01,  3.9364e-01,
         -9.9790e-01,  1.1022e-01, -9.9979e-01,  9.8090e-01, -6.9518e-01,
          7.0505e-01,  7.8127e-01, -9.9978e-01, -9.9998e-01, -3.3966e-01,
         -3.1010e-01, -4.4180e-01, -8.3812e-01,  9.9334e-01, -3.5940e-02,
          6.5241e-01, -1.8249e-01, -9.3167e-01,  3.3659e-01,  7.9161e-01,
         -2.9431e-01, -9.9687e-01,  1.0690e-01,  2.2631e-01,  6.5402e-01,
         -6.6241e-01, -9.3539e-01,  9.7245e-01, -9.4735e-01,  6.8998e-01,
          9.7441e-01,  1.7277e-01,  7.1798e-01, -9.9265e-01,  9.7759e-01,
         -9.0234e-02,  3.0252e-01, -6.8593e-01, -9.9076e-01,  9.9794e-01,
         -7.5323e-01,  4.3099e-02, -3.2234e-01, -9.9012e-01, -1.5001e-01,
          2.3804e-01, -8.0636e-01, -9.7697e-01,  4.5460e-01, -9.9186e-01,
         -9.9843e-01,  4.5891e-01, -6.7833e-01,  9.9934e-01,  9.7563e-01,
          7.5283e-01, -3.6203e-01, -8.7385e-01,  2.4692e-01, -9.9918e-01,
         -8.3757e-01, -2.8049e-01, -9.8432e-01,  5.3306e-01,  9.7881e-01,
          9.8923e-01, -4.7582e-01, -5.9262e-01, -9.1677e-01,  6.8716e-01,
          9.6552e-01,  7.9985e-01, -8.0383e-01,  2.1775e-01, -3.4245e-01,
         -9.8674e-01, -8.6857e-01,  9.9401e-01,  7.2730e-01,  9.9228e-01,
         -7.0997e-01, -5.1830e-01,  2.3733e-01, -1.4184e-01,  3.1063e-01,
         -9.9998e-01, -7.3541e-01, -3.6047e-01, -9.9777e-01,  9.9836e-01,
         -9.9851e-01, -3.7026e-01,  5.5749e-01,  3.7417e-02,  9.8192e-01,
          6.2351e-01, -9.9819e-01, -9.9760e-01, -9.1619e-01, -2.8208e-01,
          9.9302e-01, -1.0444e-01,  5.2516e-01,  6.0417e-01,  3.9913e-01,
          9.9740e-01, -8.6631e-02,  5.4695e-01,  8.0354e-01,  9.9623e-01,
          2.6266e-01, -9.9267e-01,  4.3445e-01, -9.9539e-01,  1.3555e-01,
          9.7253e-01,  9.7545e-01,  9.8620e-01,  7.5062e-01,  9.9642e-01,
         -9.9910e-01,  9.9984e-01, -9.9768e-01,  7.1753e-01,  9.9820e-01,
         -9.9023e-01,  5.3449e-02, -9.9130e-01,  1.5742e-01, -5.3399e-01,
          2.9825e-01, -7.5114e-01,  9.8778e-01, -9.8921e-01, -9.9584e-01,
          7.4022e-01,  8.1353e-01,  8.5749e-01, -3.2263e-01,  1.6313e-01,
          9.8887e-01,  2.7423e-01,  9.6701e-01, -4.7427e-01,  2.5282e-01,
          9.9999e-01, -4.7095e-01,  4.5189e-01, -9.8996e-01,  9.8648e-01,
          3.3287e-01,  4.1250e-01,  9.5950e-01, -4.1976e-01,  4.3225e-01,
          8.1792e-01, -9.9305e-01, -5.2045e-01, -9.5636e-01, -3.8701e-01,
         -2.9471e-01,  5.5507e-01,  3.8675e-01, -6.7569e-02,  1.9664e-01,
         -9.9155e-01,  3.2475e-01, -9.9240e-01,  9.8300e-01,  2.4997e-01,
          2.7541e-01, -1.5502e-01,  7.4260e-01, -9.5090e-01,  9.9453e-01,
          9.9773e-01, -1.0000e+00,  4.2920e-01,  9.8291e-01,  7.2747e-01,
          9.7929e-01, -9.8909e-01, -4.2928e-01,  7.4067e-01, -4.6398e-01,
          9.8308e-01,  6.2295e-01, -4.1026e-01,  9.7675e-01, -9.8712e-01,
         -3.8594e-02, -8.7963e-01, -2.3659e-01,  8.0683e-01, -9.7641e-01,
          2.5411e-01, -4.0663e-01,  3.3587e-01, -9.9825e-01, -8.0989e-01,
         -9.9746e-01, -3.2973e-01,  9.9367e-01,  6.0861e-01,  9.9761e-01,
          7.9804e-01, -5.1881e-01,  4.8415e-02, -9.9606e-01,  5.6737e-01,
          3.5793e-01, -7.8557e-02,  5.7549e-01, -8.1908e-01,  9.1376e-02,
         -1.6113e-01, -9.9811e-01,  2.5686e-01, -2.5253e-01,  4.0164e-01,
          8.3421e-01,  8.1444e-01, -9.7899e-01, -7.2405e-01, -8.9206e-01,
          4.5792e-01, -6.7538e-01, -8.9475e-01,  9.5371e-01, -7.9438e-02,
          9.9841e-01, -9.8829e-01, -9.1986e-01, -9.3689e-01, -4.9114e-01,
          3.8935e-01,  3.9528e-01,  3.7317e-01,  6.7705e-02, -5.2909e-01,
         -5.2766e-01,  9.7323e-01, -9.8629e-01, -9.8944e-01,  9.9382e-01,
         -2.7280e-01,  1.0328e-01, -5.3651e-01, -5.3561e-01, -5.7003e-01,
         -3.7845e-01,  5.5365e-01, -9.1640e-01, -5.5639e-01, -9.9978e-01,
         -2.9526e-01,  5.6871e-01, -9.8106e-01, -8.7977e-01, -4.9732e-01,
         -9.9999e-01,  9.8781e-01,  9.2619e-01,  9.9667e-01, -9.9727e-01,
          9.2196e-01,  3.4629e-01,  9.9274e-01, -4.1276e-01, -6.3577e-01,
          4.3152e-01,  9.9514e-01,  3.9023e-01, -5.9220e-01,  2.1778e-01,
         -1.8334e-01,  1.4113e-01, -8.6677e-01, -4.7582e-01,  6.1070e-01,
          7.2937e-01, -9.7049e-01, -9.9766e-01,  9.9777e-01, -3.6022e-01,
          9.8406e-01,  5.9269e-01,  2.6457e-01, -9.6541e-01,  4.0841e-01,
          3.4999e-01, -8.9449e-01, -9.9814e-01,  7.8095e-01, -1.0000e+00,
         -9.9097e-01,  3.4224e-01,  9.7176e-01, -9.9229e-01, -9.8083e-01,
          1.7807e-01, -9.9868e-01,  3.8393e-01,  1.1461e-01,  7.1896e-01,
         -9.9238e-01, -7.2016e-01, -6.3698e-01,  3.5324e-01,  9.7574e-01,
         -9.6327e-01,  1.1743e-01, -8.2479e-01,  9.8710e-01,  3.6649e-02,
          4.7560e-01,  1.2970e-01,  3.9256e-01,  5.7329e-01, -9.5178e-01,
          3.4018e-01, -9.9095e-01, -9.4817e-01,  9.9007e-01,  9.8360e-01,
         -9.9589e-01, -9.8852e-01, -3.8393e-01, -3.1365e-01,  9.9262e-01,
         -4.8500e-01, -9.9797e-01, -9.9840e-01,  1.0328e-01, -7.0594e-01,
          9.9292e-01, -4.1498e-01,  9.9997e-01,  9.4807e-01,  5.9744e-01,
         -2.2524e-01,  6.7606e-01,  5.4553e-01,  3.9002e-01, -4.6390e-01,
          9.9857e-01, -1.8173e-01,  9.8774e-01],
        2: [-0.6333,  0.7279,  0.9994, -0.9895,  0.8537, -0.0866,  0.9803,  0.8225,
         -0.9765, -0.7353,  0.9453,  0.9985,  0.0382, -0.9962, -0.4276, -0.9761,
          0.9865, -0.1705, -0.9996, -0.7453,  0.4019, -0.9989,  0.2618,  0.8165,
          0.9655,  0.2950,  0.9886,  0.9987,  0.9353,  0.8862,  0.4954, -0.9934,
          0.2893, -0.9958, -0.1171, -0.4972, -0.6741, -0.0572,  0.6030, -0.6539,
         -0.9269,  0.4618, -0.8940, -0.4075, -0.6538,  0.7023,  0.6071, -0.0699,
         -0.3806,  0.9997, -0.9566,  1.0000,  0.2986,  0.9999,  0.9951,  0.8451,
          0.9934,  0.5333,  0.2974,  0.9008,  0.9869, -0.4758,  0.9683, -0.6659,
         -0.6507, -0.8981,  0.4827,  0.3499, -0.8343,  0.9167,  0.8895,  0.2867,
          0.9973, -0.9599,  0.0477, -0.9443,  0.8738, -0.9994,  0.9795,  0.9990,
         -0.8299, -0.9965,  0.9966, -0.2802,  0.4965, -0.9205,  0.2218, -0.9973,
          0.2483, -0.9532, -0.5708, -0.9757, -0.1172,  0.4755,  0.9987,  0.6183,
         -0.2235,  0.1892,  0.4482,  0.3771, -0.8681, -0.4894, -0.4456,  0.6220,
         -0.0588, -0.5856, -0.9672, -0.3871, -0.7516,  0.0182,  0.9939, -0.9898,
         -0.6680,  0.2027, -0.2447, -0.2309,  0.9850,  0.8566, -0.2587,  0.9985,
         -0.0539,  0.3697,  0.9958, -0.4861,  0.7117, -0.4702,  0.3973,  0.0900,
          0.4276, -0.9101,  0.6653, -0.9920,  0.5779,  0.9957, -0.2462,  0.9994,
         -0.9967,  0.3777, -0.9996,  0.3219,  0.8702, -0.1688, -0.2676,  0.8939,
          0.9953,  0.1153,  0.1076,  0.5845, -0.7580,  0.7017,  0.8270,  0.9439,
         -0.9795,  1.0000,  0.2893,  0.0855,  0.5512, -0.0164, -0.0432, -0.0345,
          0.9416, -0.9956, -0.0663,  0.1336,  0.9962,  0.9897,  0.0222,  0.3216,
          0.9983,  0.9209,  0.5645, -0.8182, -0.8237, -0.0212,  0.2155,  0.3264,
          0.9872,  0.9944, -0.9904,  0.9936,  0.9999, -0.6329,  0.9686, -0.4184,
         -0.9984, -0.9914, -0.9860,  0.5679, -0.9458, -0.1916, -0.7013,  0.9345,
         -0.5305,  0.9594, -0.9827, -0.1899,  0.9869, -0.0980,  0.9987,  0.8377,
         -0.9983,  0.0253, -0.5147,  0.9515, -0.3520,  0.9886,  0.5333,  0.5177,
          0.7953, -1.0000, -0.5870, -0.7268, -0.7715,  0.8534,  0.9912,  0.2205,
         -0.1892, -0.1214,  0.2524,  0.9993, -0.9946, -0.3249,  0.8722, -0.9953,
         -0.9976,  0.9652, -0.1106, -0.1337, -0.6633, -0.9530, -0.0321,  0.5340,
          0.9853,  0.4310,  0.8823, -0.9992,  0.7759, -0.9848,  0.0808,  0.5084,
          0.5560, -0.5334, -0.9699,  0.3301,  0.9320,  0.4519,  0.3437, -0.7301,
          0.9128,  0.4065, -0.6282, -0.1990, -0.9866,  0.9971,  0.5645, -0.2718,
          0.9860, -0.9823,  0.2386, -0.0422, -0.0878, -1.0000, -0.5450, -0.8810,
          0.7511, -0.3713,  0.9950, -0.9645, -0.1139,  0.0855, -0.9990,  0.9388,
         -0.3453,  0.9967, -0.8852,  0.7546,  0.9973,  0.6863, -0.9797, -0.9993,
          0.4843,  0.9999, -0.9877, -0.1145,  0.9993,  0.5494, -0.9670, -0.9302,
         -0.9975, -0.9967, -0.6303,  0.1531, -0.3063,  0.9921, -0.5926, -0.2948,
          0.9882,  1.0000, -0.4898,  0.0674,  0.1960, -0.9466, -1.0000,  0.5138,
          0.1183, -0.9991,  0.9978, -0.9906,  1.0000, -0.9359,  0.5039,  0.8526,
         -0.1926,  0.1514,  0.1524,  0.9993,  0.9825, -0.2883,  0.0343, -0.2814,
          0.4878, -0.6770,  0.8744,  0.6403,  0.0475, -0.9023, -0.4246, -0.1495,
         -0.9946, -0.2744,  0.4081,  0.9339, -0.5014,  0.9615,  0.9861, -0.0045,
          0.8833, -0.3730, -0.9991, -0.0031,  0.2741, -0.5375,  0.7429,  0.6994,
          0.9920, -0.9217,  0.9999, -0.3164,  0.3576,  0.3753,  0.9995, -0.9998,
          0.1707, -0.3983,  0.7091,  0.8858,  0.9927, -0.0516,  0.5302, -0.5511,
          0.3478,  0.9536,  0.9873, -0.2445, -0.2945,  0.7974,  0.9184,  0.9919,
         -0.3362, -0.0333, -0.8385,  0.4456,  0.9634, -0.2004,  0.1727, -0.4213,
          0.0912,  0.2212,  0.1788,  0.8672,  0.2742,  0.3886, -0.9929, -0.7533,
          0.5741,  0.9970, -0.9915, -0.6028,  0.9961, -0.2844,  0.4434,  0.8055,
         -0.1519, -0.9650, -0.2620, -0.9991, -0.1574, -0.8314,  0.8754,  0.7860,
          0.1553,  0.3572, -0.1323, -0.8484,  0.9399,  0.4097,  0.9869,  0.8049,
         -0.4035, -0.2132,  0.8078,  0.2467, -0.0798,  0.9942, -0.9895,  0.9995,
         -0.2729, -0.9985,  0.6671,  0.5164, -0.9988, -0.2962, -0.9998,  0.9857,
         -0.7470,  0.3072,  0.3205, -0.9999, -1.0000, -0.5803, -0.5283, -0.4425,
         -0.8534,  0.9984,  0.1928,  0.8247, -0.1732, -0.9469,  0.1960,  0.2838,
         -0.0822, -0.9993,  0.1449,  0.4451,  0.6702, -0.0930, -0.9308,  0.9150,
         -0.9688,  0.7482,  0.9829,  0.6253,  0.8868, -0.9955,  0.9901, -0.1586,
          0.2446, -0.7391, -0.9926,  0.9990, -0.1888, -0.1324, -0.4600, -0.9929,
          0.0047, -0.1449, -0.7747, -0.9915,  0.3611, -0.9939, -0.9997,  0.2527,
         -0.1264,  0.9997,  0.9787,  0.6688, -0.0422, -0.9416,  0.1765, -0.9997,
         -0.6107,  0.0562, -0.9877,  0.7177,  0.9778,  0.9872, -0.7215, -0.5458,
         -0.9349,  0.6989,  0.9585,  0.8784, -0.6758,  0.0787, -0.3734, -0.9875,
         -0.9608,  0.9964,  0.3510,  0.9929, -0.3198,  0.0878,  0.2549, -0.3198,
          0.0043, -1.0000, -0.8098, -0.2711, -0.9992,  0.9993, -0.9994, -0.5656,
          0.7157, -0.2841,  0.9861,  0.6726, -0.9993, -0.9993, -0.9195, -0.3925,
          0.9934, -0.0663,  0.2774,  0.7449,  0.6081,  0.9986,  0.6158,  0.7741,
          0.5038,  0.9984,  0.2714, -0.9950,  0.5172, -0.9980,  0.6581,  0.9677,
          0.9887,  0.9852,  0.2695,  0.9988, -0.9996,  0.9999, -0.9993,  0.2526,
          0.9989, -0.9958,  0.3622, -0.9966, -0.0648, -0.6330,  0.1697, -0.7836,
          0.9951, -0.9970, -0.9976,  0.8337,  0.7510,  0.9305, -0.0218,  0.3849,
          0.9903,  0.2278,  0.9806, -0.5852,  0.2568,  1.0000, -0.5452,  0.2736,
         -0.9914,  0.9933,  0.5734,  0.1404,  0.9651, -0.4019,  0.3614,  0.8463,
         -0.9943, -0.4409, -0.9924,  0.1883, -0.4720,  0.4984,  0.2719, -0.2676,
          0.2125, -0.9887,  0.5237, -0.9943,  0.9891, -0.2099,  0.0614,  0.0826,
          0.9356, -0.8745,  0.9958,  0.9985, -1.0000,  0.1764,  0.9836,  0.7980,
          0.9841, -0.9898, -0.5305,  0.7785, -0.7331,  0.9899,  0.7723, -0.2914,
          0.9790, -0.9921,  0.2630, -0.8972, -0.2563,  0.8332, -0.9853,  0.6205,
          0.0033,  0.2182, -0.9988, -0.7356, -0.9982, -0.4407,  0.9940,  0.6394,
          0.9992,  0.8201, -0.5692,  0.0714, -0.9980,  0.3325,  0.4306, -0.4833,
          0.7998, -0.4111,  0.3591, -0.4268, -0.9984,  0.0651,  0.3362,  0.2510,
          0.8893,  0.8994, -0.9961, -0.8027, -0.9085,  0.3651, -0.7525, -0.8891,
          0.8597, -0.0425,  0.9992, -0.9916, -0.9520, -0.9806, -0.8050,  0.5358,
          0.4208,  0.3819, -0.1099, -0.0860, -0.6879,  0.9839, -0.9902, -0.9940,
          0.9977, -0.2308,  0.3129, -0.5393, -0.4782, -0.6424, -0.2065,  0.5254,
         -0.9411, -0.3750, -0.9999, -0.9019,  0.3188, -0.9913, -0.9151, -0.5954,
         -1.0000,  0.9942,  0.9544,  0.9986, -0.9989,  0.9344,  0.4216,  0.9947,
         -0.3640, -0.5341,  0.2348,  0.9976,  0.6415, -0.4617, -0.1321, -0.1381,
         -0.0571, -0.8871, -0.1746,  0.5657,  0.8403, -0.9659, -0.9984,  0.9985,
         -0.2167,  0.9927,  0.5250,  0.1956, -0.9649,  0.5081,  0.2023, -0.9240,
         -0.9994,  0.9127, -1.0000, -0.9938,  0.5208,  0.9790, -0.9940, -0.9818,
          0.0608, -0.9993,  0.2776, -0.1532,  0.7284, -0.9946, -0.1029, -0.5638,
          0.4115,  0.9879, -0.9733,  0.3200, -0.6661,  0.9963,  0.0760,  0.4792,
          0.1072,  0.0104,  0.3572, -0.9765, -0.5619, -0.9963, -0.9436,  0.9968,
          0.9824, -0.9981, -0.9938, -0.3294, -0.6456,  0.9930, -0.7225, -0.9986,
         -0.9993,  0.2666, -0.9004,  0.9951, -0.3321,  1.0000,  0.9495,  0.8345,
         -0.5186,  0.6581,  0.6917,  0.7257, -0.3923,  0.9992,  0.1690,  0.9939],
        3: [-0.6163,  0.6654,  0.9971, -0.9868,  0.7959, -0.0906,  0.9855,  0.4276,
         -0.9411, -0.7618,  0.9687,  0.9911,  0.2056, -0.9943, -0.7627, -0.9887,
          0.9741, -0.4587, -0.9986, -0.5401,  0.4223, -0.9953,  0.5477,  0.7814,
          0.9688,  0.2820,  0.9772,  0.9957,  0.9522,  0.8762,  0.3690, -0.9904,
          0.3857, -0.9857,  0.0775, -0.6199, -0.5847,  0.1728,  0.3890,  0.1565,
         -0.9516,  0.7641, -0.5165, -0.5289, -0.4740,  0.7989,  0.4678, -0.1215,
         -0.1869,  0.9996, -0.9710,  1.0000,  0.4536,  0.9998,  0.9891,  0.8731,
          0.9832,  0.3865,  0.1871,  0.8696,  0.9790, -0.4057,  0.9199, -0.6656,
         -0.6794, -0.9215, -0.0042,  0.2894, -0.5632,  0.9130,  0.6436,  0.3322,
          0.9958, -0.9332, -0.0169, -0.9407,  0.4924, -0.9983,  0.9615,  0.9979,
         -0.6569, -0.9953,  0.9872, -0.0937,  0.4045, -0.8235,  0.6287, -0.9903,
          0.2939, -0.8910, -0.2473, -0.9726, -0.0790,  0.3085,  0.9979,  0.7044,
         -0.2941,  0.3725,  0.2900,  0.2500, -0.7592, -0.2370, -0.7020,  0.8734,
         -0.0074,  0.0473, -0.9789, -0.3036, -0.6214,  0.1416,  0.9934, -0.9867,
         -0.6056,  0.1061, -0.4527,  0.2864,  0.9785,  0.6962, -0.1116,  0.9975,
         -0.1503, -0.4941,  0.9886, -0.7391,  0.7938, -0.1513,  0.0505, -0.3817,
          0.3407, -0.9035,  0.7923, -0.9640,  0.7575,  0.9866, -0.3693,  0.9984,
         -0.9947, -0.2074, -0.9991, -0.0739,  0.9143, -0.1586,  0.3536,  0.7527,
          0.9869,  0.2816,  0.2673,  0.5132, -0.3460,  0.1400,  0.6906,  0.9067,
         -0.9561,  1.0000,  0.2485, -0.2778,  0.4427,  0.2531,  0.2719, -0.6936,
          0.9788, -0.9843, -0.8926,  0.6752,  0.9900,  0.9674, -0.1351,  0.1940,
          0.9929,  0.7009,  0.5346, -0.7187, -0.6909,  0.5951,  0.3709,  0.3873,
          0.9629,  0.9954, -0.9841,  0.9882,  0.9998, -0.6126,  0.9066, -0.5013,
         -0.9924, -0.9828, -0.9676,  0.3049, -0.8542, -0.1916, -0.5216,  0.9600,
         -0.8028,  0.9097, -0.9689, -0.2332,  0.9803, -0.3219,  0.9963,  0.9353,
         -0.9919,  0.4006, -0.5741,  0.9658, -0.3672,  0.9821,  0.3903,  0.4725,
          0.5441, -1.0000, -0.7644, -0.5718, -0.7969,  0.8246,  0.9783,  0.5508,
         -0.3960, -0.1772,  0.3736,  0.9987, -0.9836, -0.6880,  0.7286, -0.9831,
         -0.9923,  0.9670, -0.1549,  0.4989, -0.4191, -0.9480, -0.0463, -0.1613,
          0.9773,  0.4788,  0.7585, -0.9967,  0.8825, -0.9610, -0.0550,  0.4104,
          0.6465, -0.5859, -0.9662,  0.7125,  0.9031, -0.5840,  0.4146, -0.5488,
          0.7331,  0.8096, -0.2987, -0.3994, -0.9825,  0.9927,  0.8504, -0.2660,
          0.9809, -0.9793, -0.0228,  0.4527, -0.1030, -1.0000, -0.1872, -0.8587,
          0.8817, -0.2438,  0.9806, -0.9493,  0.5252,  0.4771, -0.9963,  0.9174,
         -0.4997,  0.9877, -0.8844,  0.7080,  0.9927,  0.6123, -0.9763, -0.9978,
          0.6871,  0.9999, -0.9649, -0.1699,  0.9970,  0.8182, -0.9650, -0.9293,
         -0.9951, -0.9865, -0.6908,  0.0141,  0.0556,  0.9878,  0.3334, -0.2850,
          0.9741,  0.9999, -0.3286,  0.4204,  0.1795, -0.8805, -1.0000,  0.5728,
          0.1060, -0.9984,  0.9914, -0.9876,  1.0000, -0.9072,  0.5950,  0.8631,
          0.0129,  0.1152,  0.1838,  0.9976,  0.9793, -0.4897,  0.3277, -0.4737,
          0.2512, -0.5915,  0.9165,  0.5420, -0.1152, -0.8525,  0.1329, -0.4018,
         -0.9836,  0.1820,  0.4914,  0.9447, -0.3668,  0.9062,  0.9718, -0.2939,
          0.6281, -0.3768, -0.9992,  0.2843,  0.3285, -0.5658,  0.5403,  0.8662,
          0.9857, -0.9732,  0.9999, -0.3977,  0.4110,  0.6947,  0.9989, -0.9995,
          0.3821, -0.0066,  0.6914,  0.7770,  0.9834, -0.3081,  0.1029,  0.5978,
          0.2155,  0.9610,  0.9849,  0.0508, -0.1285,  0.8702,  0.3864,  0.9848,
         -0.5989,  0.1619, -0.6894,  0.3122,  0.9701,  0.6195, -0.6687, -0.2585,
         -0.0082,  0.0675, -0.6466,  0.5512, -0.0947,  0.2862, -0.9880, -0.3163,
          0.7091,  0.9961, -0.9859, -0.1518,  0.9878, -0.2917,  0.2302,  0.7093,
         -0.1300, -0.9317, -0.3563, -0.9956, -0.2241, -0.5605,  0.5748,  0.5368,
          0.0122, -0.0427,  0.3008, -0.9214,  0.8689,  0.3133,  0.9749,  0.7520,
         -0.0838, -0.2063,  0.5360,  0.2233, -0.2904,  0.9926, -0.9898,  0.9984,
         -0.1100, -0.9945,  0.8958,  0.3218, -0.9983,  0.0707, -0.9996,  0.9661,
         -0.7662,  0.6489,  0.8158, -0.9997, -1.0000, -0.2837, -0.3629, -0.2017,
         -0.8205,  0.9966,  0.1003,  0.5563, -0.0557, -0.9372,  0.3465,  0.7748,
         -0.0933, -0.9974,  0.3479,  0.5726,  0.2574, -0.3642, -0.9210,  0.9953,
         -0.9066,  0.7170,  0.9522,  0.4505,  0.7399, -0.9964,  0.9850,  0.2477,
          0.3778, -0.6846, -0.9705,  0.9973, -0.8851,  0.2098, -0.2458, -0.9915,
          0.1186,  0.1971, -0.7048, -0.9719,  0.1829, -0.9948, -0.9973,  0.4653,
         -0.6526,  0.9993,  0.9733,  0.7677, -0.3035, -0.9385,  0.1164, -0.9992,
         -0.7989, -0.1368, -0.9815,  0.6256,  0.9821,  0.9865, -0.0901, -0.5349,
         -0.9165,  0.6988,  0.9821,  0.8500, -0.6170,  0.1793, -0.2188, -0.9821,
         -0.8908,  0.9900,  0.7248,  0.9850, -0.7114, -0.3843, -0.0567, -0.1001,
          0.2937, -1.0000, -0.8645, -0.2809, -0.9972,  0.9971, -0.9984, -0.5223,
          0.5099, -0.1099,  0.9805,  0.6160, -0.9976, -0.9964, -0.9634, -0.2634,
          0.9804, -0.1608,  0.4726,  0.6137,  0.4352,  0.9931,  0.2956,  0.6327,
          0.7809,  0.9938,  0.0765, -0.9839,  0.5076, -0.9957, -0.0938,  0.9627,
          0.8922,  0.9835,  0.7115,  0.9966, -0.9988,  0.9998, -0.9977,  0.7522,
          0.9975, -0.9664,  0.2654, -0.9808, -0.0959, -0.4220,  0.1335, -0.7658,
          0.9831, -0.9851, -0.9876,  0.8000,  0.6952,  0.7555, -0.2815,  0.2000,
          0.9897,  0.2557,  0.9668, -0.5178,  0.1688,  1.0000, -0.0842,  0.6156,
         -0.9906,  0.9830,  0.0442,  0.2600,  0.9590, -0.3401,  0.5527,  0.8224,
         -0.9778, -0.4832, -0.9768, -0.4044, -0.5619,  0.6627,  0.4119, -0.0703,
          0.2158, -0.9692,  0.1175, -0.9936,  0.9924,  0.5297,  0.1734, -0.3078,
          0.8108, -0.9227,  0.9852,  0.9929, -1.0000,  0.3223,  0.9804,  0.5845,
          0.9842, -0.9797, -0.5557,  0.5633, -0.5568,  0.9760,  0.6536, -0.3940,
          0.9746, -0.9708,  0.3789, -0.8884, -0.4311,  0.7232, -0.9583,  0.3546,
         -0.2023,  0.1657, -0.9966, -0.9180, -0.9905, -0.2643,  0.9822,  0.7552,
          0.9975,  0.7632, -0.6442, -0.0458, -0.9958,  0.3793,  0.2977, -0.1008,
          0.6771, -0.6901,  0.2132, -0.2305, -0.9978,  0.3163, -0.1587,  0.2346,
          0.8876,  0.7286, -0.9923, -0.7915, -0.8964,  0.4845, -0.6777, -0.9040,
          0.9756, -0.4402,  0.9984, -0.9805, -0.8606, -0.9425, -0.3565, -0.0402,
          0.3248,  0.2554,  0.3181, -0.3310, -0.2105,  0.9673, -0.9818, -0.9863,
          0.9948, -0.8420,  0.2132, -0.5778, -0.6257, -0.5638, -0.2313,  0.5159,
         -0.8274, -0.3944, -0.9998, -0.4473,  0.4139, -0.9856, -0.9270, -0.5495,
         -1.0000,  0.9857,  0.9176,  0.9959, -0.9965,  0.9521,  0.3865,  0.9920,
         -0.4418, -0.6233,  0.3162,  0.9948,  0.6814, -0.5154, -0.1588, -0.1764,
         -0.0711, -0.7142, -0.3921,  0.8261,  0.6662, -0.9750, -0.9971,  0.9959,
         -0.3914,  0.9784,  0.6779,  0.1206, -0.9520,  0.5628,  0.3423, -0.9306,
         -0.9986,  0.8636, -1.0000, -0.9845,  0.4218,  0.9851, -0.9953, -0.9845,
          0.1452, -0.9986,  0.0350, -0.1033,  0.6618, -0.9777, -0.7433, -0.7133,
          0.2146,  0.9779, -0.9408,  0.4513, -0.8337,  0.9937,  0.0895,  0.4235,
         -0.4274,  0.2380,  0.7785, -0.9627, -0.1862, -0.9847, -0.9524,  0.9811,
          0.9857, -0.9951, -0.9864, -0.1842, -0.3247,  0.9912, -0.6501, -0.9960,
         -0.9975,  0.1335, -0.7430,  0.9911, -0.4087,  1.0000,  0.9531,  0.7483,
         -0.2306,  0.1224,  0.5049,  0.0548, -0.4120,  0.9980, -0.0046,  0.9836],
        4: [-0.3692,  0.7683,  0.9997, -0.9960,  0.9313, -0.5151,  0.9839,  0.8425,
         -0.9883, -0.8399,  0.9867,  0.9967,  0.4960, -0.9991, -0.5442, -0.9926,
          0.9909, -0.4797, -0.9995, -0.4182,  0.5903, -0.9997,  0.5474,  0.7378,
          0.9861,  0.3996,  0.9897,  0.9990,  0.9486,  0.7373,  0.3463, -0.9943,
          0.4904, -0.9982,  0.5211, -0.2921,  0.2226, -0.4004,  0.7226, -0.2301,
         -0.9580,  0.3579, -0.7803, -0.5726, -0.7217,  0.9207,  0.7899, -0.1723,
         -0.2815,  1.0000, -0.9724,  1.0000,  0.5432,  1.0000,  0.9946,  0.9638,
          0.9889,  0.4225,  0.5693,  0.7939,  0.9955, -0.4842,  0.9830, -0.4098,
         -0.5239, -0.9656, -0.0034,  0.6043, -0.7944,  0.9013,  0.7765,  0.4395,
          0.9983, -0.9512, -0.0266, -0.9629,  0.5700, -0.9998,  0.9892,  0.9995,
         -0.5568, -0.9980,  0.9959, -0.4730,  0.4400, -0.8117,  0.8947, -0.9971,
          0.4711, -0.9564, -0.6132, -0.9854, -0.3207,  0.5419,  0.9996,  0.7258,
         -0.5454,  0.5009,  0.2986,  0.6407, -0.8871,  0.0253, -0.9462,  0.9577,
         -0.5213, -0.7007, -0.9904, -0.0266, -0.4052,  0.3716,  0.9979, -0.9978,
         -0.2957,  0.2611, -0.5903, -0.0248,  0.9903,  0.8796, -0.2968,  0.9997,
         -0.0440, -0.5892,  0.9943,  0.0814,  0.8699, -0.5534,  0.0413,  0.0953,
          0.1374, -0.8394,  0.7037, -0.9893,  0.7348,  0.9983, -0.5512,  0.9998,
         -0.9966, -0.3892, -0.9998,  0.6092,  0.6201, -0.0177,  0.4979,  0.7511,
          0.9923,  0.4232,  0.4238,  0.6837, -0.3572,  0.2318,  0.7139,  0.9396,
         -0.9906,  1.0000, -0.3413, -0.0797,  0.3496,  0.1987,  0.5596, -0.4723,
          0.9896, -0.9965, -0.2432,  0.8538,  0.9991,  0.9911,  0.0396,  0.2851,
          0.9991,  0.6239,  0.1095, -0.6388, -0.6127,  0.7981,  0.1308,  0.5411,
          0.9908,  0.9987, -0.9945,  0.9984,  0.9999, -0.6569,  0.9776, -0.7753,
         -0.9980, -0.9968, -0.9887,  0.7134, -0.6182,  0.4012, -0.6999,  0.9714,
         -0.9445,  0.9428, -0.9826, -0.1917,  0.9944, -0.6522,  0.9993,  0.8995,
         -0.9988, -0.4933, -0.5573,  0.9877, -0.6642,  0.9929,  0.0424,  0.6676,
          0.5219, -1.0000, -0.9604, -0.5417, -0.8276,  0.8773,  0.9917,  0.8161,
         -0.5503,  0.0637,  0.3613,  0.9998, -0.9979, -0.7207,  0.6737, -0.9959,
         -0.9948,  0.9902, -0.2063, -0.1476, -0.6849, -0.9657,  0.3553,  0.4404,
          0.9902,  0.5405,  0.8352, -0.9993,  0.9768, -0.9902,  0.1265,  0.3966,
          0.8481, -0.5455, -0.9763,  0.8648,  0.9244, -0.5093,  0.5588, -0.6096,
          0.8657,  0.9572, -0.5793,  0.0412, -0.9959,  0.9981,  0.6413, -0.5607,
          0.9950, -0.9947, -0.1807,  0.8740,  0.1470, -1.0000, -0.5787, -0.7723,
          0.6906, -0.4980,  0.9930, -0.9973,  0.4264,  0.0094, -0.9994,  0.9584,
         -0.6459,  0.9959, -0.9351,  0.3009,  0.9968,  0.4276, -0.9947, -0.9997,
          0.2415,  1.0000, -0.9819, -0.4601,  0.9996,  0.9242, -0.9873, -0.9821,
         -0.9987, -0.9975, -0.5590,  0.3390, -0.1535,  0.9952, -0.2245, -0.2838,
          0.9923,  1.0000, -0.3902,  0.7160,  0.3832, -0.9536, -1.0000,  0.7715,
          0.0079, -0.9998,  0.9987, -0.9946,  1.0000, -0.9394,  0.7442,  0.9280,
         -0.1859,  0.4158,  0.3447,  0.9996,  0.9862, -0.7022,  0.4081, -0.5215,
          0.5301, -0.6456,  0.8923,  0.2786,  0.3136, -0.9235, -0.4927, -0.3506,
         -0.9974, -0.3899,  0.6654,  0.9770,  0.1171,  0.9686,  0.9947, -0.3510,
          0.6056, -0.6369, -0.9996,  0.6513,  0.6374, -0.3335,  0.7685,  0.7890,
          0.9913, -0.9908,  1.0000, -0.5799,  0.3322,  0.8928,  0.9998, -0.9999,
          0.3180,  0.3293,  0.1816,  0.7784,  0.9950, -0.2720, -0.0508,  0.5133,
          0.4411,  0.9739,  0.9921,  0.2066, -0.0388,  0.9756,  0.5813,  0.9959,
         -0.6008, -0.2820, -0.6277,  0.5841,  0.9863, -0.0196,  0.1184, -0.3232,
          0.0155, -0.1991, -0.8419,  0.7002, -0.0023,  0.5179, -0.9949, -0.5121,
          0.2784,  0.9981, -0.9867, -0.4481,  0.9942, -0.5489,  0.1920,  0.8293,
         -0.6389, -0.9856, -0.3947, -0.9993, -0.0150, -0.6875,  0.3057,  0.4269,
          0.2853,  0.0948,  0.1106, -0.9280,  0.9191,  0.4738,  0.9975,  0.8036,
         -0.6249, -0.4306,  0.3906,  0.6037, -0.4478,  0.9974, -0.9974,  0.9998,
         -0.7052, -0.9995,  0.9606,  0.4597, -0.9996,  0.4031, -1.0000,  0.9918,
         -0.8086,  0.8932,  0.9529, -1.0000, -1.0000, -0.5899,  0.1563, -0.4634,
         -0.7707,  0.9993, -0.1752,  0.7885, -0.2615, -0.9732,  0.2892,  0.9300,
          0.0527, -0.9998,  0.4145,  0.0261,  0.8044, -0.4654, -0.9491,  0.9051,
         -0.9793,  0.6685,  0.9932,  0.1588,  0.4490, -0.9985,  0.9939, -0.4866,
          0.4810, -0.7883, -0.9942,  0.9997, -0.6893, -0.0587, -0.3617, -0.9973,
         -0.0727, -0.3858, -0.3365, -0.9973,  0.5564, -0.9973, -0.9997,  0.5350,
         -0.8154,  1.0000,  0.9887,  0.7161, -0.2014, -0.9526,  0.2876, -0.9998,
         -0.6976,  0.1628, -0.9963,  0.0766,  0.9967,  0.9944, -0.1559, -0.4681,
         -0.9297,  0.7081,  0.9864,  0.9164, -0.7928,  0.3380, -0.3440, -0.9957,
         -0.9760,  0.9937,  0.9444,  0.9968, -0.8212, -0.8098,  0.2788, -0.5413,
          0.5507, -1.0000, -0.9219, -0.2625, -0.9997,  0.9998, -0.9997, -0.5996,
          0.6021,  0.3349,  0.9953,  0.8302, -0.9996, -0.9993, -0.6189, -0.2702,
          0.9966, -0.3645,  0.6586,  0.6208,  0.7261,  0.9991, -0.0032,  0.7922,
          0.8756,  0.9979,  0.6491, -0.9974,  0.1268, -0.9990,  0.4942,  0.9826,
          0.9705,  0.9880,  0.9231,  0.9994, -0.9997,  1.0000, -0.9995,  0.9415,
          0.9996, -0.9971,  0.5636, -0.9981,  0.5155, -0.5487,  0.4204, -0.7542,
          0.9943, -0.9967, -0.9957,  0.3827,  0.7635,  0.6616, -0.4622,  0.0036,
          0.9962,  0.4620,  0.9855, -0.3772, -0.0307,  1.0000, -0.1472,  0.5547,
         -0.9829,  0.9925,  0.3072,  0.6376,  0.9714, -0.6547,  0.4898,  0.9454,
         -0.9942, -0.4967, -0.9828, -0.7159, -0.5143,  0.6922,  0.5094,  0.2531,
          0.4144, -0.9954,  0.4855, -0.9986,  0.9916,  0.6160,  0.3936, -0.2553,
          0.6178, -0.9513,  0.9986,  0.9983, -1.0000,  0.3039,  0.9892,  0.4286,
          0.9946, -0.9948, -0.5701,  0.4856, -0.8257,  0.9940,  0.6761, -0.5321,
          0.9927, -0.9926,  0.2921, -0.9836, -0.2815,  0.9357, -0.9741,  0.3387,
         -0.5203,  0.4917, -0.9991, -0.8612, -0.9989, -0.6758,  0.9935,  0.1508,
          0.9989,  0.8242, -0.4914, -0.0687, -0.9981,  0.8698,  0.3758, -0.3755,
          0.2432, -0.9144,  0.2342, -0.2990, -0.9997,  0.5403, -0.3014, -0.0169,
          0.9541,  0.6982, -0.9972, -0.8381, -0.9855,  0.6712, -0.3430, -0.9434,
          0.9186,  0.1386,  0.9997, -0.9865, -0.9792, -0.9852, -0.0679, -0.1712,
          0.3865,  0.1533,  0.4160, -0.4013, -0.6194,  0.9899, -0.9933, -0.9957,
          0.9980,  0.1410,  0.4934, -0.7175, -0.7770, -0.6380, -0.5997,  0.0312,
         -0.9687, -0.5774, -1.0000, -0.0236,  0.2913, -0.9975, -0.9491, -0.5034,
         -1.0000,  0.9945,  0.9730,  0.9992, -0.9990,  0.9714,  0.5306,  0.9979,
         -0.2377, -0.6643, -0.0733,  0.9984,  0.2461, -0.3691,  0.4191, -0.4953,
         -0.0162, -0.8482, -0.6611,  0.1727,  0.5985, -0.9927, -0.9993,  0.9996,
         -0.3696,  0.9842,  0.5871, -0.0276, -0.9824,  0.7350,  0.5095, -0.9736,
         -0.9996,  0.7526, -1.0000, -0.9863,  0.4535,  0.9930, -0.9986, -0.9964,
          0.3723, -0.9996,  0.1840, -0.0489,  0.4425, -0.9950, -0.9155, -0.8944,
          0.7465,  0.9926, -0.9875, -0.1984, -0.8278,  0.9967,  0.4489,  0.5201,
         -0.0729,  0.3241,  0.7448, -0.9765, -0.4071, -0.9959, -0.9782,  0.9934,
          0.9924, -0.9984, -0.9911, -0.6526, -0.2456,  0.9942, -0.5477, -0.9995,
         -0.9993,  0.2119, -0.8161,  0.9955, -0.6450,  1.0000,  0.9698,  0.6147,
         -0.4437,  0.4978,  0.5839,  0.5699, -0.3927,  0.9999,  0.3604,  0.9880],
        5: [-3.5642e-01,  7.6660e-01,  9.9804e-01, -9.9494e-01,  9.0499e-01,
         -7.5924e-02,  9.8771e-01,  6.7606e-01, -9.9053e-01, -8.2386e-01,
          9.7910e-01,  9.9696e-01,  1.6514e-01, -9.9762e-01, -1.5443e-01,
         -9.8710e-01,  9.7647e-01, -5.2677e-01, -9.9920e-01, -3.0467e-01,
          4.6253e-02, -9.9857e-01,  7.0327e-01,  7.9473e-01,  9.2536e-01,
          3.6475e-01,  9.8829e-01,  9.9816e-01,  9.4068e-01,  9.4131e-01,
          4.5969e-01, -9.9236e-01,  5.2615e-01, -9.9396e-01,  3.2172e-01,
         -7.5438e-01, -2.9330e-01, -1.9097e-01,  3.4246e-01,  1.2360e-02,
         -9.4904e-01,  7.3913e-01, -7.4709e-01, -6.0934e-01, -7.5999e-01,
          5.7728e-01,  7.7190e-01, -1.3718e-01, -4.1254e-01,  9.9974e-01,
         -9.6841e-01,  1.0000e+00,  3.7518e-01,  9.9993e-01,  9.9732e-01,
          7.1795e-01,  9.9587e-01,  4.8021e-01,  1.6566e-01,  8.5616e-01,
          9.7707e-01, -6.4629e-01,  9.3063e-01, -7.8192e-01, -7.7522e-01,
         -8.3453e-01,  5.4631e-01,  5.1753e-01, -6.8438e-01,  8.1773e-01,
          8.5668e-01,  6.2769e-01,  9.9788e-01, -9.5106e-01, -2.9411e-03,
         -9.4868e-01,  6.6852e-01, -9.9882e-01,  9.3677e-01,  9.9882e-01,
         -6.1839e-01, -9.9764e-01,  9.9545e-01, -3.0852e-01,  2.3155e-01,
         -9.2127e-01,  5.1421e-01, -9.9642e-01,  5.4880e-01, -9.2188e-01,
         -7.2359e-01, -9.8064e-01,  6.6980e-02,  6.0088e-01,  9.9902e-01,
          7.7835e-01, -3.0128e-01,  5.3148e-01,  5.5257e-01,  7.2668e-01,
         -9.2779e-01, -4.0721e-01, -7.2828e-01,  8.1525e-01,  1.9153e-01,
         -3.6316e-01, -9.5336e-01, -2.4462e-01, -7.6407e-01,  2.4267e-01,
          9.9506e-01, -9.7039e-01, -6.3154e-01,  3.2712e-01, -6.4711e-01,
          1.6297e-01,  9.8311e-01,  7.2734e-01, -3.8989e-01,  9.9884e-01,
         -3.7888e-01,  2.1614e-01,  9.9572e-01, -1.8777e-01,  8.1291e-01,
         -5.7708e-01,  2.5392e-01, -2.5964e-02,  6.7536e-01, -8.9752e-01,
          7.5323e-01, -9.8706e-01,  5.5676e-01,  9.9555e-01, -4.0875e-01,
          9.9901e-01, -9.9780e-01,  4.8788e-01, -9.9955e-01,  8.2109e-02,
          9.1104e-01, -9.5648e-02, -6.2856e-02,  7.2674e-01,  9.9514e-01,
          1.9828e-01,  1.9257e-01,  7.7122e-01, -7.7570e-01,  1.6956e-01,
          6.2530e-01,  9.6272e-01, -9.7374e-01,  1.0000e+00,  4.3770e-02,
          4.0874e-01,  5.1137e-01,  8.8532e-02, -2.2835e-02, -3.1978e-01,
          9.7829e-01, -9.9313e-01, -4.3599e-01,  4.2975e-01,  9.9727e-01,
          9.9034e-01, -4.1268e-01,  1.3809e-01,  9.9478e-01,  8.1737e-01,
          4.5455e-01, -7.5706e-01, -7.9213e-01,  3.2661e-01,  4.4214e-01,
          4.1534e-01,  9.6873e-01,  9.9786e-01, -9.8159e-01,  9.9678e-01,
          9.9994e-01, -7.9680e-01,  9.4329e-01, -2.2924e-01, -9.9653e-01,
         -9.9489e-01, -9.8972e-01,  6.9869e-01, -8.6204e-01, -5.1368e-01,
         -6.5210e-01,  9.7051e-01, -8.1820e-01,  8.7727e-01, -9.9133e-01,
         -2.0789e-01,  9.9138e-01, -2.7586e-01,  9.9856e-01,  6.6018e-01,
         -9.9543e-01,  1.5215e-01, -6.0283e-01,  9.8019e-01, -5.0130e-01,
          9.9246e-01,  4.3456e-01,  4.9584e-01,  8.3106e-01, -1.0000e+00,
         -7.9427e-01, -3.6073e-01, -7.7528e-01,  8.9853e-01,  9.8772e-01,
          5.2539e-01, -3.6070e-01, -5.8385e-02,  7.0146e-01,  9.9914e-01,
         -9.9236e-01, -5.5297e-01,  7.7900e-01, -9.9364e-01, -9.9724e-01,
          9.7886e-01, -3.6700e-01,  5.8317e-01, -6.4135e-01, -9.6184e-01,
         -3.7618e-02,  1.3213e-01,  9.8460e-01,  2.6646e-01,  9.1801e-01,
         -9.9915e-01,  8.2945e-01, -9.7939e-01,  2.6474e-03,  4.3428e-01,
          6.4185e-01, -7.4259e-01, -9.4198e-01,  6.6336e-01,  9.4775e-01,
         -2.8342e-01,  3.3252e-01, -6.7862e-01,  8.9031e-01,  7.7529e-01,
         -2.2198e-01, -1.0867e-01, -9.8822e-01,  9.9686e-01,  7.5361e-01,
          1.1020e-01,  9.9620e-01, -9.9523e-01,  3.0135e-01,  4.3068e-01,
          1.5686e-01, -1.0000e+00, -3.0560e-01, -9.0196e-01,  8.3643e-01,
         -4.5869e-01,  9.9221e-01, -9.5472e-01, -6.2964e-02,  1.5931e-01,
         -9.9863e-01,  9.6188e-01, -5.5338e-01,  9.9473e-01, -7.9253e-01,
          7.7925e-01,  9.9729e-01,  4.5961e-01, -9.8862e-01, -9.9931e-01,
          1.5182e-01,  9.9997e-01, -9.8668e-01, -3.9033e-01,  9.9913e-01,
          7.9198e-01, -9.3637e-01, -9.7219e-01, -9.9763e-01, -9.9559e-01,
         -7.0510e-01, -4.6162e-01, -1.1446e-01,  9.8840e-01, -3.1973e-01,
         -2.9803e-01,  9.8532e-01,  9.9998e-01, -4.6495e-01,  4.1253e-01,
          2.8706e-01, -9.5282e-01, -9.9998e-01,  4.1392e-01,  9.9388e-04,
         -9.9932e-01,  9.9634e-01, -9.8967e-01,  1.0000e+00, -7.7038e-01,
          6.6178e-01,  9.0948e-01, -2.9336e-01, -1.9188e-01,  3.1360e-01,
          9.9896e-01,  9.7895e-01, -6.9626e-01,  2.9957e-01, -4.1101e-01,
          4.6430e-01, -5.6387e-01,  8.2304e-01,  5.5155e-01,  3.4525e-01,
         -8.8261e-01, -1.3387e-01, -5.1747e-01, -9.9164e-01,  4.2790e-01,
          5.8490e-01,  9.5711e-01,  1.4041e-01,  8.8506e-01,  9.9261e-01,
         -3.8461e-01,  8.0386e-01, -5.9554e-01, -9.9844e-01,  2.9939e-01,
          5.4077e-01, -4.9458e-01,  6.5518e-01,  8.7005e-01,  9.9200e-01,
         -9.7813e-01,  9.9996e-01, -3.6967e-01,  6.2519e-01,  5.3242e-01,
          9.9953e-01, -9.9990e-01,  3.9473e-01, -4.5451e-01,  6.5825e-01,
          8.1847e-01,  9.9293e-01,  6.6598e-02,  4.3072e-01, -4.1398e-01,
          8.2090e-02,  9.6482e-01,  9.8911e-01,  1.2645e-01, -5.6353e-02,
          8.9163e-01,  6.4408e-01,  9.9002e-01, -4.6839e-01, -3.7076e-01,
         -8.1218e-01,  2.9742e-01,  9.6873e-01,  7.2706e-02, -6.7865e-03,
         -3.6139e-01,  3.5492e-01, -2.0560e-01, -5.2090e-01,  5.4854e-01,
          3.7934e-01,  3.0959e-01, -9.9418e-01, -6.5549e-01,  7.0399e-01,
          9.9768e-01, -9.9171e-01, -5.1928e-01,  9.9184e-01, -4.6645e-01,
          5.1544e-01,  8.8863e-01,  5.8067e-02, -9.5258e-01, -4.1317e-01,
         -9.9840e-01,  4.0927e-02, -6.8958e-01,  5.5002e-01,  8.6650e-01,
          1.2266e-01, -1.6570e-01,  9.0898e-02, -7.8863e-01,  7.8153e-01,
          5.7119e-01,  9.7681e-01,  8.7184e-01, -4.2890e-01, -2.0510e-01,
          6.1690e-01,  4.9376e-01, -3.9510e-01,  9.9446e-01, -9.9013e-01,
          9.9916e-01, -5.5511e-01, -9.9800e-01,  8.6015e-01,  3.5947e-01,
         -9.9882e-01, -2.0606e-01, -9.9982e-01,  9.9250e-01, -8.3251e-01,
          6.9434e-01,  6.9950e-01, -9.9991e-01, -1.0000e+00, -5.0334e-01,
         -6.4284e-01, -2.3603e-01, -8.0309e-01,  9.9534e-01,  1.5388e-01,
          7.5789e-01, -1.8876e-01, -9.5614e-01,  4.5651e-01,  6.6139e-01,
         -5.7446e-02, -9.9823e-01,  3.4970e-01,  5.9826e-01,  5.0540e-01,
         -4.5573e-01, -9.5681e-01,  9.8325e-01, -9.7443e-01,  8.0929e-01,
          9.7422e-01,  5.8732e-01,  7.7499e-01, -9.9559e-01,  9.8553e-01,
         -2.0552e-02,  5.2394e-01, -6.1216e-01, -9.8887e-01,  9.9860e-01,
         -8.3695e-01,  2.1054e-01, -3.8064e-01, -9.9403e-01,  2.5474e-01,
          1.7155e-01, -6.7165e-01, -9.8927e-01,  6.3884e-01, -9.9225e-01,
         -9.9955e-01,  6.8616e-01, -4.4052e-01,  9.9979e-01,  9.8424e-01,
          7.8972e-01, -3.8530e-01, -9.3944e-01,  1.0806e-01, -9.9939e-01,
         -6.9365e-01,  2.1264e-01, -9.8301e-01,  7.8265e-01,  9.7458e-01,
          9.9259e-01, -4.4926e-01, -6.2724e-01, -8.5727e-01,  7.1431e-01,
          9.5077e-01,  8.9012e-01, -8.1954e-01,  3.4116e-01, -2.9396e-01,
         -9.9106e-01, -8.7401e-01,  9.9687e-01,  6.6584e-01,  9.8841e-01,
         -5.5308e-01, -1.5925e-01,  2.6246e-01, -4.8569e-01, -4.2412e-03,
         -9.9999e-01, -7.1471e-01, -4.4133e-01, -9.9826e-01,  9.9817e-01,
         -9.9896e-01, -5.5848e-01,  6.2926e-01,  1.2637e-01,  9.8282e-01,
          7.7568e-01, -9.9905e-01, -9.9871e-01, -9.8457e-01, -2.1744e-01,
          9.9257e-01, -5.2820e-02,  5.2959e-01,  3.7136e-01,  8.2491e-01,
          9.9766e-01, -4.6222e-01,  5.4691e-01,  7.6726e-01,  9.9786e-01,
          6.1361e-01, -9.9624e-01,  6.2854e-01, -9.9769e-01,  3.0867e-01,
          9.6189e-01,  9.7428e-01,  9.8549e-01,  6.7416e-01,  9.9897e-01,
         -9.9939e-01,  9.9991e-01, -9.9862e-01,  6.6149e-01,  9.9911e-01,
         -9.9068e-01,  4.2313e-01, -9.9492e-01,  4.8248e-02, -8.4531e-01,
          4.6001e-01, -8.1793e-01,  9.9391e-01, -9.9374e-01, -9.9567e-01,
          4.2780e-01,  5.4645e-01,  8.3804e-01,  3.9530e-02,  5.3903e-01,
          9.9512e-01,  3.4930e-01,  9.8053e-01, -3.3840e-01,  3.8896e-01,
          1.0000e+00, -4.4561e-01,  3.1001e-01, -9.9626e-01,  9.7062e-01,
          1.3133e-01,  5.7270e-01,  9.3847e-01, -4.6251e-01,  4.9201e-01,
          8.2547e-01, -9.9315e-01, -6.1516e-01, -9.8278e-01, -2.8685e-01,
         -2.2857e-01,  6.1953e-01,  5.6592e-01,  1.9295e-01,  3.0250e-01,
         -9.9049e-01,  4.7806e-01, -9.9536e-01,  9.8773e-01,  1.9362e-01,
          3.2996e-01, -2.8732e-01,  7.9672e-01, -9.5784e-01,  9.9585e-01,
          9.9898e-01, -1.0000e+00,  4.8625e-01,  9.8812e-01,  7.1833e-01,
          9.7948e-01, -9.8858e-01, -4.2404e-01,  7.5967e-01, -5.7348e-01,
          9.8429e-01,  7.7599e-01, -5.4579e-01,  9.7485e-01, -9.9338e-01,
          1.6065e-01, -8.8388e-01, -4.8017e-01,  9.1432e-01, -9.8552e-01,
          3.6012e-01, -2.3056e-01,  4.5164e-01, -9.9822e-01, -8.1553e-01,
         -9.9837e-01, -4.6604e-01,  9.9648e-01,  5.9210e-01,  9.9905e-01,
          6.8530e-01, -6.6040e-01, -9.4691e-02, -9.9792e-01,  3.0526e-01,
          3.8308e-01, -2.8369e-01, -5.0559e-02, -6.7507e-01,  3.9323e-01,
         -6.4471e-01, -9.9907e-01,  3.8718e-01,  7.1557e-03,  2.8559e-01,
          9.0519e-01,  7.3779e-01, -9.7480e-01, -6.4177e-01, -8.9919e-01,
          7.2473e-01, -7.1246e-01, -9.7441e-01,  9.4043e-01, -2.6014e-01,
          9.9916e-01, -9.9403e-01, -7.8587e-01, -9.0498e-01, -6.7437e-01,
         -1.3853e-01,  4.6233e-01,  9.2619e-02,  4.5829e-01, -3.0873e-01,
         -6.2325e-01,  9.8342e-01, -9.9093e-01, -9.9110e-01,  9.9650e-01,
         -5.8760e-01, -9.6743e-02, -3.9513e-01, -6.8297e-01, -6.5163e-01,
         -4.8904e-01,  5.0691e-01, -9.2200e-01, -5.9916e-01, -9.9995e-01,
         -7.4408e-01,  3.5011e-01, -9.8429e-01, -9.3719e-01, -7.5029e-01,
         -1.0000e+00,  9.9319e-01,  9.6080e-01,  9.9767e-01, -9.9854e-01,
          9.4866e-01,  5.6957e-01,  9.9777e-01, -4.3564e-01, -8.0363e-01,
          4.0401e-01,  9.9736e-01,  8.1165e-01, -5.6418e-01,  1.6239e-01,
         -3.0295e-01, -2.3164e-01, -8.2747e-01, -3.4624e-01,  3.6424e-01,
          8.1976e-01, -9.7799e-01, -9.9863e-01,  9.9905e-01, -5.0400e-01,
          9.9008e-01,  7.2547e-01,  2.8944e-01, -9.5664e-01,  7.0221e-01,
         -1.0357e-01, -8.7453e-01, -9.9918e-01,  7.5942e-01, -1.0000e+00,
         -9.9382e-01,  7.4309e-01,  9.9120e-01, -9.9655e-01, -9.7025e-01,
          4.2957e-01, -9.9918e-01, -6.8676e-02, -2.0194e-01,  6.0002e-01,
         -9.9553e-01, -5.8757e-01, -7.8175e-01,  4.7940e-01,  9.8424e-01,
         -9.8006e-01,  5.0472e-01, -7.8733e-01,  9.9655e-01,  5.8023e-02,
          6.0490e-01, -5.6005e-02,  7.9968e-02,  4.1414e-01, -9.7203e-01,
         -1.8991e-01, -9.9579e-01, -9.2973e-01,  9.9308e-01,  9.7989e-01,
         -9.9732e-01, -9.9396e-01,  8.3315e-02, -2.5526e-01,  9.9527e-01,
         -6.4836e-01, -9.9898e-01, -9.9843e-01, -1.2412e-02, -7.9976e-01,
          9.9608e-01, -4.3265e-01,  9.9999e-01,  9.4814e-01,  6.2720e-01,
         -7.2097e-01,  4.3903e-01,  6.5930e-01,  5.4907e-01, -4.0927e-01,
          9.9933e-01,  1.6813e-01,  9.9457e-01]
    }

    encodingExamples2 = {
        1: [-0.4290,  0.5826,  0.9949, -0.9743,  0.7096, -0.4042,  0.9145,  0.1713,
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
        2: [-2.1451e-01,  5.6787e-01,  9.9562e-01, -9.7916e-01,  7.5928e-01,
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
        3: [-0.1696,  0.6456,  0.9928, -0.9531,  0.7475, -0.4578,  0.6849,  0.0665,
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
        4: [-0.0536,  0.6557,  0.9790, -0.9358,  0.3839,  0.4686,  0.7389,  0.0607,
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
        5: [-2.6264e-01,  7.4860e-01,  9.9577e-01, -9.4586e-01,  8.1546e-01,
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


    ev = evaluate(prolog,background,neg2,hs,primitives,encodingExamples2)

    positive = [Atom(b45, [
        Structure(s, [List([F2, 'c', E2, Q2, E2,'h','c',F2, C2, 'q']),
                      List([F2, E2, Q2, E2, F2, C2])])]),
           Atom(b45, [Structure(s, [List(
               ['o', N2, A2, 'z', 'g', 'f']),
               List(
                   [N2,A2])])]),
           Atom(b45, [Structure(s, [List([T2,'r','g','y',T2,P2]),
                                    List([T2,T2,P2])])]),
           Atom(b45, [
               Structure(s, [List(['c',O2,'o','j','j',H2,F2,M2,'g',C2]),
                             List([O2, H2, F2, M2, C2])])]),
           Atom(b45, [Structure(s, [List(['x','u','k','l','w','f',Z2,L2,R2,'h',U2,'t']),
                                    List([Z2,L2,R2,U2])])])]

    positive2 = [Atom(b45, [
            Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, one, zero, hook2a]),
                          List([one, zero])])]),
        Atom(b45, [
        Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, one, hook2a]),
                      List([one])])]),

            Atom(b45, [
                Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, three, hook2a ]),
                              List([three])])]),
            Atom(b45, [
                Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, five,zero ,hook2a ]),
                              List([five,zero])])]),
            Atom(b45, [
                Structure(s, [List(['n', 'u', 'm', 'b', 'e', 'r', hook1a, five, hook2a ]),
                              List([five])])])]
    ev.generateExamples((Clause(head,[]),0), positive2, 4,encoding2)






