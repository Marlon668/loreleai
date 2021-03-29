import typing
from abc import ABC, abstractmethod

from orderedset import OrderedSet
import time

from pylo.engines.prolog.SWIProlog import Pair

from docs.mytuple import mytuple
from loreleai.language.lp import c_var, c_pred, Clause, Procedure, Atom, Body, Not, List, c_const, c_functor, c_literal, Structure
from loreleai.learning.hypothesis_space import TopDownHypothesisSpace
from loreleai.learning.language_filtering import has_singleton_vars, has_duplicated_literal
from loreleai.learning.language_manipulation import plain_extension
from loreleai.learning.task import Task, Knowledge
from loreleai.reasoning.lp.prolog import SWIProlog, Prolog
import numpy as np


class encoding:
    def __init__(self,primitives):
        encodingprimitives = []
        for prim in primitives:
            for prim2 in primitives:
                element = [prim,prim2]
                encodingprimitives.append(element)
        self._dictionary = {}
        for i in range(0,len(encodingprimitives)):
            self._dictionary[tuple(encodingprimitives[i])] = i



    def primitives(self,primitives,x):
        self._encodingprimitives = []
        self._primitivesindices = {}
        index = 0
        for prim in primitives:
            self._primitivesindices[prim] = index
            index = index + x
            for i in range(0,x):
                self._encodingprimitives.append(0)

    def variables(self,y,l):
        number = min(26,y)
        self._encodingvariables = []
        self._variablesindices = {}
        index = 0
        for code in range(ord('A'), ord('A') + number):
            self._variablesindices[c_var(chr(code))] = index
            index = index + l
            for i in range(0, l):
                self._encodingvariables.append(0)

    def addproblem(self,problem,x):
        index = len(self._encodingProblems)
        self._problemindices[problem] = index
        for i in range(0,x):
            self._encodingProblems.append(0)

    def encode2(self,clause:Clause):
        head = clause.get_head()
        if not self._problemindices.__contains__(head.get_predicate()):
            self.addproblem(head.get_predicate(),10)
        problems = self._encodingProblems.copy()
        primitives = self._encodingprimitives.copy()
        variables = self._encodingvariables.copy()
        problems[self._problemindices[head.get_predicate()]]=1
        variables[self._variablesindices[head.get_variables()[0]]] =1
        cijfer = 3
        for literal in clause.get_literals():
            startindexliteral = self._primitivesindices[literal.get_predicate()]
            fillin = False
            while fillin == False:
                if(primitives[startindexliteral]==0):
                    primitives[startindexliteral]=cijfer
                    fillin = True
                else:
                    startindexliteral +=1
            startindexvariable = self._variablesindices[literal.get_variables()[0]]
            fillin = False
            while fillin == False:
                if (variables[startindexvariable] == 0):
                    variables[startindexvariable] = cijfer
                    fillin = True
                else:
                    startindexvariable +=1
            if(len(literal.get_variables()) ==2):
                startindexvariable = self._variablesindices[literal.get_variables()[1]]
                fillin = False
                while fillin == False:
                    if (variables[startindexvariable] == 0):
                        variables[startindexvariable] = cijfer +1
                        fillin = True
                    else:
                        startindexvariable += 1
            cijfer += 2
        return problems + primitives + variables

    def encode(self,clause:Clause):
        encodingClause = np.zeros(1850)
        vars = []
        set = {}
        index = 0
        for lit in [clause.get_head(),*clause.get_literals()]:
            var = ''
            for variable in lit.get_variables():
                var += variable.get_name()
            if var in set:
                index = set[var]
                vars[index][1].append(lit)
            else:
                set[var] = index
                index +=1
                list = [lit]
                if len(var) == 1:
                    value = 100000*(ord(var)-64)+3500*(ord(var)-64)+130*(ord(var)-64)
                else:
                    if len(var) == 2:
                        if ord(var[0])<=ord(var[-1]):
                            value = 100000*(ord(var[0])-64)+3500*(ord(var[0])-64)+130*(ord(var[-1])-64)
                        else:
                            value = 100000*(ord(var[0])-64)+3500*(ord(var[0])-64)+130*(ord(var[-1])-64) + 1
                    else:
                        if ord(var[0]) <= ord(var[1]) <= ord(var[2]):
                            value = 100000 * (ord(var[0]) - 64) + 3500 * (ord(var[1]) - 64) + 130 * (
                                        ord(var[2]) - 64)
                        else:
                            if ord(var[0]) <= ord(var[2]) <= ord(var[1]):
                                value = 100000 * (ord(var[0]) - 64) + 3500 * (ord(var[2]) - 64) + 130 * (
                                            ord(var[1]) - 64) + 1
                            else:
                                if ord(var[1]) <= ord(var[0]) <= ord(var[2]):
                                    value = 100000 * (ord(var[1]) - 64) + 3500 * (ord(var[0]) - 64) + 130 * (
                                            ord(var[2]) - 64) + 2
                                else:
                                    if ord(var[1]) <= ord(var[0]) <= ord(var[2]):
                                        value = 100000 * (ord(var[1]) - 64) + 3500 * (ord(var[0]) - 64) + 130 * (
                                                ord(var[2]) - 64) + 3
                                    else:
                                        if ord(var[2]) <= ord(var[0]) <= ord(var[1]):
                                            value = 100000 * (ord(var[2]) - 64) + 3500 * (
                                                        ord(var[0]) - 64) + 130 * (
                                                            ord(var[1]) - 64) + 4
                                        else:
                                            value = 100000 * (ord(var[2]) - 64) + 3500 * (
                                                    ord(var[1]) - 64) + 130 * (
                                                            ord(var[0]) - 64) + 5

                vars.append((value,list))
        vars.sort()
        newClause = []
        for v in vars:
            newClause = newClause + v[1]
        encoding = [self.variableSubstition(newClause[i:i+2]) for i in range(len(newClause) - 1)]
        for element in encoding:
            encodingClause[self._dictionary.get(tuple(element))] += 1
        encodingClause[1849] = len(clause.get_variables())
        return encodingClause

    def encode2(self,clause:Clause):
        encodingClause = np.zeros(1850)
        vars = []
        set = {}
        index = 0
        for lit in [clause.get_head(), *clause.get_literals()]:
            var = ''
            for variable in lit.get_variables():
                var += variable.get_name()
            if var in set:
                index = set[var]
                vars[index]._clause.append(lit)
            else:
                set[var] = index
                index += 1
                list = [lit]
                if len(var) == 1:
                    value = 100000 * (ord(var) - 64) + 3500 * (ord(var) - 64) + 130 * (ord(var) - 64)
                else:
                    if len(var) == 2:
                        if ord(var[0]) <= ord(var[-1]):
                            value = 100000 * (ord(var[0]) - 64) + 3500 * (ord(var[0]) - 64) + 130 * (ord(var[-1]) - 64)
                        else:
                            value = 100000 * (ord(var[0]) - 64) + 3500 * (ord(var[0]) - 64) + 130 * (ord(var[-1]) - 64)+1
                    else:
                        if ord(var[0]) <= ord(var[1]) <= ord(var[2]):
                            value = 100000 * (ord(var[0]) - 64) + 3500 * (ord(var[1]) - 64) + 130 * (
                                        ord(var[2]) - 64)
                        else:
                            if ord(var[0]) <= ord(var[2]) <= ord(var[1]):
                                value = 100000 * (ord(var[0]) - 64) + 3500 * (ord(var[2]) - 64) + 130 * (
                                            ord(var[1]) - 64) + 1
                            else:
                                if ord(var[1]) <= ord(var[0]) <= ord(var[2]):
                                    value = 100000 * (ord(var[1]) - 64) + 3500 * (ord(var[0]) - 64) + 130 * (
                                            ord(var[2]) - 64) + 2
                                else:
                                    if ord(var[1]) <= ord(var[0]) <= ord(var[2]):
                                        value = 100000 * (ord(var[1]) - 64) + 3500 * (ord(var[0]) - 64) + 130 * (
                                                ord(var[2]) - 64) + 3
                                    else:
                                        if ord(var[2]) <= ord(var[0]) <= ord(var[1]):
                                            value = 100000 * (ord(var[2]) - 64) + 3500 * (
                                                        ord(var[0]) - 64) + 130 * (
                                                            ord(var[1]) - 64) + 4
                                        else:
                                            value = 100000 * (ord(var[2]) - 64) + 3500 * (
                                                    ord(var[1]) - 64) + 130 * (
                                                            ord(var[0]) - 64) + 5
                vars.append(mytuple(value,0,0,list))
        #print(vars)
        vars.sort()


        newClause = []
        for v in vars:
            newClause = newClause + v._clause

        encoding = [self.variableSubstition(newClause[i:i+2]) for i in range(len(newClause) - 1)]
        for element in encoding:
            encodingClause[self._dictionary.get(tuple(element))] += 1
        encodingClause[1849] = len(clause.get_variables())
        return encodingClause

    def variableSubstition(self,list):
        var1 = list[0].get_variables()
        var2 = list[1].get_variables()
        if var1[0]==var2[0]:
            newList = []
            if len(var1) == 1:
                newList.append(list[0].get_predicate()('X'))
            else:
                if len(var1) == 2:
                    newList.append(list[0].get_predicate()('X','Y'))
                else:
                    newList.append(list[0].get_predicate()('X','Y','Z'))
            if len(var2) == 1:
                newList.append(list[1].get_predicate()('X'))
            else:
                if len(var2) == 2:
                    newList.append(list[1].get_predicate()('X', 'Y'))
                else:
                    newList.append(list[1].get_predicate()('X', 'Y', 'Z'))
            return newList
        if (len(var1)<3)&(len(var2)<3):
            if len(var2)==2:
                if(var1[0]==var2[1]):
                    newList = []
                    if len(var1) == 1:
                        newList.append(list[0].get_predicate()('X'))
                    else:
                        newList.append(list[0].get_predicate()('X', 'Y'))
                    newList.append(list[1].get_predicate()('Y','X'))
                    return newList
                if len(var1)>1:
                    if var1[1] == var2[1]:
                        newList = []
                        newList.append(list[0].get_predicate()('Y', 'X'))
                        newList.append(list[1].get_predicate()('Y', 'X'))
                        return newList
            if len(var1)>1:
                if var1[1]==var2[0]:
                    newList = []
                    newList.append(list[0].get_predicate()('Y','X'))
                    if len(var2) == 1:
                        newList.append(list[1].get_predicate()('X'))
                    else:
                        newList.append(list[1].get_predicate()('X','Y'))
                    return newList
            newList = []
            if len(var1) == 1:
                newList.append(list[0].get_predicate()('Y'))
            else:
                newList.append(list[0].get_predicate()('Y', 'Z'))
            if len(var2) == 1:
                newList.append(list[1].get_predicate()('Y'))
            else:
                newList.append(list[1].get_predicate()('Y', 'Z'))
            return newList
        else:
            if len(var2)>1:
                if var1[0]== var2[1]:
                    newList = []
                    if len(var2)==2:
                       newList.append(list[0].get_predicate()('X','Y','Z'))
                       newList.append(list[1].get_predicate()('Y','X'))
                    else:
                       newList = []
                       if len(var1) == 1:
                           newList.append(list[0].get_predicate()('X'))
                       else:
                           if len(var1) == 2:
                               newList.append(list[0].get_predicate()('X', 'Y'))
                           else:
                               newList.append(list[0].get_predicate()('X', 'Y', 'Z'))
                       newList.append(list[1].get_predicate()('Y', 'X', 'Z'))
                    return newList
                if len(var2)>2:
                    if var1[0]==var2[2]:
                        newList = []
                        if len(var1) == 1:
                            newList.append(list[0].get_predicate()('X'))
                        else:
                            if len(var1) == 2:
                                newList.append(list[0].get_predicate()('X', 'Y'))
                            else:
                                newList.append(list[0].get_predicate()('X', 'Y', 'Z'))
                        newList.append(list[1].get_predicate()('Y', 'Z', 'X'))
                        return newList

            if len(var1)>1:
                if var1[1]==var2[0]:
                    if len(var1)==2:
                        newList = []
                        newList.append(list[0].get_predicate()('Y','X'))
                        newList.append(list[1].get_predicate()('X','Y','Z'))
                    else:
                        newList = []
                        newList.append(list[0].get_predicate()('Y','X','Z'))
                        if len(var2)==1:
                            newList.append(list[1].get_predicate()('X'))
                        else:
                            if len(var2)==2:
                                newList.append(list[1].get_predicate()('X','Y'))
                            else:
                                newList.append(list[1].get_predicate()('X','Y','Z'))
                        return newList
                if len(var2) > 1:
                    if var1[1]==var2[1]:
                        if len(var2) == 2:
                            newList = []
                            newList.append(list[0].get_predicate()('Y', 'X', 'Z'))
                            newList.append(list[1].get_predicate()('Y', 'X'))
                        else:
                            newList = []
                            if len(var1)==2:
                                newList.append(list[0].get_predicate()('Y', 'X'))
                            else:
                                newList.append(list[0].get_predicate()('Y', 'X', 'Z'))
                            newList.append(list[1].get_predicate()('Y', 'X', 'Z'))
                        return newList
                    if len(var2)== 3:
                        if var1[1]==var2[2]:
                            newList = []
                            if len(var1) == 2:
                                newList.append(list[0].get_predicate()('Y','X'))
                            else:
                                newList.append(list[0].get_predicate()('Y', 'X', 'Z'))
                            newList.append(list[1].get_predicate()('Y', 'Z', 'X'))
                            return newList

                if len(var1) > 2:
                    if var1[2] == var2[0]:
                        newList = []
                        newList.append(list[0].get_predicate()('Y', 'Z', 'X'))
                        if len(var2) == 1:
                            newList.append(list[1].get_predicate()('X'))
                        else:
                            if len(var2) == 2:
                                newList.append(list[1].get_predicate()('X','Y'))
                            else:
                                newList.append(list[1].get_predicate()('X', 'Y', 'Z'))
                        return newList
                    if len(var2)>1:
                        if var1[2] == var2[1]:
                            newList = []
                            newList.append(list[0].get_predicate()('Y', 'Z', 'X'))
                            if len(var2) == 2:
                                newList.append(list[1].get_predicate()('Y', 'X'))
                            else:
                                newList.append(list[1].get_predicate()('Y', 'X', 'Z'))
                        if len(var2) == 3:
                            if var1[2] == var2[2]:
                                newList = []
                                newList.append(list[0].get_predicate()('Y', 'Z', 'X'))
                                newList.append(list[1].get_predicate()('Y', 'Z', 'X'))
                                return newList
            if len(var1)==1:
                newList = []
                newList.append(list[0].get_predicate()('Y'))
                newList.append(list[1].get_predicate()('Y', 'Z', 'U'))
                return newList
            if len(var1)==2:
                newList = []
                newList.append(list[0].get_predicate()('Y','Z'))
                newList.append(list[1].get_predicate()('Y', 'Z', 'U'))
                return newList
            if len(var1) == 3 & len(var2)==3:
                newList = []
                newList.append(list[0].get_predicate()('Y','Z','U'))
                newList.append(list[1].get_predicate()('Y', 'Z', 'U'))
                return newList
            if len(var2) == 1:
                newList = []
                newList.append(list[0].get_predicate()('Y','Z','U'))
                newList.append(list[1].get_predicate()('Y'))
                return newList
            if len(var2) == 2:
                newList = []
                newList.append(list[0].get_predicate()('Y','Z','U'))
                newList.append(list[1].get_predicate()('Y','Z'))
                return newList




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
        expansion_hooks_reject=[lambda x, y: has_singleton_vars(x, y),
                                lambda x, y: has_duplicated_literal(x, y)],
        recursive_procedures=True)
    primitives = [b45(c_var('X')),b45(c_var('Y')),not_space(c_var('X')), not_space(c_var('Y')), mk_uppercase(c_var('X'), c_var('Y')),
                  mk_uppercase(c_var('Y'), c_var('X')),
                  mk_uppercase(c_var('Y'), c_var('Z')), mk_lowercase(c_var('X'), c_var('Y')),
                  mk_lowercase(c_var('Y'), c_var('X')),
                  mk_lowercase(c_var('Y'), c_var('Z')), is_empty(c_var('X')), is_empty(c_var('Y')), is_space(c_var('X'))
        , is_space(c_var('Y')), is_uppercase(c_var('X')), is_uppercase(c_var('Y')), not_uppercase(c_var('X')), not_uppercase(c_var('Y')),
        is_lowercase(c_var('X')), is_lowercase(c_var('Y')), not_lowercase(c_var('X')), not_lowercase(c_var('Y')), is_letter(c_var('X')),
                  is_letter(c_var('Y')), not_letter(c_var('X')), not_letter(c_var('Y')), is_number(c_var('X')),
                  is_number(c_var('Y')), not_number('X'), not_number('Y'), skip1(c_var('X'), c_var('Y')),
                  skip1(c_var('Y'), c_var('X')),
                  skip1(c_var('Y'), c_var('Z')), copy1(c_var('X'), c_var('Y')), copy1(c_var('Y'), c_var('X')),
                  copy1(c_var('Y'), c_var('Z')),
                  write1(c_var('X'), c_var('Y'),c_var('Z')), write1(c_var('Y'), c_var('X'),c_var('Z')), write1(c_var('Y'), c_var('Z'),c_var('X')),
                  write1(c_var('Y'), c_var('Z'),c_var('U')),copyskip1(c_var('X'), c_var('Y')),
                  copyskip1(c_var('Y'), c_var('X')), copyskip1(c_var('Y'), c_var('Z'))]
    print('prim : ' , primitives)
    print(len(primitives))
    head = Atom(b45, [A])
    body1 = Atom(mk_uppercase, [B,A])
    body2 = Atom(copyskip1, [A,D])
    body3 = Atom(not_number, [B])
    body4 = Atom(is_lowercase, [B])
    body5 = Atom(is_lowercase, [C,D])
    body6 = Atom(write1, [E,c_var('F'),B])
    body9 = Atom(skip1, [C, D])
    body8 = Atom(is_space, [D])
    body7 = Atom(skip1,[D,C])
    clause = Clause(head, Body(body2,body3,body4,body9,body1,body6))
    print(clause)

    #clause = b45("A") <= is_uppercase("A"),mk_uppercase("A","B"),is_lowercase("B"),mk_lowercase("B","C"),mk_lowercase("C","D"),is_space("D"),skip1("D","E")
    #print(clause)
    encoding = encoding(primitives)
    tic = time.perf_counter()
    encoding.encode(clause)
    toc = time.perf_counter()
    #tic = time.perf_counter()
    #encoding.encode(clause)
    #toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.10f} seconds")
    print(encoding.encode(clause))
