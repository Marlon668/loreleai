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
#from docs.SWIProlog2 import SWIProlog
from loreleai.reasoning.lp.prolog import SWIProlog, Prolog
from MyModel import encoding2
from MyModel import Network


class decoding:
    def __init__(self, primitives):
        encodingprimitives = []
        self._primitives = primitives

    def decode(self, vector):
        decode = []
        for i in range(0,len(vector)):
            decode.append(mytuple(vector[i],0,0,self._primitives[i]))
        return decode





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
    write1 = c_pred("write1", 2)
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
    network = Network.Network(3, [2532, 10, 17])

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
                  write1(c_var('X'), c_var('Y')), write1(c_var('Y'), c_var('X')), write1(c_var('Y'), c_var('Z')),
                  copyskip1(c_var('X'), c_var('Y')),
                  copyskip1(c_var('Y'), c_var('X')), copyskip1(c_var('Y'), c_var('Z'))]
    # clause = b45("A") <= is_uppercase("A"),mk_uppercase("A","B"),is_lowercase("B"),mk_lowercase("B","C"),mk_lowercase("C","D"),is_space("D"),skip1("D","E")
    encoding = encoding2.encoding(primitives2)
    head = Atom(b45, [A])
    body1 = Atom(is_uppercase, [A])
    body2 = Atom(mk_uppercase, [A, B])
    body3 = Atom(is_lowercase, [B])
    body4 = Atom(mk_lowercase, [B, C])
    body5 = Atom(mk_lowercase, [C, D])
    body6 = Atom(is_space, [D])
    body7 = Atom(skip1, [D, C])
    clause = Clause(head, Body(body4, body2, body3, body1))
    vector = [-0.2781, 0.6987, 0.9962, -0.9904, 0.7330, 0.2802, 0.9755, 0.0176,
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
            0.5506, -0.2584, 0.1526, -0.1592, -0.2938, 0.9982, -0.8250, 0.9892]
    encoding = [*encoding.encode(clause),*vector]
    print(network.FeedForward(encoding))
    primitives = [not_space, mk_uppercase, mk_lowercase, is_empty, is_space, is_uppercase, not_uppercase, is_lowercase,
                  not_lowercase, is_letter, not_letter, is_number, not_number, skip1, copy1, write1, copyskip1]
    decoding = decoding(primitives)
    print(decoding.decode(network.FeedForward(encoding)))