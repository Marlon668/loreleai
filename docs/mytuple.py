from pylo.language.commons import Clause


class mytuple:

    def __init__(self,value:float,neural:float,expansion:float,clause:Clause):
        self._value = value
        self._clause = clause
        self._neural = neural
        self._expansion = expansion

    def __lt__(self, nxt):
        return self._value<=nxt._value

    def __repr__(self):
        return self._clause.__repr__()