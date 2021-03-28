from pylo.language.commons import Clause


class mytuple:

    def __init__(self,value:float,neural:float,expansion:float,clause:Clause):
        self._value = value
        self._clause = clause
        self._neural = neural
        self._expansion = expansion

    def __lt__(self, nxt):
        #if self._value == nxt._value:
        #    if self._expansion == nxt._expansion:
        #        return self._neural<=nxt._neural
        #    else:
        #        return self._expansion<=nxt._expansion
        #else:
        #    return self._value<=nxt._value
        return self._value<=nxt._value

    def __repr__(self):
        return self._clause.__repr__()