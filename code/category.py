from abc import ABC, abstractmethod

class Category(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs): pass

    @abstractmethod
    def compose(self, other): pass
    def __matmul__(self,other): return self.compose(other)

    @classmethod
    @abstractmethod
    def identity(cls, obj): pass

    @property
    @abstractmethod
    def source(self): pass
    @property
    @abstractmethod
    def target(self): pass

class StrictMonoidal(Category):
    @abstractmethod
    def bimap(self, other): pass
    def __and__(self, other): return self.bimap(other)

    @staticmethod
    @abstractmethod
    def unit(): pass

    @staticmethod
    @abstractmethod
    def factor1(xy, x): pass
    @staticmethod
    @abstractmethod
    def factor2(xy, y): pass

class StrictSymmetric(StrictMonoidal):
    @classmethod
    @abstractmethod
    def swapper(cls, obj1, obj2): pass
    def swapped(self, obj1, obj2):
        return type(self).swapper(obj1, obj2) @ self

    @staticmethod
    @abstractmethod
    def factor(xy, x): pass
    def factor1(xy, x): return factor(xy, x)
    def factor2(xy, x): return factor(xy, x)


class StrictMarkov(StrictSymmetric):
    @classmethod
    @abstractmethod
    def copier(cls, obj): pass
    def copied(self):
        return type(self).copier(self.target) @ self

    @classmethod
    @abstractmethod
    def discarder(cls, obj): pass

    @classmethod
    def projector1(cls, obj1, obj2):
        return cls.identity(obj1) & cls.discarder(obj2)
    def proj1(self, x):
        cls = type(self)
        y = cls.factor(self.target, x)
        return cls.projector1(x,y) @ self
    @classmethod
    def projector2(cls, obj1, obj2):
        return cls.projector1(obj2, obj1) @ cls.swapper(obj1,obj2)
    def proj2(self, y):
        cls = type(self)
        x = cls.factor(self.target, y)
        return cls.projector2(x,y) @ self

    def appendor(self):
        cls = type(self)
        source = self.source
        return (cls.identity(source) & self) @ cls.copier(source)
    def integrate1_through(self, morphism):
        return morphism.appendor() @ self

    def prependor(self):
        cls = type(self)
        l = self.target
        r = self.source
        return cls.swapper(r,l) @ self.appendor()
    def integrate2_through(self, morphism):
        return morphism.prependor() @ self

    def __mul__(self, other):
        cls = type(self)
        unit = cls.unit()
        if other.source == cls.unit():
            assert other.target == self.source
            return other.integrate2_through(self)
        elif self.source == cls.unit():
            assert self.target == other.source
            return self.integrate1_through(other)
        else: raise ValueError

    @abstractmethod
    def recovery_from1(self, x): pass
    def recovery_from2(self, y):
        xy = self.target
        x = type(self).factor(xy, y)
        return self.swapped(x,y).recovery_from1(y)
    def __truediv__(self, x):
        return self.recovery_from1(x)
    def __floordiv__(self, x):
        return self.recovery_from2(x)

    def bayes_invert(probability, conditional):
        return (conditional * probability) / conditional.target

    def update(self, dynamics, instrument, measurement):
        prior = dynamics @ self
        updater = prior.bayes_invert(instrument)
        return updater @ measurement
