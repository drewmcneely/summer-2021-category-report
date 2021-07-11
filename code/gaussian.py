import numpy as np
from scipy.linalg import block_diag
from category import StrictMarkov

class Gaussian(StrictMarkov):
    def __init__(self, matrix, **kwargs):
        (target, source) = matrix.shape

        self.matrix = matrix
        allowed_keys = {"mean", "covariance"}
        self.mean = np.zeros(target)
        self.covariance = np.zeros((target, target))
        self.__dict__.update((k,v) for (k,v) in kwargs.items() if k in allowed_keys)

        assert (self.mean.shape == (target,))
        assert (self.covariance.shape == (target,target))
        eigs, evecs = np.linalg.eig(self.covariance)
        assert all([eig >= 0 for eig in eigs])

    def __str__(self):
        F = str(self.matrix)
        x = str(self.mean.reshape(self.target,1))
        P = str(self.covariance)

        num_blanks = self.target - 1
        beginning = ['N( '] + ['   ']*num_blanks
        middle = ['    ']*num_blanks + [' , ']
        ending = ['']*num_blanks + [' )']

        splt_lines = zip(beginning,\
                x.split('\n'),\
                middle,\
                P.split('\n'),\
                ending)

        return '\n'.join(''.join(z) for z in splt_lines)

    def terminal_matrix(n): return np.empty((0,n))
    def elemental_matrix(n): return np.empty((n,0))

    def unit(): return 0
    # Creator Methods
    @classmethod
    def certain_state(cls, state):
        matrix = Gaussian.elemental_matrix(len(state))
        return cls(matrix=matrix, mean=state)

    @classmethod
    def uncertain_state(cls, mean, covariance):
        matrix = Gaussian.elemental_matrix(len(mean))
        return cls(matrix=matrix, mean=mean, covariance=covariance)

    @classmethod
    def affine(cls, matrix, vector):
        return cls(matrix=matrix, vector=vector)

    @classmethod
    def linear(cls, matrix): return cls(matrix=matrix)

    # Category
    @classmethod
    def identity(cls, n): return cls(matrix=np.identity(n))

    def compose(self, other):
        assert (self.source == other.target)

        F = other.matrix
        G = self.matrix
        H = G@F

        x = other.mean
        y = self.mean
        z = G@x + y

        P = other.covariance
        Q = self.covariance
        R = G@P@G.T + Q

        return Gaussian(matrix=H, mean=z, covariance=R)

    @property
    def source(self): return self.matrix.shape[1]
    @property
    def target(self): return self.matrix.shape[0]

    # StrictMonoidal
    def bimap(self,other):
        matrix = block_diag(self.matrix, other.matrix)
        mean = np.concatenate((self.mean, other.mean))
        covariance = block_diag(self.covariance, other.covariance)
        return Gaussian(matrix=matrix, mean=mean, covariance=covariance)

    # Symmetric
    @classmethod
    def swapper(cls, n1, n2):
        z21 = np.zeros((n2,n1))
        z12 = np.zeros((n1,n2))
        i1 = np.identity(n1)
        i2 = np.identity(n2)
        return cls(matrix=np.block([[z21,i2],[i1,z12]]))

    @staticmethod
    def factor(xy, x): return xy - x

    # StrictMarkov
    @classmethod
    def copier(cls, n):
        I = np.identity(n)
        return cls(matrix=np.vstack((I,I)))
    @classmethod
    def discarder(cls, n): return cls(matrix=Gaussian.terminal_matrix(n))

    def recovery_from1(self, nx):
        assert (self.source == 0)

        nxy = self.target
        ny = nxy - nx

        M = self.covariance

        px = self.proj1(nx)
        x = px.mean
        P = px.covariance
        Pinv = np.linalg.inv(P)

        py = self.proj2(ny)
        y = py.mean
        R = py.covariance
        Q = M[:ny,ny:]

        F = Q @ Pinv
        z = y - Q @ Pinv @ x
        S = R - Q @ Pinv @ Q.T

        return Gaussian(matrix=F, mean=z, covariance=S)

if __name__ == "__main__":
    x0 = np.array([3,2])
    P0 = np.identity(2)
    prior = Gaussian.uncertain_state(x0, P0)

    F = np.array([[0,1],[1,2]])
    Q = np.array([[1,0],[0,2]])
    v = np.zeros(2)
    dynamics = Gaussian(matrix=F, mean=v, covariance=Q)

    H = np.array([[1,0]])
    R = np.array([[1]])
    w = np.zeros(1)
    instrument = Gaussian(matrix=H, mean=w, covariance=R)

    z = np.array([3])
    measurement = Gaussian.certain_state(z)

    posterior = prior.update(dynamics, instrument, measurement)

    print("Posterior from categorical filter:")
    print(posterior)

    xbar = F@x0
    Pbar = F@P0@F.T + Q

    ytilde = z - H@xbar
    S = H@Pbar@H.T + R
    K = Pbar@H.T@np.linalg.inv(S)
    
    xhat = xbar + K@ytilde
    Phat = Pbar - K@H@Pbar

    print("Posterior from regular Kalman filter:")
    print(Gaussian.uncertain_state(xhat, Phat))
