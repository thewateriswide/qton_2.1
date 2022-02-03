# 
# This code is part of Qton.
# 
# Qton Version: 2.1.0
# 
# This file: qton\simulators\statevector.py
# Author(s): Yunheng Ma
# Timestamp: 2022-02-03 10:18:24
# 

import numpy as np


__all__ = ["normalize",
           "random_qubit",
           "random_vector",
           "random_gate",
           "expectation",
           "Qstatevector",
           ]


def normalize(svec):
    '''
    To normalize a satevector.
    
    -In(1):
        1. svec --- statevector to be normalized.
            type: numpy.ndarray, 1D, complex
            
    -Return(1):
        1. --- normalized statevector.
            type: numpy.ndarray, 1D, complex
    '''
    norm2 = np.einsum('i,i->', svec, svec.conj()).real
    norm = np.sqrt(norm2)
    return svec / norm


def random_qubit(global_phase=True):
    '''
    Returns a random single-qubit statevector.
    
    $$
    |\psi\rangle = {\rm e}^{{\rm i}\gamma}\big(\cos\frac{\theta}{2}|0\rangle + 
        {\rm e}^{{\rm i}\phi}\sin\frac{\theta}{2}|1\rangle\big)
    $$
    
    -In(1):
        1. global_phase --- global phase appears?
            type: bool

    -Return(1):
        1. svec --- single-qubit statevector.
            type: numpy.ndarray, 1D, complex
    '''
    x, y, z = np.random.standard_normal(3)
    r = np.sqrt(x**2 + y**2 + z**2)  # fails if r==0
    rx, ry, rz = x/r, y/r, z/r

    theta = np.arccos(rz)
    phi = np.arctan(ry/rx)

    svec = np.zeros(2, complex)
    svec[0] = np.cos(0.5*theta)
    svec[1] = np.exp(1j*phi)*np.sin(0.5*theta)

    if global_phase:
        gamma = np.random.random() * np.pi * 2
        svec *= np.exp(1j*gamma)
    return svec


def random_vector(num_qubits=1):
    '''
    Returns a uniformly random qubit statevector in Hilbert space.

    The idea is about to replace each basis with a random qubit statevector.

    -In(1):
        1. num_qubits --- number of qubits.
            type: int

    -Return(1):
        1. svec --- qubit statevector.
            type: numpy.ndarray, 1D, complex
    '''
    svec = np.ones(2**num_qubits, complex)
    for i in range(num_qubits):
        k = 2**(i+1)
        for j in range(2**(num_qubits-i-1)):
            tmp = np.kron(random_qubit(), np.ones(2**i))
            svec[j*k:j*k+k] *= tmp
    return svec


def random_gate(num_qubits=1):
    '''
    Returns a uniformly random qubit gate.

    The idea is about to randomly pick up a group of bases.
    
    -In(1):
        1. num_qubits --- number of qubits.
            type: int
            
    -Return(1):
        1. --- qubit gate matrix.
            type: numpy.ndarray, 2D, complex
    '''
    base = []
    # fails if these vectors are linear dependent
    for i in range(2**num_qubits):
        base.append(random_vector(num_qubits))
    
    # Schmidt orthogonalization
    for i in range(2**num_qubits):
        for j in range(i):
            base[i] = base[i] - np.dot(base[j].conj(), base[i]) * base[j]
        base[i] = normalize(base[i])
    return np.vstack(base)


def expectation(gate, svec):
    '''
    Expectation of a quantum gate on a statevector.
    
    -In(2):
        1. gate --- quantum gate.
            type: numpy.ndarray, 2D, complex
        2. svec --- statevector.
            type: numpy.ndarray, 1D, complex

    -Return(1):
        1. --- expectation value.
            type: complex
    '''
    return np.einsum('i,ij,j->', svec.conj(), gate, svec)


# alphabet
alp = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 
    'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
      ]


from ._basic_qcircuit_ import _Basic_qcircuit_


class Qstatevector(_Basic_qcircuit_):
    '''
    Quantum circuit represented by circuit statevector.

    -Attributes(3):
        1. backend --- how to execute the circuit; 'statevector', 
            'density_matrix', 'unitary', or 'superoperator'.
            type: str
        2. num_qubits --- number of qubits.
            type: int
        3. state --- circuit state representation.
            type: numpy.ndarray, complex        
    
    -Methods(60):
        1. __init__(self, num_qubits=1)
        2. _apply_(self, op, *qubits)
        3. apply(self, op, *qubits)
        4. measure(self, qubit, delete=False)
        5. add_qubit(self, num_qubits=1)
        6. sample(self, shots=1024, output='memory')
        7. copy(self)
        8. i(self, qubits)
        9. x(self, qubits)
        10. y(self, qubits)
        11. z(self, qubits)
        12. h(self, qubits)
        13. s(self, qubits)
        14. t(self, qubits)
        15. sdg(self, qubits)
        16. tdg(self, qubits)
        17. rx(self, theta, qubits)
        18. ry(self, theta, qubits)
        19. rz(self, theta, qubits)
        20. p(self, phi, qubits)
        21. u1(self, lamda, qubits)
        22. u2(self, phi, lamda, qubits)
        23. u3(self, theta, phi, lamda, qubits)
        24. u(self, theta, phi, lamda, gamma, qubits)
        25. swap(self, qubit1, qubit2)
        26. cx(self, qubits1, qubits2)
        27. cy(self, qubits1, qubits2)
        28. cz(self, qubits1, qubits2)
        29. ch(self, qubits1, qubits2)
        30. cs(self, qubits1, qubits2)
        31. ct(self, qubits1, qubits2)
        32. csdg(self, qubits1, qubits2)
        33. ctdg(self, qubits1, qubits2)
        34. crx(self, theta, qubits1, qubits2)
        35. cry(self, theta, qubits1, qubits2)
        36. crz(self, theta, qubits1, qubits2)
        37. cp(self, phi, qubits1, qubits2)
        38. fsim(self, theta, phi, qubits1, qubits2)
        39. cu1(self, lamda, qubits1, qubits2)
        40. cu2(self, phi, lamda, qubits1, qubits2)
        41. cu3(self, theta, phi, lamda, qubits1, qubits2)
        42. cu(self, theta, phi, lamda, gamma, qubits1, qubits2)
        43. cswap(self, qubit1, qubit2, qubit3)
        44. ccx(self, qubits1, qubits2, qubits3)
        45. ccy(self, qubits1, qubits2, qubits3)
        46. ccz(self, qubits1, qubits2, qubits3)
        47. cch(self, qubits1, qubits2, qubits3)
        48. ccs(self, qubits1, qubits2, qubits3)
        49. cct(self, qubits1, qubits2, qubits3)
        50. ccsdg(self, qubits1, qubits2, qubits3)
        51. cctdg(self, qubits1, qubits2, qubits3)
        52. ccrx(self, theta, qubits1, qubits2, qubits3)
        53. ccry(self, theta, qubits1, qubits2, qubits3)
        54. ccrz(self, theta, qubits1, qubits2, qubits3)
        55. ccp(self, phi, qubits1, qubits2, qubits3)
        56. cfsim(self, theta, phi, qubits1, qubits2, qubits3)
        57. ccu1(self, lamda, qubits1, qubits2, qubits3)
        58. ccu2(self, phi, lamda, qubits1, qubits2, qubits3)
        59. ccu3(self, theta, phi, lamda, qubits1, qubits2, qubits3)
        60. ccu(self, theta, phi, lamda, gamma, qubits1, qubits2, qubits3)
    '''
    backend = 'statevector'


    def __init__(self, num_qubits=1):
        super().__init__(num_qubits)
        self.state = np.zeros(2**num_qubits, complex)
        self.state[0] = 1.0
        return None


#
# `_apply_` method is realized in two ways below.
# One is based on `numpy.einsum`, another is based on `numpy.tensordot` and 
# `numpy.einsum`. The former has a limitation on gate size and circuit size,
# while the latter doesn't. Each of them has its advantages.
# 

    # def _apply_(self, op, *qubits):
    # # based on `numpy.einsum`
    #     super()._apply_(op, *qubits)
    #     global alp

    #     if 2*op.num_qubits + self.num_qubits > len(alp):
    #         raise Exception('Not enough indices for calculation.')

    #     op_idx = alp[-2*op.num_qubits:]
    #     state_idx = alp[:self.num_qubits]
    #     for i in range(op.num_qubits):
    #         state_idx[self.num_qubits-qubits[i]-1] = op_idx[op.num_qubits+i]
    #     start = ''.join(op_idx) + ',' + ''.join(state_idx)

    #     for i in range(op.num_qubits):
    #         state_idx[self.num_qubits-qubits[i]-1] = op_idx[i]
    #     end = ''.join(state_idx)

    #     rep = op.represent.reshape([2]*2*op.num_qubits)
    #     self.state = self.state.reshape([2]*self.num_qubits)
    #     self.state = np.einsum(start+'->'+end, rep, self.state).reshape(-1)
    #     return None


    def _apply_(self, op, *qubits):
    # based on `numpy.tensordot` and `numpy.einsum`
        super()._apply_(op, *qubits)
        global alp
        
        a_idx = [*range(op.num_qubits, 2*op.num_qubits)]
        b_idx = [self.num_qubits-i-1 for i in qubits]
        if op.category != 'gate': 
            raise Exception('Use "gate", not {}'.format(op.category))
        rep = op.represent.reshape([2]*2*op.num_qubits)
        self.state = self.state.reshape([2]*self.num_qubits)
        self.state = np.tensordot(rep, self.state, axes=(a_idx, b_idx))

        s = ''.join(alp[:self.num_qubits])
        end = s
        start = ''
        for i in range(op.num_qubits):
            start += end[self.num_qubits-qubits[i]-1]
            s = s.replace(start[i], '')
        start = start + s
        self.state = np.einsum(start+'->'+end, self.state).reshape(-1)
        return None


    def measure(self, qubit, delete=False):
        super().measure(qubit, delete)
        
        state = self.state.reshape([2]*self.num_qubits)
        dic = locals()

        string0 = ':, '*(self.num_qubits-qubit-1) + '0, ' + ':, '*qubit
        string1 = ':, '*(self.num_qubits-qubit-1) + '1, ' + ':, '*qubit

        exec('reduced0 = state[' + string0 + ']', dic)
        measured = dic['reduced0'].reshape(-1)
        probability0 = np.einsum('i,i->', measured, measured.conj())

        if np.random.random() < probability0:
            bit = 0
            if delete:
                self.state = measured
                self.num_qubits -= 1
            else:
                exec('state[' + string1 + '] = 0.', dic)
                self.state = dic['state'].reshape(-1)
            self.state /= np.sqrt(probability0)
        else:
            bit = 1
            if delete:
                exec('reduced1 = state[' + string1 + ']', dic)
                self.state = dic['reduced1'].reshape(-1)
                self.num_qubits -= 1
            else:
                exec('state[' + string0 + '] = 0.', dic)
                self.state = dic['state'].reshape(-1)
            self.state /= np.sqrt(1. - probability0)
        return bit