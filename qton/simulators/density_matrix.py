# 
# This code is part of Qton.
# 
# Qton Version: 2.1.0
# 
# This file: qton\simulators\density_matrix.py
# Author(s): Yunheng Ma
# Timestamp: 2022-02-03 10:18:24
# 

import numpy as np


__all__ = ["svec2dmat",
           "random_qubit",
           "expectation",
           "operate",
           "Qdensity_matrix",
           ]


def svec2dmat(svec):
    '''
    Statevector to density matrix.
    
    $$
    \rho = |\psi\rangle \langle\psi|
    $$
    
    -In(1):
        1. svec --- quantum statevector.
            type: numpy.ndarray, 1D, complex
    
    -Return(1):
        1. --- density matrix corresponding to the statevector.
            type: numpy.ndarray, 2D, complex
    '''
    # return np.enisum('i,j->ij', svec, svec.conj())
    return np.outer(svec, svec.conj())


def random_qubit(mixed=True):
    '''
    Returns a random single-qubit density matrix. 
    
    $$
    \rho = \frac{1}{2}(I + r_x \sigma_x + r_y \sigma_y + r_z \sigma_z)
    $$

    $r_x^2 + r_y^2 + r_z^2 < 1$ for mixed state.

    -In(1):
        1. mixed --- a mixed state?
            type: bool

    -Return(1):
        1. dmat --- single-qubit density matrix.
            type: numpy.ndarray, 2D, complex
    '''
    x, y, z = np.random.standard_normal(3)
    r = np.sqrt(x**2 + y**2 + z**2)
    if r == 0:
        rx = ry = rz = 0.
    else:
        rx, ry, rz = x/r, y/r, z/r

    if mixed:
        e = np.random.random()
        rx, ry, rz = rx*e, ry*e, rz*e

    dmat = np.zeros((2,2), complex)
    dmat[0, 0] = 1. + rz
    dmat[0, 1] = rx - 1j*ry
    dmat[1, 0] = rx + 1j*ry
    dmat[1, 1] = 1. - rz 
    dmat *= 0.5
    return dmat
    

def expectation(oper, dmat):
    '''
    Expectation of quantum operations on a density matrix.

    $$
    \langle E \rangle = \sum_k {\rm Tr}(E_k\rho)
    $$
    
    -In(2):
        1. oper --- quantum operations.
            type: list, numpy.ndarray, 2D, complex
        2. dmat --- density matrix of system.
            type: numpy.ndarray, 2D, complex

    -Return(1):
        1. ans --- expectation value.
            type: complex
    '''
    if type(oper) is not list:
        oper = [oper]

    ans = 0.
    for i in range(len(oper)):
        # ans += np.einsum('ij,ji->', oper[i], dmat)
        ans += np.trace(np.matmul(oper[i], dmat))
    return ans


def operate(oper, dmat):
    '''
    Implement a single-qubit or double-qubit quantum operation.

    $$
    \rho' = \sum_k E_k \rho E_k^\dagger
    $$

    -In(2):
        1. oper --- the quantum operations.
            type: list, numpy.ndarray, 2D, complex
        2. dmat --- density matrix.
            type: numpy.ndarray, 2D, complex
            
    -Return(1):
        1. ans --- density matrix after implementation.
            type: numpy.ndarray, 2D, complex
    '''
    if type(oper) is not list:
        oper = [oper]

    n = len(oper)
    ans = np.zeros(dmat.shape, complex)
    for i in range(n):
        # ans += np.einsum('ij,jk,lk->il', oper[i], dmat, oper[i].conj())
        ans += np.matmul(oper[i],
            np.matmul(dmat, oper[i].transpose().conj()))
    return ans


# alphabet
alp = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
      ]
ALP = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
      ]


from ._basic_qcircuit_ import _Basic_qcircuit_
from qton.operators.channels import *
from qton.operators.superop import to_superop


class Qdensity_matrix(_Basic_qcircuit_):
    '''
    Quantum circuit represented by circuit density matrix.

    -Attributes(3):
        1. backend --- how to execute the circuit; 'statevector', 
            'density_matrix', 'unitary', or 'superoperator'.
            type: str
        2. num_qubits --- number of qubits.
            type: int
        3. state --- circuit state representation.
            type: numpy.ndarray, complex        
    
    -Methods(68):
        1. __init__(self, num_qubits=1)
        2. _apply_(self, op, *qubits)
        3. apply(self, op, *qubits)
        4. measure(self, qubit, delete=False)
        5. add_qubit(self, num_qubits=1)
        6. sample(self, shots=1024, output='memory')
        7. copy(self)
        8. reduce(self, qubits)
        9. bit_flip(self, p, qubits)
        10. phase_flip(self, p, qubits)
        11. bit_phase_flip(self, p, qubits)
        12. depolarize(self, p, qubits)
        13. amplitude_damping(self, gamma, qubits)
        14. generalized_amplitude_damping(self, p, gamma, qubits)
        15. phase_damping(self, lamda, qubits)
        16. i(self, qubits)
        17. x(self, qubits)
        18. y(self, qubits)
        19. z(self, qubits)
        20. h(self, qubits)
        21. s(self, qubits)
        22. t(self, qubits)
        23. sdg(self, qubits)
        24. tdg(self, qubits)
        25. rx(self, theta, qubits)
        26. ry(self, theta, qubits)
        27. rz(self, theta, qubits)
        28. p(self, phi, qubits)
        29. u1(self, lamda, qubits)
        30. u2(self, phi, lamda, qubits)
        31. u3(self, theta, phi, lamda, qubits)
        32. u(self, theta, phi, lamda, gamma, qubits)
        33. swap(self, qubit1, qubit2)
        34. cx(self, qubits1, qubits2)
        35. cy(self, qubits1, qubits2)
        36. cz(self, qubits1, qubits2)
        37. ch(self, qubits1, qubits2)
        38. cs(self, qubits1, qubits2)
        39. ct(self, qubits1, qubits2)
        40. csdg(self, qubits1, qubits2)
        41. ctdg(self, qubits1, qubits2)
        42. crx(self, theta, qubits1, qubits2)
        43. cry(self, theta, qubits1, qubits2)
        44. crz(self, theta, qubits1, qubits2)
        45. cp(self, phi, qubits1, qubits2)
        46. fsim(self, theta, phi, qubits1, qubits2)
        47. cu1(self, lamda, qubits1, qubits2)
        48. cu2(self, phi, lamda, qubits1, qubits2)
        49. cu3(self, theta, phi, lamda, qubits1, qubits2)
        50. cu(self, theta, phi, lamda, gamma, qubits1, qubits2)
        51. cswap(self, qubit1, qubit2, qubit3)
        52. ccx(self, qubits1, qubits2, qubits3)
        53. ccy(self, qubits1, qubits2, qubits3)
        54. ccz(self, qubits1, qubits2, qubits3)
        55. cch(self, qubits1, qubits2, qubits3)
        56. ccs(self, qubits1, qubits2, qubits3)
        57. cct(self, qubits1, qubits2, qubits3)
        58. ccsdg(self, qubits1, qubits2, qubits3)
        59. cctdg(self, qubits1, qubits2, qubits3)
        60. ccrx(self, theta, qubits1, qubits2, qubits3)
        61. ccry(self, theta, qubits1, qubits2, qubits3)
        62. ccrz(self, theta, qubits1, qubits2, qubits3)
        63. ccp(self, phi, qubits1, qubits2, qubits3)
        64. cfsim(self, theta, phi, qubits1, qubits2, qubits3)
        65. ccu1(self, lamda, qubits1, qubits2, qubits3)
        66. ccu2(self, phi, lamda, qubits1, qubits2, qubits3)
        67. ccu3(self, theta, phi, lamda, qubits1, qubits2, qubits3)
        68. ccu(self, theta, phi, lamda, gamma, qubits1, qubits2, qubits3)
    '''
    backend = 'density_matrix'
         

    def __init__(self, num_qubits=1):
        super().__init__(num_qubits)
        self.state = np.zeros((2**num_qubits, 2**num_qubits), complex)
        self.state[0, 0] = 1.0
        return None


    # def _apply_(self, op, *qubits):
    #     super()._apply_(op, *qubits)
    #     global alp

    #     if 2*op.num_qubits + self.num_qubits > len(alp):
    #         raise Exception('Not enough indices for calculation.')

    #     op_idxl = alp[-2*op.num_qubits:]
    #     op_idxr = ALP[-2*op.num_qubits:]
    #     state_idxl = alp[:self.num_qubits]
    #     state_idxr = ALP[:self.num_qubits]

    #     for i in range(op.num_qubits):
    #         state_idxl[self.num_qubits-qubits[i]-1] = op_idxr[i]
    #         state_idxr[self.num_qubits-qubits[i]-1] = op_idxr[op.num_qubits+i]
    #     start = ''.join(op_idxl) + ''.join(op_idxr) + ',' + \
    #         ''.join(state_idxl) + ''.join(state_idxr)

    #     for i in range(op.num_qubits):
    #         state_idxl[self.num_qubits-qubits[i]-1] = op_idxl[i]
    #         state_idxr[self.num_qubits-qubits[i]-1] = op_idxl[op.num_qubits+i]
    #     end = ''.join(state_idxl) + ''.join(state_idxr)

    #     # using super operator representation is a better choice.
    #     if op.category != 'superop': op = to_superop(op)
    #     rep = op.represent.reshape([2]*4*op.num_qubits)
    #     self.state = self.state.reshape([2]*2*self.num_qubits)
    #     self.state = np.einsum(start+'->'+end, rep, self.state).reshape(
    #         [2**self.num_qubits]*2)
    #     return None


    def _apply_(self, op, *qubits):
        super()._apply_(op, *qubits)
        global alp

        a_idx = [*range(2*op.num_qubits, 4*op.num_qubits)]
        b_idx = [self.num_qubits-i-1 for i in qubits] + \
            [2*self.num_qubits-i-1 for i in qubits]
        # using super operator representation is a better choice.
        if op.category != 'superop': op = to_superop(op)
        rep = op.represent.reshape([2]*4*op.num_qubits)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.tensordot(rep, self.state, axes=(a_idx, b_idx))

        s = ''.join(alp[:2*self.num_qubits])
        end = s
        start = ''
        for i in range(op.num_qubits):
            start += end[self.num_qubits-qubits[i]-1]
            s = s.replace(start[i], '')
        for i in range(op.num_qubits):
            start += end[2*self.num_qubits-qubits[i]-1]
            s = s.replace(start[op.num_qubits+i], '')
        start = start + s
        self.state = np.einsum(
            start+'->'+end, self.state).reshape(2**self.num_qubits, -1)
        return None


    def measure(self, qubit, delete=False):
        super().measure(qubit, delete)
        state = self.state.reshape([2]*2*self.num_qubits)
        dic = locals()

        string0 = ':, '*(self.num_qubits-qubit-1) + '0, ' + \
            ':, '*(self.num_qubits-1) + '0, ' + ':, '*qubit
        string1 = ':, '*(self.num_qubits-qubit-1) + '1, ' + \
            ':, '*(self.num_qubits-1) + '1, ' + ':, '*qubit

        exec('reduced0 = state[' + string0 + ']', dic)
        measured = dic['reduced0'].reshape(2**(self.num_qubits-1), -1)
        probability0 = np.trace(measured)

        if np.random.random() < probability0:
            bit = 0
            if delete:
                self.state = measured
                self.num_qubits -= 1
            else:
                exec('state[' + string1 + '] = 0.', dic)
                self.state = dic['state'].reshape(2**self.num_qubits, -1)
            self.state /= probability0
        else:
            bit = 1
            if delete:
                exec('reduced1 = state[' + string1 + ']', dic)
                self.state = dic['reduced1'].reshape(2**(self.num_qubits-1),-1)
                self.num_qubits -= 1
            else:
                exec('state[' + string0 +'] = 0.', dic)
                self.state = dic['state'].reshape(2**self.num_qubits, -1)
            self.state /= (1. - probability0)
        return bit


    def reduce(self, qubits):
        '''
        Recuced density matrix after partial trace over given qubits.

        Circuit will remove the given qubits.
        
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
                
        -Influenced(2):
            1. self.state --- qubit density matrix.
                type: numpy.ndarray, 2D, complex
            2. self.num_qubit --- number of qubits.
                type: int
        '''
        if type(qubits) is int:
            q = [qubits]
        else:
            q = list(qubits)

        if max(q) >= self.num_qubits:
            raise Exception('Qubit index oversteps.')

        if len(q) > len(set(q)):
            raise Exception('Duplicate qubits in input.')

        if self.num_qubits - len(q) < 1:
            raise Exception('Must keep one qubit at least.')

        global alp
        
        s = alp[:2*self.num_qubits]
        for i in q:
            s[2*self.num_qubits-i-1] = s[self.num_qubits-i-1]
        start = ''.join(s)

        for i in q:
            s[self.num_qubits-i-1] = ''
            s[2*self.num_qubits-i-1] = ''
        end = ''.join(s)

        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.einsum(
            start+'->'+end,self.state).reshape([2**(self.num_qubits-len(q))]*2)
        self.num_qubits -= len(q)
        return None


# 
# Kraus channel methods.
# 

    def bit_flip(self, p, qubits):
        '''
        Bit flip channel.

        -In(2):
            1. p --- the probability for not flipping.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
                
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Bit_flip([p]), qubits)
        return None


    def phase_flip(self, p, qubits):
        '''
        Phase flip channel.
        
        -In(2):
            1. p --- the probability for not flipping.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Phase_flip([p]), qubits)
        return None


    def bit_phase_flip(self, p, qubits):
        '''
        Bit phase flip channel.
        
        -In(2):
            1. p --- the probability for not flipping.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Bit_phase_flip([p]), qubits)
        return None


    def depolarize(self, p, qubits):
        '''
        Depolarizing channel.
    
        -In(2):
            1. p --- the probability for depolarization.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Depolarize([p]), qubits)
        return None


    def amplitude_damping(self, gamma, qubits):
        '''
        Amplitude damping channel.
        
        -In(2):
            1. gamma --- probability such as losing a photon.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Amplitude_damping([gamma]), qubits)
        return None


    def generalized_amplitude_damping(self, p, gamma, qubits):
        '''
        Generalized amplitude damping channel.
        
        -In(3):
            1. p --- the probability for acting normal amplitude damping.
                type: float
            2. gamma --- probability such as losing a photon.
                type: float
            3. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Generalized_amplitude_damping([p, gamma]), qubits)
        return None


    def phase_damping(self, lamda, qubits):
        '''
        Phase damping channel.
        
        -In(2):
            1. lamda --- probability such as a photon from the system has been 
                scattered(without loss of energy).
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Phase_damping([lamda]), qubits)
        return None