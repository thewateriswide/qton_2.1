# 
# This code is part of Qton.
# 
# Qton Version: 2.1.0
# 
# This file: qton\simulators\unitary.py
# Author(s): Yunheng Ma
# Timestamp: 2022-02-03 10:18:24
# 

import numpy as np


__all__ = ["Qunitary"]


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


class Qunitary(_Basic_qcircuit_):
    '''
    Quantum circuit represented by circuit unitary.

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
    backend = 'unitary'

    
    def __init__(self, num_qubits=1):
        super().__init__(num_qubits)
        self.state = np.eye((2**num_qubits), dtype=complex)
        return None


    # def _apply_(self, op, *qubits):
    #     super()._apply_(op, *qubits)
    #     global alp

    #     if 2*op.num_qubits + self.num_qubits > len(alp):
    #         raise Exception('Not enough indices for calculation.')

    #     op_idxl = alp[-op.num_qubits:]
    #     op_idxr = ALP[-op.num_qubits:]
    #     state_idxl = alp[:self.num_qubits]
    #     state_idxr = ALP[:self.num_qubits]

    #     for i in range(op.num_qubits):
    #         state_idxl[self.num_qubits-qubits[i]-1] = op_idxr[i]
    #     start = ''.join(op_idxl) + ''.join(op_idxr) + ',' + \
    #         ''.join(state_idxl) + ''.join(state_idxr)

    #     for i in range(op.num_qubits):
    #         state_idxl[self.num_qubits-qubits[i]-1] = op_idxl[i]
    #     end = ''.join(state_idxl) + ''.join(state_idxr)

    #     rep = op.represent.reshape([2]*2*op.num_qubits)
    #     self.state = self.state.reshape([2]*2*self.num_qubits)
    #     self.state = np.einsum(
    #         start+'->'+end, rep, self.state).reshape(2**self.num_qubits, -1)
    #     return None


    def _apply_(self, op, *qubits):
        super()._apply_(op, *qubits)
        global alp

        a_idx = [*range(op.num_qubits, 2*op.num_qubits)]
        b_idx = [self.num_qubits-i-1 for i in qubits]
        if op.category != 'gate': 
            raise Exception('Use "gate", not {}'.format(op.category))
        rep = op.represent.reshape([2]*2*op.num_qubits)
        self.state = self.state.reshape([2]*2*self.num_qubits)
        self.state = np.tensordot(rep, self.state, axes=(a_idx, b_idx))

        s = ''.join(alp[:2*self.num_qubits])
        end = s
        start = ''
        for i in range(op.num_qubits):
            start += end[self.num_qubits-qubits[i]-1]
            s = s.replace(start[i], '')
        start = start + s
        self.state = np.einsum(
            start+'->'+end, self.state).reshape(2**self.num_qubits, -1)
        return None


    def measure(self, qubit, delete=False):
        super().measure(qubit, delete)

        state = self.state.reshape([2]*2*self.num_qubits)
        svec = self.state[:, 0].reshape([2]*self.num_qubits)
        dic = locals()

        string0 = ':, '*(self.num_qubits-qubit-1) + '0, ' + ':, '*qubit
        string1 = ':, '*(self.num_qubits-qubit-1) + '1, ' + ':, '*qubit

        exec('reduced0 = svec[' + string0 + ']', dic)
        measured = dic['reduced0'].reshape(-1)
        probability0 = np.einsum('i,i->', measured, measured.conj())

        if np.random.random() < probability0:
            bit = 0
            if delete:
                exec('reduced0 = state[' + string0 + string0 + ']', dic)
                self.state = dic['reduced0'].reshape(2**(self.num_qubits-1),-1)
                self.num_qubits -= 1
            else:
                exec('state[' + string1 + ':, '*self.num_qubits + '] = 0.',dic)
                self.state = dic['state'].reshape(2**self.num_qubits, -1)
            self.state /= np.sqrt(probability0)
        else:
            bit = 1
            if delete:
                exec('reduced1 = state[' + string1 + string0 + ']', dic)
                self.state = dic['reduced1'].reshape(2**(self.num_qubits-1),-1)
                self.num_qubits -= 1
            else:
                exec('state[' + string0 + ':, '*self.num_qubits + '] = 0.',dic)
                self.state = dic['state'].reshape(2**self.num_qubits, -1)
            self.state /= np.sqrt(1. - probability0)
        return bit