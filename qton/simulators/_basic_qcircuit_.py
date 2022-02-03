# 
# This code is part of Qton.
# 
# Qton Version: 2.1.0
# 
# This file: qton\simulators\_basic_qcircuit_.py
# Author(s): Yunheng Ma
# Timestamp: 2022-02-03 10:18:24
# 

import numpy as np


__all__ = ["_Basic_qcircuit_"]


from qton.operators.gates import *


class _Basic_qcircuit_:
    '''
    Basic of quantum circuits.
    
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
    backend = ''
    num_qubits = 0
    state = None


    def __init__(self, num_qubits=1):
        '''
        Initialization.
        
        -In(1):
            1. num_qubits --- number of qubits.
                type: int
                
        -Influenced(2):
            1. self.num_qubits --- number of qubits.
                type: int
            2. self.state --- circuit state representation.
                type: numpy.ndarray, complex
        '''
        self.num_qubits = num_qubits
        return None


    def _apply_(self, op, *qubits):
        '''
        Apply an operation on given qubits.
        
        -In(2):
            1. op --- qubit operation.
                type: qton operator
            2. qubits --- qubit indices.
                type: int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex   
        '''
        if op.num_qubits > len(set(qubits)):
            raise Exception('Duplicate qubits in input.')
        return None


    def apply(self, op, *qubits):
        '''
        Apply an operation on given qubits.
        
        -In(2):
            1. op --- qubit operation.
                type: qton operator
            2. qubits --- qubit indices.
                type: int, list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex   
        '''
        if op.num_qubits > self.num_qubits:
            raise Exception('Improper operator.')
            
        if op.num_qubits > len(qubits):
            raise Exception('Missing qubit argument(s).')
            
        if op.num_qubits < len(qubits):
            raise Exception('Too many qubit indices.')
        
        num_list = 0
        pos_list = 0
        qs = []
        for i in range(len(qubits)):
            if type(qubits[i]) is not int:
                qs = (list(qubits[i]))
                num_list += 1
                pos_list = i
                if num_list > 1:
                    raise Exception('Too many controls or tagerts.')
        
        if qs == []:
            self._apply_(op, *qubits)
        else:
            for q in qs:
                self._apply_(op, *qubits[:pos_list], q, *qubits[pos_list+1:])
        return None


    def measure(self, qubit, delete=False):
        '''
        Projective measurement on a given qubit.

        For clarity, only allow one qubit to be measured.
        
        -In(2):
            1. qubit --- index of measured qubit.
                type: int
            2. delete --- delete this qubit after measurement?
                type: bool

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex
                
        -Return(1):
            1. bit --- measured output, 0 or 1.
                type: int
        '''
        if delete and self.num_qubits < 2:
            raise Exception('Cannot delete the last qubit.')
        return None


    def add_qubit(self, num_qubits=1):
        '''
        Add qubit(s) at the tail of the circuit.
        
        -In(1):
            1. num_qubits --- number of qubits to be added in.
                type: int

        -Influenced(2):
            1. self.num_qubits --- number of qubits.
                type: int
            2. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        if self.backend == 'statevector':
            new = np.zeros(2**num_qubits, complex)
            new[0] = 1.
        elif self.backend == 'density_matrix':
            new = np.zeros((2**num_qubits, 2**num_qubits), complex)
            new[0, 0] = 1.
        elif self.backend == 'unitary':
            new = np.eye((2**num_qubits), dtype=complex)
        elif self.backend == 'superoperator':
            new = np.zeros((2**num_qubits, 2**num_qubits), complex)
            new[0, 0] = 1.
        else:
            raise Exception('Unrecognized circuit backend.')
            
        self.state = np.kron(new, self.state)
        self.num_qubits += num_qubits
        return None


    def sample(self, shots=1024, output='memory'):
        '''
        Sample a statevector, sampling with replacement.
        
        -In(2):
            1. shots --- sampling times.
                type: int
            2. output --- output date type
                type: str: "memory", "statistic", "counts"

        -Return(1):
            1.(3):
                1. memory --- every output.
                    type: list, int
                2. statistic --- repeated times of every basis.
                    type: numpy.ndarray, int
                3. counts --- counts for basis.
                    type: dict            
        '''
        if self.backend == 'statevector':
            distribution = self.state * self.state.conj()
        elif self.backend == 'density_matrix':
            distribution = self.state.diagonal()
        elif self.backend == 'unitary':
            distribution = self.state[:, 0] * self.state[:, 0].conj()
        elif self.backend == 'superoperator':
            distribution = self.state.diagonal()
        else:
            raise Exception('Unrecognized circuit backend.')

        from random import choices
        N = 2**self.num_qubits
        memory = choices(range(N), weights=distribution, k=shots)

        if output == 'memory':
            return memory

        elif output == 'statistic':
            statistic = np.zeros(N, int)
            for i in memory:
                statistic[i] += 1
            return statistic

        elif output == 'counts':
            counts = {}
            for i in memory:
                key = format(i, '0%db' % self.num_qubits)
                if key in counts:
                    counts[key] += 1
                else:
                    counts[key] = 1
            return counts

        else:
            raise Exception('Unrecognized output type.')

    
    def copy(self):
        '''
        Return an fully independent copy of current circuit instance.

        -Return(1):
            1. --- a copy of current instance.
                type: qton circuit class
        '''
        from copy import deepcopy
        
        return deepcopy(self)


# 
# Gate methods.
# 

    def i(self, qubits):
        '''
        Identity gate.
        
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(I_gate(), qubits)
        return None


    def x(self, qubits):
        '''
        Pauli-X gate.
        
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(X_gate(), qubits)
        return None


    def y(self, qubits):
        '''
        Pauli-Y gate.
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Y_gate(), qubits)
        return None


    def z(self, qubits):
        '''
        Pauli-Z gate.

        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Z_gate(), qubits)
        return None


    def h(self, qubits):
        '''
        Hadamard gate.

        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(H_gate(), qubits)
        return None


    def s(self, qubits):
        '''
        Phase S gate.

        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(S_gate(), qubits)
        return None


    def t(self, qubits):
        '''
        pi/8 T gate.

        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(T_gate(), qubits)
        return None


    def sdg(self, qubits):
        '''
        S dagger gate.
        
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(S_gate(dagger=True), qubits)
        return None


    def tdg(self, qubits):
        '''
        T dagger gate.
        
        -In(1):
            1. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(T_gate(dagger=True), qubits)
        return None


    def rx(self, theta, qubits):
        '''
        Rotation along X axis.
        
        -In(2):
            1. theta --- rotation angle.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Rx_gate([theta]), qubits)
        return None


    def ry(self, theta, qubits):
        '''
        Rotation along Y axis.
        
        -In(2):
            1. theta --- rotation angle.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Ry_gate([theta]), qubits)
        return None


    def rz(self, theta, qubits):
        '''
        Rotation along Z axis.
        
        -In(2):
            1. theta --- rotation angle.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Rz_gate([theta]), qubits)
        return None


    def p(self, phi, qubits):
        '''
        Phase gate.

        -In(2):
            1. phi --- phase angle.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(P_gate([phi]), qubits)
        return None


    def u1(self, lamda, qubits):
        '''
        U1 gate.
        
        -In(2):
            1. lamda --- phase angle.
                type: float
            2. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U1_gate([lamda]), qubits)
        return None


    def u2(self, phi, lamda, qubits):
        '''
        U2 gate.
        
        -In(3):
            1. phi --- phase angle.
                type: float
            2. lamda --- phase angle.
                type: float
            3. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U2_gate([phi, lamda]), qubits)
        return None


    def u3(self, theta, phi, lamda, qubits):
        '''
        U3 gate.

        -In(4):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U3_gate([theta, phi, lamda]), qubits)
        return None


    def u(self, theta, phi, lamda, gamma, qubits):
        '''
        Universal gate.
        
        -In(5):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. gamma --- global phase.
                type: float
            5. qubits --- qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U_gate([theta, phi, lamda, gamma]), qubits)
        return None


    def swap(self, qubit1, qubit2):
        '''
        Swap gate.
        
        -In(2):
            1. qubit1 --- first qubit index.
                type: int
            2. qubit2 --- second qubit index.
                type: int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self._apply_(Swap_gate(), qubit1, qubit2)
        return None


    def cx(self, qubits1, qubits2):
        '''
        Controlled Pauli-X gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(X_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def cy(self, qubits1, qubits2):
        '''
        Controlled Pauli-Y gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Y_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def cz(self, qubits1, qubits2):
        '''
        Controlled Pauli-Z gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Z_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def ch(self, qubits1, qubits2):
        '''
        Controlled Hadamard gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(H_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def cs(self, qubits1, qubits2):
        '''
        Controlled S gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(S_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def ct(self, qubits1, qubits2):
        '''
        Controlled T gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(T_gate(num_ctrls=1), qubits1, qubits2)
        return None


    def csdg(self, qubits1, qubits2):
        '''
        Controlled S dagger gate.

        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(S_gate(num_ctrls=1, dagger=True), qubits1, qubits2)
        return None


    def ctdg(self, qubits1, qubits2):
        '''
        Controlled T dagger gate.
        
        -In(2):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(T_gate(num_ctrls=1, dagger=True), qubits1, qubits2)
        return None


    def crx(self, theta, qubits1, qubits2):
        '''
        Controlled rotation along X axis.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Rx_gate([theta], num_ctrls=1), qubits1, qubits2)
        return None


    def cry(self, theta, qubits1, qubits2):
        '''
        Controlled rotation along Y axis.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Ry_gate([theta], num_ctrls=1), qubits1, qubits2)
        return None


    def crz(self, theta, qubits1, qubits2):
        '''
        Controlled rotation along Z axis.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Rz_gate([theta], num_ctrls=1), qubits1, qubits2)
        return None


    def cp(self, phi, qubits1, qubits2):
        '''
        Controlled phase gate.
        
        -In(3):
            1. phi --- phase angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(P_gate([phi], num_ctrls=1), qubits1, qubits2)
        return None


    def fsim(self, theta, phi, qubits1, qubits2):
        '''
        fSim gate.
        
        -In(3):
            1. theta --- rotation angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. qubits1 --- first qubit indices.
                type: int; list, int
            4. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Fsim_gate([theta, phi]), qubits1, qubits2)
        return None


    def cu1(self, lamda, qubits1, qubits2):
        '''
        Controlled U1 gate.

        -In(3):
            1. lamda --- phase angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U1_gate([lamda], num_ctrls=1), qubits1, qubits2)
        return None


    def cu2(self, phi, lamda, qubits1, qubits2):
        '''
        Controlled U2 gate.
        
        -In(4):
            1. phi --- phase angle.
                type: float
            2. lamda --- phase angle.
                type: float
            3. qubits1 --- first qubit indices.
                type: int; list, int
            4. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U2_gate([phi, lamda], num_ctrls=1), qubits1, qubits2)
        return None


    def cu3(self, theta, phi, lamda, qubits1, qubits2):
        '''
        Controlled U3 gate.
        
        -In(5):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. qubits1 --- first qubit indices.
                type: int; list, int
            5. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U3_gate([theta, phi, lamda], num_ctrls=1), qubits1, qubits2)
        return None


    def cu(self, theta, phi, lamda, gamma, qubits1, qubits2):
        '''
        Controlled universal gate.

        -In(6):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. gamma --- global phase.
                type: float
            5. qubits1 --- first qubit indices.
                type: int; list, int
            6. qubits2 --- second qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U_gate([theta, phi, lamda, gamma], num_ctrls=1), qubits1, qubits2)
        return None


    def cswap(self, qubit1, qubit2, qubit3):
        '''
        Controlled swap gate.
        
        -In(3):
            1. qubit1 --- first qubit index.
                type: int
            2. qubit2 --- second qubit index.
                type: int
            3. qubit3 --- third qubit index.
                type: int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self._apply_(Swap_gate(num_ctrls=1), qubit1, qubit2, qubit3)
        return None


    def ccx(self, qubits1, qubits2, qubits3):
        '''
        Double controlled Pauli-X gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(X_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccy(self, qubits1, qubits2, qubits3):
        '''
        Double controlled Pauli-Y gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Y_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccz(self, qubits1, qubits2, qubits3):
        '''
        Double controlled Pauli-Z gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Z_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def cch(self, qubits1, qubits2, qubits3):
        '''
        Double controlled Hadamard gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(H_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccs(self, qubits1, qubits2, qubits3):
        '''
        Double controlled S gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(S_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def cct(self, qubits1, qubits2, qubits3):
        '''
        Double controlled T gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(T_gate(num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccsdg(self, qubits1, qubits2, qubits3):
        '''
        Double controlled S dagger gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(S_gate(num_ctrls=2, dagger=True), qubits1, qubits2, qubits3)
        return None


    def cctdg(self, qubits1, qubits2, qubits3):
        '''
        Double controlled T dagger gate.
        
        -In(3):
            1. qubits1 --- first qubit indices.
                type: int; list, int
            2. qubits2 --- second qubit indices.
                type: int; list, int
            3. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(T_gate(num_ctrls=2, dagger=True), qubits1, qubits2, qubits3)
        return None


    def ccrx(self, theta, qubits1, qubits2, qubits3):
        '''
        Double controlled rotation along X axis.
        
        -In(4):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Rx_gate([theta], num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccry(self, theta, qubits1, qubits2, qubits3):
        '''
        Double controlled rotation along Y axis.
        
        -In(4):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Ry_gate([theta], num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccrz(self, theta, qubits1, qubits2, qubits3):
        '''
        Double controlled rotation along Z axis.
        
        -In(4):
            1. theta --- rotation angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Rz_gate([theta], num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccp(self, phi, qubits1, qubits2, qubits3):
        '''
        Double controlled phase gate.
        
        -In(4):
            1. phi --- phase angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(P_gate([phi], num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def cfsim(self, theta, phi, qubits1, qubits2, qubits3):
        '''
        Controlled fSim gate.
        
        -In(5):
            1. theta --- rotation angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. qubits1 --- first qubit indices.
                type: int; list, int
            4. qubits2 --- second qubit indices.
                type: int; list, int
            5. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(Fsim_gate([theta, phi], num_ctrls=1), qubits1, qubits2, qubits3)
        return None


    def ccu1(self, lamda, qubits1, qubits2, qubits3):
        '''
        Double controlled U1 gate.
        
        -In(4):
            1. lamda --- phase angle.
                type: float
            2. qubits1 --- first qubit indices.
                type: int; list, int
            3. qubits2 --- second qubit indices.
                type: int; list, int
            4. qubits3 --- third qubit indices.
                type: int; list, int
        
        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U1_gate([lamda], num_ctrls=2), qubits1, qubits2, qubits3)
        return None


    def ccu2(self, phi, lamda, qubits1, qubits2, qubits3):
        '''
        Double controlled U2 gate.
        
        -In(5):
            1. phi --- phase angle.
                type: float
            2. lamda --- phase angle.
                type: float
            3. qubits1 --- first qubit indices.
                type: int; list, int
            4. qubits2 --- second qubit indices.
                type: int; list, int
            5. qubits3 --- third qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U2_gate([phi, lamda], num_ctrls=2), qubits1, qubits2,qubits3)
        return None


    def ccu3(self, theta, phi, lamda, qubits1, qubits2, qubits3):
        '''
        Double controlled U3 gate.
        
        -In(6):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. qubits1 --- first qubit indices.
                type: int; list, int
            5. qubits2 --- second qubit indices.
                type: int; list, int
            6. qubits3 --- third qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U3_gate([theta, phi, lamda], num_ctrls=2), qubits1, qubits2, 
            qubits3)
        return None


    def ccu(self, theta, phi, lamda, gamma, qubits1, qubits2, qubits3):
        '''
        Double controlled universal gate.
        
        -In(7):
            1. theta --- amplitude angle.
                type: float
            2. phi --- phase angle.
                type: float
            3. lamda --- phase angle.
                type: float
            4. gamma --- global phase.
                type: float
            5. qubits1 --- first qubit indices.
                type: int; list, int
            6. qubits2 --- second qubit indices.
                type: int; list, int
            7. qubits3 --- third qubit indices.
                type: int; list, int

        -Influenced(1):
            1. self.state --- circuit state representation.
                type: numpy.ndarray, complex    
        '''
        self.apply(U_gate([theta, phi, lamda, gamma], num_ctrls=2), qubits1, 
            qubits2, qubits3)
        return None