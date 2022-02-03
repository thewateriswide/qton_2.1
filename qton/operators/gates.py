# 
# This code is part of Qton.
# 
# Qton Version: 2.1.0
# 
# This file: qton\operators\gates.py
# Author(s): Yunheng Ma
# Timestamp: 2022-02-03 10:18:24
# 

import numpy as np


__all__ = ["H_gate",
           "I_gate",
           "X_gate",
           "Y_gate",
           "Z_gate",
           "S_gate",
           "T_gate",
           "Swap_gate",
           "P_gate",
           "U_gate",
           "Rx_gate",
           "Ry_gate",
           "Rz_gate",
           "U1_gate",
           "U2_gate",
           "U3_gate",
           "Fsim_gate",
           ]


def build_gate(matrix):
    '''
    Build a quantum gate object from a numpy matrix.
    
    -In(1):
        1. matrix --- gate matrix.
            type: numpy.ndarray, 2D, complex
    
    -Return(1):
        1. ans --- compatible gate object. 
            type: qton gate class
    '''
    class Blank_gate(_Basic_gate_):
        def __init__(self):
            pass
            return None
    m = matrix.shape[0]
    n = matrix.shape[1]
    num_qubits = np.log(m) / np.log(2)
    if m != n or num_qubits%1 != 0 :
        raise Exception('Matrix shape is not allowed.')
    num_qubits = int(num_qubits)

    ans = Blank_gate()
    ans.basename = 'custom'
    ans.represent = matrix
    ans.num_ctrls = 0
    ans.num_qubits = num_qubits
    ans.num_params = 0
    ans.dagger = False
    return ans


from ._basic_op_ import _Basic_op_


class _Basic_gate_(_Basic_op_):
    '''
    Basic of quatnum gates.

    -Attributes(7):
        1. category --- is this a "gate", "channel", "superop" or else.
            type: str
        2. basename --- indicate this object is build from whom, e.g., 
            "Hadamard".
            type: str
        3. represent --- object representation.
            type: numpy.ndarray, 2D, complex
        4. num_ctrls --- number of controls.
            type: int
        5. num_qubits --- number of qubits.
            type: int
        6. num_params --- number of parameters.
            type: int
        7. dagger --- if dagger operation applied or not.
            type: bool
    
    -Methods(3):
        1. __init__(self, params=[], num_ctrls=0, dagger=False)
        2. _add_ctrl_(self, num_ctrls=1)
        3. _represent_(self, params)
    '''
    category = 'gate'
    represent = np.array(None)
    num_qubits = 1


    def __init__(self, params=[], num_ctrls=0, dagger=False):
        '''
        Initialization.

        -In(3):
            1. params --- parameters for parameterised gates.
                type: list, float
            2. num_ctrls --- number of controls.
                type: int
            3. dagger --- if dagger operation applied or not.
                type: bool

        -Influenced(4):
            1. self.represent --- gate representation.
                type: numpy.ndarray, 2D, complex
            2. self.num_ctrls --- number of controls.
                type: int
            3. self.num_qubits --- number of qubits.
                type: int
            4. self.dagger --- if dagger operation applied or not.
                type: bool
        '''
        self._represent_(params)

        if num_ctrls != 0:
            self._add_ctrl_(num_ctrls=num_ctrls)

        if dagger == True:
            self.dagger = True
            self.represent = self.represent.transpose().conj()
        return None


    def _add_ctrl_(self, num_ctrls=1):
        '''
        Add control(s) to the gate representation.

        -In(1):
            1. num_ctrls --- number of controls to add.
                type: int

        -Influenced(3):
            1. self.represent --- gate representation.
                type: numpy.ndarray, 2D, complex
            2. self.num_ctrls --- number of controls.
                type: int
            3. self.num_qubits --- number of qubits.
                type: int
        '''
        pad_width = 2**(num_ctrls + self.num_qubits) - 2**self.num_qubits
        self.represent = np.pad(self.represent, (pad_width, 0), 
            mode='constant', constant_values=0.)
        for i in range(pad_width):
            self.represent[i, i] = 1.
        self.num_ctrls  += num_ctrls
        self.num_qubits += num_ctrls
        return None


    def _represent_(self, params):
        '''
        Set the representation of the gate.

        -In(1):
            1. params --- parameters for parameterised gates.
                type: list, float

        -Influenced(1):
            1. self.represent --- gate representation.
                type: numpy.ndarray, 2D, complex
        '''
        pass
        return None


# 
# 1-qubit gates, without parameters.
# 

class I_gate(_Basic_gate_):
    basename = 'Identity'
    represent = np.array([[1, 0], [0, 1.]])


class X_gate(_Basic_gate_):
    basename = 'Pauli-X'
    represent = np.array([[0, 1], [1, 0.]])


class Y_gate(_Basic_gate_):
    basename = 'Pauli-Y'
    represent = np.array([[0, -1j], [1j, 0.]])


class Z_gate(_Basic_gate_):
    basename = 'Pauli-Z'
    represent = np.array([[1, 0], [0, -1.]])


class H_gate(_Basic_gate_):
    basename = 'Hadamard'
    represent = np.sqrt(0.5) * np.array([[1, 1], [1, -1]])


class S_gate(_Basic_gate_):
    basename = 'S-gate or Z-root'
    represent = np.array([[1, 0], [0., 1j]])


class T_gate(_Basic_gate_):
    basename = 'T-gate or S-root'
    represent = np.array([[1, 0], [0, (1 + 1j) * np.sqrt(0.5)]])


# 
# 1-qubit gates, with parameters.
# 

class Rx_gate(_Basic_gate_):
    basename = 'Rotation-X'
    num_params = 1

    def _represent_(self, params):
        def _rx_(theta):
            t = theta * 0.5
            return np.array([
                [np.cos(t), -1j * np.sin(t)],
                [-1j * np.sin(t), np.cos(t)],
            ], complex)
        self.represent = _rx_(params[0])
        return None


class Ry_gate(_Basic_gate_):
    basename = 'Rotation-Y'
    num_params = 1

    def _represent_(self, params):
        def _ry_(theta):
            t = theta * 0.5
            return np.array([
                [np.cos(t), -np.sin(t)],
                [np.sin(t),  np.cos(t)],
            ], complex)
        self.represent = _ry_(params[0])
        return None


class Rz_gate(_Basic_gate_):
    basename = 'Rotation-Z'
    num_params = 1

    def _represent_(self, params):
        def _rz_(theta):
            t = theta * 0.5
            return np.array([
                [np.exp(-1j * t), 0],
                [0, np.exp( 1j * t)],
            ])
        self.represent = _rz_(params[0])
        return None


class P_gate(_Basic_gate_):
    basename = 'Phase'
    num_params = 1

    def _represent_(self, params):
        def _p_(phi):
            return np.array([[1, 0], [0, np.exp(1j * phi)]])
        self.represent = _p_(params[0])
        return None


class U1_gate(_Basic_gate_):
    basename = 'U1-gate'
    num_params = 1

    def _represent_(self, params):
        def _u1_(lamda):
            return np.array([[1, 0], [0, np.exp(1j * lamda)]])
        self.represent = _u1_(params[0])
        return None


class U2_gate(_Basic_gate_):
    basename = 'U2-gate'
    num_params = 2

    def _represent_(self, params):
        def _u2_(phi, lamda):
            return np.array([
                [1, -np.exp(1j * lamda)],
                [np.exp(1j * phi), np.exp(1j * lamda + 1j * phi)]
            ]) * np.sqrt(0.5)
        self.represent = _u2_(*params[0:self.num_params])
        return None


class U3_gate(_Basic_gate_):
    basename = 'U3-gate'
    num_params = 3

    def _represent_(self, params):
        def _u3_(theta, phi, lamda):
            t = theta * 0.5
            return np.array([
                [np.cos(t), -np.exp(1j * lamda) * np.sin(t)],
                [np.exp(1j * phi) * np.sin(t),
                 np.exp(1j * lamda + 1j * phi) * np.cos(t)]])
        self.represent = _u3_(*params[0:self.num_params])
        return None


class U_gate(_Basic_gate_):
    basename = 'Universal'
    num_params = 4

    def _represent_(self, params):
        def _u_(theta, phi, lamda, gamma):
            t = theta * 0.5
            return np.array([
                [np.cos(t), -np.exp(1j * lamda) * np.sin(t)],
                [np.exp(1j * phi) * np.sin(t),
                 np.exp(1j * lamda + 1j * phi) * np.cos(t)]
            ]) * np.exp(1j * gamma)
        self.represent = _u_(*params[0:self.num_params])
        return None


# 
# 2-qubit gates, without parameters.
# 

class Swap_gate(_Basic_gate_):
    basename = 'Swap'
    num_qubits = 2
    represent = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1.],
    ])


# 
# 2-qubit gates, with parameters.
# 

class Fsim_gate(_Basic_gate_):
    basename = 'fSim'
    num_qubits = 2
    num_params = 2

    def _represent_(self, params):
        def _fsim_(theta, phi):
            return np.array([
                [1, 0, 0, 0],
                [0, np.cos(theta), -1j*np.sin(theta), 0],
                [0, -1j*np.sin(theta), np.cos(theta), 0],
                [0, 0, 0, np.exp(-1j*phi)]
            ])
        self.represent = _fsim_(*params[0:self.num_params])
        return None