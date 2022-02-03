# 
# This code is part of Qton.
# 
# Qton Version: 2.1.0
# 
# This file: qton\operators\channels.py
# Author(s): Yunheng Ma
# Timestamp: 2022-02-03 10:18:24
# 

import numpy as np


__all__ = ["Bit_flip",
           "Phase_flip",
           "Bit_phase_flip",
           "Depolarize",
           "Amplitude_damping",
           "Generalized_amplitude_damping",
           "Phase_damping",
           ]


def build_channel(matrix_list):
    '''
    Build a quantum channel object from a numpy matrix list.
    
    -In(1):
        1. matrix_list --- matrix list.
            type: list, numpy.ndarray, 2D, complex
    
    -Return(1):
        1. ans --- compatible channel object. 
            type: qton channel class
    '''
    class Blank_channel(_Basic_channel_):
        def __init__(self):
            pass
            return None
    for i in matrix_list:
        m = i.shape[0]
        n = i.shape[1]
        num_qubits = np.log(m) / np.log(2)
        if m != n or num_qubits%1 != 0 :
            raise Exception('Matrix Shape is not allowed.')
    num_qubits = int(num_qubits)

    ans = Blank_channel()
    ans.basename = 'custom'
    ans.represent = matrix_list
    ans.num_ctrls = 0
    ans.num_qubits = num_qubits
    ans.num_params = 0
    ans.dagger = False
    return ans


def to_channel(gate_obj):
    '''
    Change a gate object to a channel object.
    
    -In(1):
        1. gate_obj --- gate object.
            type: qton gate class

    -Return(1):
        1. ans --- compatible channel object.
            type: qton channel class
    '''
    if gate_obj.category != 'gate':
        raise Exception('Input class should be gate type.')
        
    class Blank_channel(_Basic_channel_):
        def __init__(self): 
            pass 
            return None
    ans = Blank_channel()
    ans.basename   = gate_obj.basename
    ans.represent  =[gate_obj.represent]
    ans.num_ctrls  = gate_obj.num_ctrls
    ans.num_qubits = gate_obj.num_qubits
    ans.num_params = gate_obj.num_params
    ans.dagger = gate_obj.dagger
    return ans


from ._basic_op_ import _Basic_op_


class _Basic_channel_(_Basic_op_):
    '''
    Basic of quantum channels.
    
    -Attributes(7):
        1. category --- is this a "gate", "channel", "superop" or else.
            type: str
        2. basename --- indicate this object is build from whom, e.g., 
            "Bit flip".
            type: str
        3. represent --- object representation.
            type: list, numpy.ndarray, 2D, complex
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
    category = 'channel'
    represent = [np.array(None)]
    num_qubits = 1


    def __init__(self, params=[], num_ctrls=0, dagger=False):
        '''
        Initialization.

        -In(3):
            1. params --- parameters for parameterised channels.
                type: list, float
            2. num_ctrls --- number of controls.
                type: int
            3. dagger --- if dagger operation applied or not.
                type: bool

        -Influenced(4):
            1. self.represent --- channel representation.
                type: list, numpy.ndarray, 2D, complex
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
            for i in range(len(self.represent)):
                self.represent[i] = self.represent[i].transpose().conj()
        return None


    def _add_ctrl_(self, num_ctrls=1):
        '''
        Add control(s) to the channel representation.

        -In(1):
            1. num_ctrls --- number of controls to add.
                type: int

        -Influenced(3):
            1. self.represent --- channel representation.
                type: list, numpy.ndarray, 2D, complex
            2. self.num_ctrls --- number of controls.
                type: int
            3. self.num_qubits --- number of qubits.
                type: int
        '''
        pad_width = 2**(num_ctrls + self.num_qubits) - 2**self.num_qubits
        for i in range(len(self.represent)):
            self.represent[i] = np.pad(self.represent[i], (pad_width, 0), 
                mode='constant', constant_values=0.)
            for j in range(pad_width):
                self.represent[i][j, j] = 1.
        self.num_ctrls  += num_ctrls
        self.num_qubits += num_ctrls
        return None


    def _represent_(self, params):
        '''
        Set the representation of the channel.

        -In(1):
            1. params --- parameters for parameterised channels.
                type: list, float

        -Influenced(1):
            1. self.represent --- channel representation.
                type: numpy.ndarray, 2D, complex
        '''
        pass
        return None


I = np.array([[1,   0], [0,  1.]])
X = np.array([[0,   1], [1,  0.]])
Y = np.array([[0, -1j], [1j, 0.]])
Z = np.array([[1,   0], [0, -1.]])


class Bit_flip(_Basic_channel_):
    basename = 'Bit flip'
    num_params = 1

    def _represent_(self, params):
        def _bit_flip_(p):
            global I, X
            E0 = np.sqrt(p)*I
            E1 = np.sqrt(1-p)*X
            return [E0, E1]
        self.represent = _bit_flip_(params[0])
        return None


class Phase_flip(_Basic_channel_):
    basename = 'Phase flip'
    num_params = 1
    
    def _represent_(self, params):
        def _phase_flip_(p):
            global I, Z
            E0 = np.sqrt(p)*I
            E1 = np.sqrt(1-p)*Z
            return [E0, E1]
        self.represent = _phase_flip_(params[0])
        return None


class Bit_phase_flip(_Basic_channel_):
    basename = 'Bit phase flip'
    num_params = 1

    def _represent_(self, params):
        def _bit_phase_flip_(p):
            global I, Y
            E0 = np.sqrt(p)*I
            E1 = np.sqrt(1-p)*Y
            return [E0, E1]
        self.represent = _bit_phase_flip_(params[0])
        return None


class Depolarize(_Basic_channel_):
    basename = 'Depolarize'
    num_params = 1

    def _represent_(self, params):
        def _depolarize_(p):
            global I, X, Y, Z
            E0 = np.sqrt(1-p)*I
            E1 = np.sqrt(p/3)*X
            E2 = np.sqrt(p/3)*Y
            E3 = np.sqrt(p/3)*Z
            return [E0, E1, E2, E3]
        self.represent = _depolarize_(params[0])
        return None


# class Depolarize(_Basic_channel_):
#     basename = 'Depolarize'
#     num_params = 1

#     def _represent_(self, params):
#         def _depolarize_(p):
#             global I, X, Y, Z
#             E0 = np.sqrt(1-0.75*p)*I
#             E1 = 0.5*np.sqrt(p)*X
#             E2 = 0.5*np.sqrt(p)*Y
#             E3 = 0.5*np.sqrt(p)*Z
#             return [E0, E1, E2, E3]
#         self.represent = _depolarize_(params[0])
#         return None


class Amplitude_damping(_Basic_channel_):
    basename = 'Amplitude damping'
    num_params = 1

    def _represent_(self, params):
        def _amplitude_damping_(gamma):
            E0 = np.array([[1, 0], [0, np.sqrt(1-gamma)]])
            E1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
            return [E0, E1]
        self.represent = _amplitude_damping_(params[0])  
        return None


class Generalized_amplitude_damping(_Basic_channel_):
    basename = 'Generalized amplitude damping'
    num_params = 2

    def _represent_(self, params):
        def _generalized_amplitude_damping_(p, gamma):
            E0 = np.sqrt(p)*np.array([[1, 0], [0, np.sqrt(1-gamma)]])
            E1 = np.sqrt(p)*np.array([[0, np.sqrt(gamma)], [0, 0]])
            E2 = np.sqrt(1-p)*np.array([[np.sqrt(1-gamma), 0], [0, 1]])
            E3 = np.sqrt(1-p)*np.array([[0, 0], [np.sqrt(gamma), 0]])
            return [E0, E1, E2, E3]
        self.represent = _generalized_amplitude_damping_(
            *params[0:self.num_params])
        return None


class Phase_damping(_Basic_channel_):
    basename = 'Phase damping'
    num_params = 1
    
    def _represent_(self, params):
        def _phase_damping_(lamda):
            E0 = np.array([[1, 0], [0, np.sqrt(1-lamda)]])
            E1 = np.array([[0, 0], [0, np.sqrt(lamda)]])
            return [E0, E1]
        self.represent = _phase_damping_(params[0])
        return None