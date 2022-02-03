# 
# This code is part of Qton.
# 
# Qton Version: 2.1.0
# 
# This file: qton\operators\superop.py
# Author(s): Yunheng Ma
# Timestamp: 2022-02-03 10:18:24
# 

import numpy as np


__all__ = []


def build_superop(matrix):
    '''
    Build a quantum super operator object from a numpy matrix.
    
    -In(1):
        1. matrix --- super operator matrix.
            type: numpy.ndarray, 2D, complex
    
    -Return(1):
        1. ans --- compatible superop object. 
            type: qton superop class
    '''
    class Blank_superop(_Basic_superop_):
        pass

    m = matrix.shape[0]
    n = matrix.shape[1]
    num_qubits = np.log(m) / np.log(4)
    if m != n or num_qubits%1 != 0 :
        raise Exception('Matrix Shape is not allowed.')
    num_qubits = int(num_qubits)

    ans = Blank_superop()
    ans.basename = 'custom'
    ans.represent = matrix
    ans.num_ctrls = 0
    ans.num_qubits = num_qubits
    ans.num_params = 0
    ans.dagger = False
    return ans


def to_superop(op_obj):
    '''
    Change a gate or channel object to a superop object.
    
    -In(1):
        1. op_obj --- gate or channel object.
            type: qton gate class; qton channel class

    -Return(1):
        1. ans --- superop object.
            type: qton superop class
    '''
    class Blank_superop(_Basic_superop_):
        pass

    ans = Blank_superop()  # do not use "_Basic_superop_" here
    if op_obj.category == 'gate':        
        ans.represent = np.kron(op_obj.represent, op_obj.represent.conj())
    elif op_obj.category == 'channel':
        N = 4**op_obj.num_qubits
        ans.represent = np.zeros((N, N), complex)
        for i in range(len(op_obj.represent)):
            ans.represent += np.kron(op_obj.represent[i], 
                op_obj.represent[i].conj())
    else:
        raise Exception('Only gates and channels are supported.')
    
    ans.basename   = op_obj.basename
    ans.num_ctrls  = op_obj.num_ctrls
    ans.num_qubits = op_obj.num_qubits
    ans.num_params = op_obj.num_params
    ans.dagger = op_obj.dagger
    return ans


from ._basic_op_ import _Basic_op_


class _Basic_superop_(_Basic_op_):
    '''
    Basic of quantum super operators.
    
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
    '''
    category = 'superop'
    represent = np.array(None)