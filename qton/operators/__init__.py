# 
# This code is part of Qton.
# 
# Qton Version: 2.1.0
# 
# This file: qton\operators\__init__.py
# Author(s): Yunheng Ma
# Timestamp: 2022-02-03 10:18:24
# 

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
           "Bit_flip",
           "Phase_flip",
           "Bit_phase_flip",
           "Depolarize",
           "Amplitude_damping",
           "Generalized_amplitude_damping",
           "Phase_damping",
        #    "I",
        #    "X",
        #    "Y",
        #    "Z",
        #    "H",
        #    "S",
        #    "T",
        #    "Swap",
        #    "CX",
        #    "CY",
        #    "CZ",
        #    "CH",
        #    "CS",
        #    "CT",
        #    "RX",
        #    "RY",
        #    "RZ",
        #    "P",
        #    "U1",
        #    "U2",
        #    "U3",
        #    "U",
        #    "FSIM",
        #    "CRX",
        #    "CRY",
        #    "CRZ",
        #    "CP",
        #    "CU1",
        #    "CU2",
        #    "CU3",
        #    "CU",
        #    "CCRX",
        #    "CCRY",
        #    "CCRZ",
        #    "CCP",
        #    "CCU1",
        #    "CCU2",
        #    "CCU3",
        #    "CCU",
        #    "bit_flip",
        #    "phase_flip",
        #    "bit_phase_flip",
        #    "depolarize",
        #    "amplitude_damping",
        #    "generalized_amplitude_damping",
        #    "phase_damping",
           ]


from .gates import *
from .channels import *

  
# 
# 1-qubit fixed quantum gates.
# 

I = I_gate().represent
X = X_gate().represent
Y = Y_gate().represent
Z = Z_gate().represent
H = H_gate().represent
S = S_gate().represent
T = T_gate().represent


# 
# 2-qubit fixed quantum gates.
# 

Swap = Swap_gate().represent
CX = X_gate(num_ctrls=1).represent
CY = Y_gate(num_ctrls=1).represent
CZ = Z_gate(num_ctrls=1).represent
CH = H_gate(num_ctrls=1).represent
CS = S_gate(num_ctrls=1).represent
CT = T_gate(num_ctrls=1).represent


# 
# 1-qubit quantum gates, with parameters.
# 

def RX(theta): 
    return Rx_gate(params=[theta]).represent
def RY(theta): 
    return Ry_gate(params=[theta]).represent
def RZ(theta): 
    return Rz_gate(params=[theta]).represent

def P (phi):               
    return P_gate (params=[phi]).represent
def U1(lamda):             
    return U1_gate(params=[lamda]).represent
def U2(phi, lamda):        
    return U2_gate(params=[phi, lamda]).represent
def U3(theta, phi, lamda): 
    return U3_gate(params=[theta, phi, lamda]).represent
def U (theta, phi, lamda): 
    return U_gate (params=[theta, phi, lamda]).represent


# 
# 2-qubit quantum gates, with parameters.
# 

def FSIM(theta, phi):
    return Fsim_gate(params=[theta, phi]).represent

def CRX(theta): 
    return Rx_gate(params=[theta], num_ctrls=1).represent
def CRY(theta): 
    return Ry_gate(params=[theta], num_ctrls=1).represent
def CRZ(theta): 
    return Rz_gate(params=[theta], num_ctrls=1).represent

def CP (phi):               
    return P_gate (params=[phi], num_ctrls=1).represent
def CU1(lamda):             
    return U1_gate(params=[lamda], num_ctrls=1).represent
def CU2(phi, lamda):        
    return U2_gate(params=[phi, lamda], num_ctrls=1).represent
def CU3(theta, phi, lamda): 
    return U3_gate(params=[theta, phi, lamda], num_ctrls=1).represent
def CU (theta, phi, lamda): 
    return U_gate (params=[theta, phi, lamda], num_ctrls=1).represent


# 
# 3-qubit quantum gates, with 2 controls and parameters.
# 

def CCRX(theta): 
    return Rx_gate(params=[theta], num_ctrls=2).represent
def CCRY(theta): 
    return Ry_gate(params=[theta], num_ctrls=2).represent
def CCRZ(theta): 
    return Rz_gate(params=[theta], num_ctrls=2).represent

def CCP (phi):               
    return P_gate (params=[phi], num_ctrls=2).represent
def CCU1(lamda):             
    return U1_gate(params=[lamda], num_ctrls=2).represent
def CCU2(phi, lamda):        
    return U2_gate(params=[phi, lamda], num_ctrls=2).represent
def CCU3(theta, phi, lamda): 
    return U3_gate(params=[theta, phi, lamda], num_ctrls=2).represent
def CCU (theta, phi, lamda): 
    return U_gate (params=[theta, phi, lamda], num_ctrls=2).represent


# 
# Standard quantum channels.
# 

def bit_flip(p): 
    return Bit_flip(params=[p]).represent
def phase_flip(p): 
    return Phase_flip(params=[p]).represent
def bit_phase_flip(p): 
    return Bit_phase_flip(params=[p]).represent
def depolarize(p): 
    return Depolarize(params=[p]).represent

def amplitude_damping(gamma): 
    return Amplitude_damping(params=[gamma]).represent
def generalized_amplitude_damping(p, gamma): 
    return Generalized_amplitude_damping(params=[p, gamma]).represent
def phase_damping(lamda): 
    return Phase_damping(params=[lamda]).represent