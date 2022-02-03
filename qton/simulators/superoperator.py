# 
# This code is part of Qton.
# 
# Qton Version: 2.1.0
# 
# This file: qton\simulators\superoperator.py
# Author(s): Yunheng Ma
# Timestamp: 2022-02-03 10:18:24
# 

import numpy as np


__all__ = ["Qsuperoperator"]


from .density_matrix import Qdensity_matrix


class Qsuperoperator(Qdensity_matrix):
    '''
    Quantum circuit operated by super operators.

    This is a wrapped of "Qdensity_matrix".

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
    backend = 'superoperator'