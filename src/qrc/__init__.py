"""Quantum Reservoir Computing with engineered dissipation.

Reimplementation of the baseline models from

    A. Sannia, R. Martinez-Pena, M. C. Soriano, G. L. Giorgi, R. Zambrini,
    "Dissipation as a resource for Quantum Reservoir Computing",
    Quantum 8, 1291 (2024); arXiv:2212.12078v2,

plus a modular framework for dissipation-engineering experiments (Track A: rate
engineering, Track B: jump-operator engineering).

Subpackage layout follows the project specification:
    operators    - Pauli algebra, tensor embedding, partial trace
    liouvillian  - vectorised Liouvillian / Lindblad superoperators
    reservoirs   - Fuji-Nakajima, continuous-dissipation, generic Lindblad
    tasks        - STM, NARMA, parity, Mackey-Glass
    readout      - observables, feature extraction, linear training, metrics
    diagnostics  - spectral gap, mixing time, trace distance, effective rank
"""

__version__ = "0.1.0"
