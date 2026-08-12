import numpy as np
import pytest

from qrc import operators as op


def test_pauli_algebra():
    assert np.allclose(op.SX @ op.SX, op.I2)
    assert np.allclose(op.SX @ op.SY, 1j * op.SZ)
    assert np.allclose(op.SMINUS @ np.array([0, 1]), np.array([1, 0]))  # |1>->|0>
    assert np.allclose(op.SPLUS @ op.SMINUS, np.diag([0, 1]))  # number op |1><1|


def test_embed_ordering():
    # sigma^z on site 0 of 2 qubits = Z (x) I
    z0 = op.pauli_op("z", 0, 2)
    assert np.allclose(z0, np.kron(op.SZ, op.I2))
    z1 = op.pauli_op("z", 1, 2)
    assert np.allclose(z1, np.kron(op.I2, op.SZ))


def test_input_state():
    for s in [0.0, 0.3, 1.0]:
        rho = op.input_state(s)
        assert np.isclose(np.trace(rho).real, 1.0)
        assert np.isclose(rho[1, 1].real, s)  # <1|psi|1> = s
        assert np.allclose(rho, rho.conj().T)


def test_input_state_out_of_range():
    with pytest.raises(ValueError):
        op.input_state(1.5)


def test_partial_trace_product():
    rA = op.input_state(0.3)
    rB = op.input_state(0.8)
    rho = np.kron(rA, rB)
    assert np.allclose(op.partial_trace(rho, keep=[0], n_qubits=2), rA)
    assert np.allclose(op.partial_trace(rho, keep=[1], n_qubits=2), rB)
    assert np.allclose(op.trace_out(rho, discard=[0], n_qubits=2), rB)


def test_partial_trace_bell_maximally_mixed():
    v = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    bell = np.outer(v, v.conj())
    red = op.partial_trace(bell, keep=[0], n_qubits=2)
    assert np.allclose(red, 0.5 * np.eye(2))


def test_partial_trace_three_qubits():
    rs = [op.input_state(x) for x in (0.2, 0.5, 0.9)]
    rho = op.kron_list(rs)
    assert np.allclose(op.partial_trace(rho, keep=[1], n_qubits=3), rs[1])
    keep02 = op.partial_trace(rho, keep=[0, 2], n_qubits=3)
    assert np.allclose(keep02, np.kron(rs[0], rs[2]))
