// test schur decomposition
import assert from 'assert'

import math from '../../../../../src/defaultInstance.js'

describe('schur', function () {
  it('should calculate schur decomposition of order 5 Array with numbers', function () {
    const A = [
      [-5.3, -1.4, -0.2, 0.7, 1.0],
      [-0.4, -1.0, -0.1, -1.2, 0.7],
      [0.3, 0.7, -2.5, 0.7, -0.3],
      [3.6, -0.1, 1.4, -2.4, 0.3],
      [2.8, 0.7, 1.4, 0.5, -4.8]
    ]
    const result = math.schur(A)
    const T = result.T
    const U = result.U

    // Verify T is quasi-upper triangular
    const n = T.length
    for (let i = 2; i < n; i++) {
      for (let j = 0; j < i - 1; j++) {
        assert.ok(Math.abs(T[i][j]) < 1e-10, `T[${i},${j}] = ${T[i][j]} should be near zero`)
      }
    }

    // Verify A = U * T * U'
    const UT = math.transpose(U)
    const reconstructed = math.multiply(math.multiply(U, T), UT)
    const diff = math.norm(math.subtract(reconstructed, A))
    assert.ok(diff < 1e-10, `Reconstruction error ${diff} should be small`)
  })

  it('should calculate schur decomposition of order 5 Matrix with numbers', function () {
    const A = math.matrix([
      [-5.3, -1.4, -0.2, 0.7, 1.0],
      [-0.4, -1.0, -0.1, -1.2, 0.7],
      [0.3, 0.7, -2.5, 0.7, -0.3],
      [3.6, -0.1, 1.4, -2.4, 0.3],
      [2.8, 0.7, 1.4, 0.5, -4.8]
    ])
    const result = math.schur(A)
    const T = result.T
    const U = result.U

    // Verify T is quasi-upper triangular
    const n = T.size()[0]
    for (let i = 2; i < n; i++) {
      for (let j = 0; j < i - 1; j++) {
        assert.ok(Math.abs(T.get([i, j])) < 1e-10, `T[${i},${j}] should be near zero`)
      }
    }

    // Verify A = U * T * U'
    const UT = math.transpose(U)
    const reconstructed = math.multiply(math.multiply(U, T), UT)
    const diff = math.norm(math.subtract(reconstructed, A))
    assert.ok(diff < 1e-10, `Reconstruction error ${diff} should be small`)
  })

  it('should produce quasi-upper triangular matrix for rotation matrices', function () {
    // Rotation matrix from random QR decomposition
    const R = math.matrix([
      [-0.03591206220229135, -0.09100469507870354, 0.9952027277203429],
      [-0.3802068171617618, -0.9197139315803332, -0.09782157349362577],
      [0.9242040358990549, -0.38189583596928406, -0.0015717815434243287]
    ])

    const schurResult = math.schur(R)
    const T = schurResult.T
    const U = schurResult.U
    const n = T.size()[0]
    const tolerance = 1e-10

    // Check that T is quasi-upper triangular:
    // - Elements below the first subdiagonal should be zero
    // - First subdiagonal may have non-zero elements (for 2x2 blocks)
    for (let i = 2; i < n; i++) {
      for (let j = 0; j < i - 1; j++) {
        const val = Math.abs(T.get([i, j]))
        assert.ok(val < tolerance,
          `T[${i},${j}] = ${T.get([i, j])} has absolute value ${val} which exceeds tolerance ${tolerance}`)
      }
    }

    // Verify that A = U * T * U^T
    const UT = math.transpose(U)
    const reconstructed = math.multiply(math.multiply(U, T), UT)
    const diff = math.norm(math.subtract(reconstructed, R))
    assert.ok(diff < tolerance,
      `Reconstruction error ${diff} exceeds tolerance ${tolerance}`)
  })
})
