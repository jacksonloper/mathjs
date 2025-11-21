import assert from 'assert'
import math from '../../../../../src/defaultInstance.js'
const { rsf2csf, schur, matrix, multiply, conj, transpose, abs, complex } = math

// Helper to check approximate equality
function approxEqual (a, b, tolerance = 1e-10) {
  assert.ok(Math.abs(a - b) < tolerance, `${a} should be approximately ${b}`)
}

describe('rsf2csf', function () {
  it('should convert real Schur form to complex Schur form for 2x2 rotation matrix', function () {
    const theta = Math.PI / 6
    const A = [
      [Math.cos(theta), -Math.sin(theta)],
      [Math.sin(theta), Math.cos(theta)]
    ]

    const { T, U } = schur(A)
    const { T: Tc, U: Uc } = rsf2csf(T, U)

    // Complex Schur form should be upper triangular
    // Check that T[1][0] is zero
    assert.ok(abs(Tc[1][0]) < 1e-10, 'T[1][0] should be zero')

    // Verify A = U * T * U^H
    const UT = multiply(Uc, Tc)
    const UH = transpose(conj(Uc))
    const reconstructed = multiply(UT, UH)

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        approxEqual(reconstructed[i][j].re, A[i][j], 1e-10)
        approxEqual(reconstructed[i][j].im, 0, 1e-10)
      }
    }
  })

  it('should convert real Schur form to complex Schur form for 3x3 matrix', function () {
    const A = [[0, 2, 2], [0, 1, 2], [1, 0, 1]]

    const { T, U } = schur(A)
    const { T: Tc, U: Uc } = rsf2csf(T, U)

    // Complex Schur form should be upper triangular
    // Check that subdiagonal elements are zero
    for (let i = 1; i < 3; i++) {
      for (let j = 0; j < i; j++) {
        assert.ok(abs(Tc[i][j]) < 1e-10, `T[${i}][${j}] should be zero`)
      }
    }

    // Verify A = U * T * U^H
    const UT = multiply(Uc, Tc)
    const UH = transpose(conj(Uc))
    const reconstructed = multiply(UT, UH)

    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        approxEqual(reconstructed[i][j].re, A[i][j], 1e-9)
        approxEqual(reconstructed[i][j].im, 0, 1e-9)
      }
    }
  })

  it('should handle already triangular matrices', function () {
    const A = [[2, 1], [0, 3]]

    const { T, U } = schur(A)
    const { T: Tc } = rsf2csf(T, U)

    // Should remain triangular
    assert.ok(abs(Tc[1][0]) < 1e-10, 'T[1][0] should be zero')

    // Eigenvalues should be on the diagonal
    const diag = [Tc[0][0], Tc[1][1]]
    assert.ok(
      (abs(diag[0] - complex(2, 0)) < 1e-10 && abs(diag[1] - complex(3, 0)) < 1e-10) ||
      (abs(diag[0] - complex(3, 0)) < 1e-10 && abs(diag[1] - complex(2, 0)) < 1e-10),
      'Diagonal elements should be eigenvalues'
    )
  })

  it('should work with Matrix input', function () {
    const theta = Math.PI / 4
    const A = matrix([
      [Math.cos(theta), -Math.sin(theta)],
      [Math.sin(theta), Math.cos(theta)]
    ])

    const { T, U } = schur(A)
    const { T: Tc } = rsf2csf(T, U)

    // Result should be Matrix
    assert.ok(math.isMatrix(Tc), 'Tc should be a Matrix')

    // Complex Schur form should be upper triangular
    const TcArray = Tc.toArray()
    assert.ok(abs(TcArray[1][0]) < 1e-10, 'T[1][0] should be zero')
  })

  it('should handle 4x4 matrix with mixed eigenvalues', function () {
    // Matrix with both real and complex eigenvalues
    const A = [
      [1, 2, 0, 0],
      [-2, 1, 1, 0],
      [0, 0, 3, 0],
      [0, 0, 0, 4]
    ]

    const { T, U } = schur(A)
    const { T: Tc, U: Uc } = rsf2csf(T, U)

    // Complex Schur form should be upper triangular
    for (let i = 1; i < 4; i++) {
      for (let j = 0; j < i; j++) {
        assert.ok(abs(Tc[i][j]) < 1e-10, `T[${i}][${j}] should be zero`)
      }
    }

    // Verify A = U * T * U^H
    const UT = multiply(Uc, Tc)
    const UH = transpose(conj(Uc))
    const reconstructed = multiply(UT, UH)

    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        approxEqual(reconstructed[i][j].re, A[i][j], 1e-9)
        approxEqual(reconstructed[i][j].im, 0, 1e-9)
      }
    }
  })

  it('should throw error for non-square matrices', function () {
    assert.throws(function () {
      rsf2csf([[1, 2, 3], [4, 5, 6]], [[1, 0], [0, 1]])
    }, /square/)
  })

  it('should throw error for mismatched matrix sizes', function () {
    assert.throws(function () {
      rsf2csf([[1, 2], [3, 4]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    }, /same size/)
  })

  it('should preserve eigenvalues on the diagonal', function () {
    const A = [[0, -1], [1, 0]] // 90-degree rotation

    const { T, U } = schur(A)
    const { T: Tc } = rsf2csf(T, U)

    // Eigenvalues should be i and -i
    const eigs = [Tc[0][0], Tc[1][1]].sort((a, b) => a.im - b.im)
    approxEqual(eigs[0].re, 0, 1e-10)
    approxEqual(eigs[0].im, -1, 1e-10)
    approxEqual(eigs[1].re, 0, 1e-10)
    approxEqual(eigs[1].im, 1, 1e-10)
  })
})
