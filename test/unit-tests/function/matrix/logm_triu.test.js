import assert from 'assert'
import { approxDeepEqual, approxEqual } from '../../../../tools/approx.js'
import math from '../../../../src/defaultInstance.js'
const { logm_triu: logmTriu, matrix, expm, re, im } = math // eslint-disable-line camelcase

describe('logm_triu', function () {
  it('should compute logarithm of 1x1 triangular matrix', function () {
    const T = [[Math.E]]
    const result = logmTriu(T)
    approxDeepEqual(result, [[1]], 1e-10)
  })

  it('should compute logarithm of 2x2 diagonal matrix with positive elements', function () {
    const T = [[Math.E, 0], [0, Math.E * Math.E]]
    const result = logmTriu(T)
    approxDeepEqual(result, [[1, 0], [0, 2]], 1e-10)
  })

  it('should compute logarithm of 2x2 upper triangular matrix with positive diagonal', function () {
    // T = [[2, 1], [0, 3]]
    // log(T) should have log(2) and log(3) on diagonal
    const T = [[2, 1], [0, 3]]
    const result = logmTriu(T)

    // Check diagonal
    approxEqual(result[0][0], Math.log(2), 1e-10)
    approxEqual(result[1][1], Math.log(3), 1e-10)

    // Check that result is triangular
    approxEqual(result[1][0], 0, 1e-10)

    // Verify by roundtrip: exp(log(T)) ≈ T
    const reconstructed = expm(result).valueOf()
    approxDeepEqual(reconstructed, T, 1e-9)
  })

  it('should compute logarithm of 3x3 upper triangular matrix with positive diagonal', function () {
    const T = [[2, 1, 0.5], [0, 3, 1], [0, 0, 4]]
    const result = logmTriu(T)

    // Check diagonal
    approxEqual(result[0][0], Math.log(2), 1e-10)
    approxEqual(result[1][1], Math.log(3), 1e-10)
    approxEqual(result[2][2], Math.log(4), 1e-10)

    // Check that result is triangular
    approxEqual(result[1][0], 0, 1e-10)
    approxEqual(result[2][0], 0, 1e-10)
    approxEqual(result[2][1], 0, 1e-10)
  })

  it('should handle matrix with equal diagonal elements', function () {
    const T = [[2, 1], [0, 2]]
    const result = logmTriu(T)

    // Should use L'Hôpital's rule for off-diagonal
    approxEqual(result[0][0], Math.log(2), 1e-10)
    approxEqual(result[1][1], Math.log(2), 1e-10)
    approxEqual(result[0][1], 0.5, 1e-10) // T[0,1] / T[0,0] = 1/2
  })

  it('should return complex result for matrix with negative diagonal element', function () {
    // T = [[-1, 1], [0, 2]]
    // log(-1) = log(1) + i*π = 0 + i*π
    const T = [[-1, 1], [0, 2]]
    const result = logmTriu(T)

    // Result should be complex
    const r00 = result[0][0]
    const r11 = result[1][1]

    // Check that result contains complex numbers
    assert.strictEqual(r00.type, 'Complex')
    approxEqual(re(r00), 0, 1e-10) // Real part of log(-1)
    approxEqual(im(r00), Math.PI, 1e-10) // Imaginary part of log(-1)

    approxEqual(re(r11), Math.log(2), 1e-10)
    approxEqual(im(r11), 0, 1e-10)
  })

  it('should return complex result for 2x2 matrix with all negative diagonal', function () {
    const T = [[-2, 1], [0, -3]]
    const result = logmTriu(T)

    // log(-2) = log(2) + i*π
    // log(-3) = log(3) + i*π
    const r00 = result[0][0]
    const r11 = result[1][1]

    approxEqual(re(r00), Math.log(2), 1e-10)
    approxEqual(im(r00), Math.PI, 1e-10)

    approxEqual(re(r11), Math.log(3), 1e-10)
    approxEqual(im(r11), Math.PI, 1e-10)
  })

  it('should work with Matrix input', function () {
    const T = matrix([[Math.E, 1], [0, Math.E * Math.E]])
    const result = logmTriu(T)

    assert.strictEqual(result.type, 'DenseMatrix')
    approxEqual(result.get([0, 0]), 1, 1e-10)
    approxEqual(result.get([1, 1]), 2, 1e-10)
  })

  it('should throw error for non-square matrix', function () {
    assert.throws(function () {
      logmTriu([[1, 2, 3], [0, 1, 2]])
    }, /Matrix must be square/)
  })

  it('should throw error for non-triangular matrix', function () {
    assert.throws(function () {
      logmTriu([[1, 2], [3, 4]])
    }, /Matrix must be upper triangular/)
  })

  it('should handle 3x3 matrix with mixed positive/negative diagonal', function () {
    const T = [[2, 1, 0.5], [0, -3, 1], [0, 0, 4]]
    const result = logmTriu(T)

    // Should return complex result due to negative diagonal element
    const r11 = result[1][1]
    assert.strictEqual(r11.type, 'Complex')
    approxEqual(re(r11), Math.log(3), 1e-10)
    approxEqual(im(r11), Math.PI, 1e-10)
  })
})
