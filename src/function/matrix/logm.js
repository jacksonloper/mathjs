import { isSparseMatrix } from '../../utils/is.js'
import { format } from '../../utils/string.js'
import { factory } from '../../utils/factory.js'

const name = 'logm'
const dependencies = ['typed', 'matrix', 'log', 'multiply', 'subtract', 'add', 'divide', 'abs', 'identity', 'schur', 'transpose', 'rsf2csf', 'logm_triu', 'conj']

// Constants for numerical tolerances
const EPS = 2.220446049250313e-16 // machine epsilon
const TRIANGULAR_TOL = 100 * EPS // tolerance for checking triangularity
const BLOCK_DETECTION_TOL = 10 * EPS // tolerance for detecting 2x2 blocks

export const createLogm = /* #__PURE__ */ factory(name, dependencies, ({ typed, matrix, log, multiply, subtract, add, divide, abs, identity, schur, transpose, rsf2csf, logm_triu, conj }) => { // eslint-disable-line camelcase
  /**
   * Calculate the matrix logarithm of a square matrix. The matrix logarithm is
   * the inverse of the matrix exponential. Not to be confused with log(a),
   * which performs element-wise logarithm.
   *
   * This function implements the scipy-compatible Schur-Parlett algorithm from:
   * - scipy.linalg.logm (SciPy's matrix logarithm implementation)
   * - "Functions of Matrices: Theory and Computation" by N. J. Higham (2008)
   * - Al-Mohy and Higham (2011), "Improved Inverse Scaling and Squaring Algorithms"
   *
   * The algorithm (matching scipy.linalg.logm):
   * 1. If A is already upper triangular, apply logm_triu directly
   * 2. Otherwise, compute real Schur decomposition A = U*T*U^T
   * 3. If T has 2x2 blocks (complex eigenvalues), convert to complex Schur via rsf2csf
   * 4. Compute log(T) using logm_triu (Parlett recurrence)
   * 5. Transform back: log(A) = U*log(T)*U^H
   *
   * This matches scipy's approach exactly, including the rsf2csf conversion step
   * for matrices with complex eigenvalues.
   *
   * Syntax:
   *
   *     math.logm(A)
   *
   * Examples:
   *
   *     const A = [[1, 2], [0, 1]]
   *     math.logm(A)        // returns approximately [[0, 2], [0, 0]]
   *
   *     const B = math.expm([[1, 2], [3, 4]])
   *     const C = math.logm(B)  // C should be approximately [[1, 2], [3, 4]]
   *
   * See also:
   *
   *     expm, log, sqrtm, schur, rsf2csf, logm_triu
   *
   * @param {Matrix | Array} A  A square matrix
   * @return {Matrix | Array}   The matrix logarithm of A
   */
  return typed(name, {
    Array: function (A) {
      const matrixA = matrix(A)
      const result = _logm(matrixA)
      return result.valueOf()
    },

    Matrix: function (A) {
      return _logm(A)
    }
  })

  /**
   * Internal implementation of matrix logarithm using scipy-compatible Schur-Parlett algorithm
   * Falls back to Taylor series for sparse matrices or if Schur fails
   */
  function _logm (A) {
    // Check if matrix is square
    const size = A.size()
    if (size.length !== 2 || size[0] !== size[1]) {
      throw new RangeError('Matrix must be square ' +
        '(size: ' + format(size) + ')')
    }

    const n = size[0]

    // Handle 1x1 matrix
    if (n === 1) {
      const val = A.get([0, 0])
      const logVal = log(val)
      const result = matrix([[logVal]])
      return isSparseMatrix(A) ? A.createSparseMatrix(result) : result
    }

    // For sparse matrices, throw an error (Schur requires QR which doesn't support sparse)
    // Taylor series fallback was removed as it's not numerically stable
    if (isSparseMatrix(A)) {
      throw new Error('logm does not support sparse matrices. Convert to dense matrix first using matrix.toDenseMatrix() or matrix.toArray()')
    }

    // Check if A is already upper triangular
    let isTriangular = true
    for (let i = 1; i < n; i++) {
      for (let j = 0; j < i; j++) {
        const val = A.get([i, j])
        const absVal = abs(val)
        if (Number(absVal) > TRIANGULAR_TOL) {
          isTriangular = false
          break
        }
      }
      if (!isTriangular) break
    }

    if (isTriangular) {
      // A is already triangular - use logm_triu directly
      return logm_triu(A)
    }

    try {
      // Compute Schur decomposition
      const schurResult = schur(A)
      let U = schurResult.U
      let T = schurResult.T

      // Check if T is strictly upper triangular or has 2x2 blocks
      let hasBlocks = false
      for (let i = 1; i < n; i++) {
        const val = T.get([i, i - 1])
        const absVal = abs(val)
        if (Number(absVal) > BLOCK_DETECTION_TOL) {
          hasBlocks = true
          break
        }
      }

      // If T has 2x2 blocks, convert to complex Schur form (matching scipy)
      if (hasBlocks) {
        const rsf2csfResult = rsf2csf(T, U)
        T = rsf2csfResult.T
        U = rsf2csfResult.U
      }

      // Compute log(T) using logm_triu
      const logT = logm_triu(T)

      // Transform back: log(A) = U * log(T) * U^H
      // U^H is conjugate transpose
      const UH = transpose(conj(U))
      let result = multiply(U, logT)
      result = multiply(result, UH)

      return result
    } catch (error) {
      // If Schur decomposition fails, re-throw the error with context
      throw new Error('Matrix logarithm computation failed: ' + error.message)
    }
  }
})
