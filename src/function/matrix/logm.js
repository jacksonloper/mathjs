import { isSparseMatrix } from '../../utils/is.js'
import { format } from '../../utils/string.js'
import { factory } from '../../utils/factory.js'

const name = 'logm'
const dependencies = ['typed', 'matrix', 'log', 'multiply', 'subtract', 'add', 'divide', 'abs', 'identity', 'sqrtm']

export const createLogm = /* #__PURE__ */ factory(name, dependencies, ({ typed, matrix, log, multiply, subtract, add, divide, abs, identity, sqrtm }) => {
  /**
   * Calculate the matrix logarithm of a square matrix. The matrix logarithm is
   * the inverse of the matrix exponential. Not to be confused with log(a),
   * which performs element-wise logarithm.
   *
   * This function uses the inverse scaling and squaring method with Taylor series.
   * The matrix is repeatedly square-rooted until it is close to the identity matrix,
   * then the Taylor series log(I + X) ≈ sum_{k=1}^∞ (-1)^{k+1} X^k / k is used,
   * and finally the result is scaled back by 2^m. This approach mirrors the
   * scipy.linalg.logm implementation for real matrices.
   *
   * For more details, see:
   * - "Functions of Matrices: Theory and Computation" by N. J. Higham (2008)
   * - scipy.linalg.logm documentation
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
   *     expm, log, sqrtm
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
   * Internal implementation of matrix logarithm
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

    // Use inverse scaling and squaring method
    // The idea: repeatedly take square roots until ||A - I|| < 0.5,
    // then use Taylor series log(A) = log(I + X) ≈ sum_{k=1}^∞ (-1)^{k+1} X^k / k
    // Finally, multiply result by 2^m where m is number of square roots taken

    const I = identity(n)
    let AScaled = A
    let m = 0

    // Take square roots until close to identity
    const maxSquareRoots = 20
    for (let i = 0; i < maxSquareRoots; i++) {
      // Compute ||A - I||_1 (column sum norm)
      let normValue = 0
      for (let col = 0; col < n; col++) {
        let colSum = 0
        for (let row = 0; row < n; row++) {
          const val = AScaled.get([row, col])
          const diff = row === col ? subtract(val, 1) : val
          colSum = add(colSum, abs(diff))
        }
        const colSumNum = Number(colSum)
        if (colSumNum > normValue) {
          normValue = colSumNum
        }
      }

      if (normValue < 0.5) {
        break
      }

      // Take square root
      AScaled = sqrtm(AScaled)
      m++

      if (m >= maxSquareRoots) {
        break
      }
    }

    // Compute X = A - I
    const X = subtract(AScaled, I)

    // Compute log(I + X) using Taylor series: sum_{k=1}^p (-1)^{k+1} X^k / k
    const p = 25 // Number of terms in series

    let logA = zeros(n, n)
    let Xpower = X

    for (let k = 1; k <= p; k++) {
      const sign = (k % 2 === 1) ? 1 : -1
      const term = divide(multiply(sign, Xpower), k)

      logA = add(logA, term)

      if (k < p) {
        Xpower = multiply(Xpower, X)
      }
    }

    // Undo scaling: log(A) = 2^m * log(A^(1/2^m))
    if (m > 0) {
      logA = multiply(Math.pow(2, m), logA)
    }

    return isSparseMatrix(A) ? A.createSparseMatrix(logA) : logA
  }

  /**
   * Create a zero matrix
   */
  function zeros (rows, cols) {
    const result = []
    for (let i = 0; i < rows; i++) {
      result[i] = []
      for (let j = 0; j < cols; j++) {
        result[i][j] = 0
      }
    }
    return matrix(result)
  }
})
