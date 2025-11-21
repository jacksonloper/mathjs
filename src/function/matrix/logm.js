import { isSparseMatrix } from '../../utils/is.js'
import { format } from '../../utils/string.js'
import { factory } from '../../utils/factory.js'

const name = 'logm'
const dependencies = ['typed', 'matrix', 'log', 'multiply', 'subtract', 'add', 'divide', 'abs', 'identity', 'schur', 'transpose', 'sqrtm', 'rsf2csf', 'logm_triu', 'conj']

export const createLogm = /* #__PURE__ */ factory(name, dependencies, ({ typed, matrix, log, multiply, subtract, add, divide, abs, identity, schur, transpose, sqrtm, rsf2csf, logm_triu, conj }) => { // eslint-disable-line camelcase
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

    // For sparse matrices, use fallback (Schur requires QR which doesn't support sparse)
    if (isSparseMatrix(A)) {
      return _logmTaylorSeries(A, n)
    }

    // Check if A is already upper triangular
    let isTriangular = true
    for (let i = 1; i < n; i++) {
      for (let j = 0; j < i; j++) {
        if (Math.abs(Number(A.get([i, j]))) > 1e-10) {
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
        if (Math.abs(Number(T.get([i, i - 1]))) > 1e-10) {
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
      // If Schur fails, fall back to Taylor series
      return _logmTaylorSeries(A, n)
    }
  }

  /**
   * Fallback implementation using inverse scaling and squaring with Taylor series
   * (for sparse matrices or if Schur fails)
   */
  function _logmTaylorSeries (A, n) {
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

    // Compute X = AScaled - I
    const X = subtract(AScaled, I)

    // Compute log(I + X) using Taylor series
    let logA = _taylorLog(X, n)

    // Undo scaling: log(A) = 2^m * log(A^(1/2^m))
    if (m > 0) {
      logA = multiply(Math.pow(2, m), logA)
    }

    return isSparseMatrix(A) ? A.createSparseMatrix(logA) : logA
  }

  /**
   * Compute log(I + X) using Taylor series
   */
  function _taylorLog (X, n) {
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

    return logA
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
