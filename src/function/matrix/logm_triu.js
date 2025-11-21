import { factory } from '../../utils/factory.js'
import { format } from '../../utils/string.js'

const name = 'logm_triu'
const dependencies = ['typed', 'matrix', 'log', 'multiply', 'subtract', 'add', 'divide', 'abs', 'complex']

export const createLogmTriu = /* #__PURE__ */ factory(name, dependencies, ({ typed, matrix, log, multiply, subtract, add, divide, abs, complex }) => {
  /**
   * Calculate the matrix logarithm of an upper triangular matrix.
   *
   * This function implements scipy's _logmTriu approach:
   * - If all diagonal entries are non-negative real, computes real logarithm
   * - Otherwise, converts to complex and computes complex logarithm
   *
   * Uses Parlett recurrence (Algorithm 11.9 from Higham 2008) for the computation.
   *
   * Syntax:
   *
   *     math.logm_triu(T)
   *
   * Examples:
   *
   *     const T = [[2, 1], [0, 3]]
   *     math.logm_triu(T)  // returns [[log(2), ...], [0, log(3)]]
   *
   *     const T2 = [[-1, 1], [0, 2]]
   *     math.logm_triu(T2) // returns complex result since diagonal has negative value
   *
   * See also:
   *
   *     logm, schur, rsf2csf
   *
   * @param {Matrix | Array} T  An upper triangular matrix
   * @return {Matrix | Array}   The matrix logarithm of T
   */
  return typed(name, {
    Array: function (T) {
      const matrixT = matrix(T)
      const result = _logmTriu(matrixT)
      return result.valueOf()
    },

    Matrix: function (T) {
      return _logmTriu(T)
    }
  })

  /**
   * Internal implementation of triangular matrix logarithm
   */
  function _logmTriu (T) { // eslint-disable-line camelcase
    // Check if matrix is square
    const size = T.size()
    if (size.length !== 2 || size[0] !== size[1]) {
      throw new RangeError('Matrix must be square ' +
        '(size: ' + format(size) + ')')
    }

    const n = size[0]

    // Handle 1x1 matrix
    if (n === 1) {
      const val = T.get([0, 0])
      const logVal = log(val)
      return matrix([[logVal]])
    }

    // Check if matrix is upper triangular
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        const val = T.get([i, j])
        if (abs(val) > 1e-14) {
          throw new Error('Matrix must be upper triangular')
        }
      }
    }

    // Check diagonal elements to determine if we need complex arithmetic
    let hasNegativeDiag = false
    for (let i = 0; i < n; i++) {
      const diag = T.get([i, i])
      const diagNum = Number(diag)
      if (diagNum < 0) {
        hasNegativeDiag = true
        break
      }
    }

    if (hasNegativeDiag) {
      // Convert to complex and compute complex logarithm
      return _logmTriuComplex(T, n)
    } else {
      // Compute real logarithm
      return _logmTriuReal(T, n)
    }
  }

  /**
   * Compute logarithm of upper triangular matrix with all non-negative diagonal
   * (real arithmetic)
   */
  function _logmTriuReal (T, n) {
    const result = []

    // Initialize result matrix
    for (let i = 0; i < n; i++) {
      result[i] = []
      for (let j = 0; j < n; j++) {
        result[i][j] = 0
      }
    }

    // Compute diagonal elements
    for (let i = 0; i < n; i++) {
      const tii = T.get([i, i])
      result[i][i] = log(tii)
    }

    // Use Parlett's recurrence for off-diagonal elements
    for (let diag = 1; diag < n; diag++) {
      for (let i = 0; i < n - diag; i++) {
        const j = i + diag

        // Compute sum for Parlett recurrence
        let sum = 0
        for (let k = i + 1; k < j; k++) {
          sum = add(sum, subtract(
            multiply(result[i][k], T.get([k, j])),
            multiply(T.get([i, k]), result[k][j])
          ))
        }

        const tii = T.get([i, i])
        const tjj = T.get([j, j])
        const tij = T.get([i, j])
        const fii = result[i][i]
        const fjj = result[j][j]

        // Check if diagonal elements are close
        const diff = subtract(tii, tjj)
        const absDiff = abs(diff)
        const fdiff = subtract(fii, fjj)

        // If eigenvalues are well-separated, use standard Parlett formula
        if (Number(absDiff) > 1e-10) {
          // F[i,j] = (T[i,j] * (F[i,i] - F[j,j]) - sum) / (T[i,i] - T[j,j])
          const numerator = subtract(
            multiply(tij, fdiff),
            sum
          )
          result[i][j] = divide(numerator, diff)
        } else {
          // Eigenvalues nearly equal - use L'Hôpital's rule
          // For logarithm: d(log(t))/dt = 1/t
          if (Number(abs(sum)) < 1e-14) {
            // Simple case: F[i,j] = T[i,j] / T[i,i]
            result[i][j] = divide(tij, tii)
          } else {
            // With sum terms, use average
            const avgDiag = multiply(0.5, add(tii, tjj))
            result[i][j] = divide(subtract(tij, sum), avgDiag)
          }
        }
      }
    }

    return matrix(result)
  }

  /**
   * Compute logarithm of upper triangular matrix with negative diagonal elements
   * (complex arithmetic)
   */
  function _logmTriuComplex (T, n) {
    const result = []

    // Initialize result matrix with complex zeros
    for (let i = 0; i < n; i++) {
      result[i] = []
      for (let j = 0; j < n; j++) {
        result[i][j] = complex(0, 0)
      }
    }

    // Compute diagonal elements (complex logarithm)
    for (let i = 0; i < n; i++) {
      const tii = T.get([i, i])
      const tiiNum = Number(tii)

      if (tiiNum >= 0) {
        // Non-negative real: log(x) = log|x| + i*0
        result[i][i] = complex(Math.log(tiiNum), 0)
      } else {
        // Negative real: log(x) = log|x| + i*π
        result[i][i] = complex(Math.log(Math.abs(tiiNum)), Math.PI)
      }
    }

    // Use Parlett's recurrence for off-diagonal elements
    for (let diag = 1; diag < n; diag++) {
      for (let i = 0; i < n - diag; i++) {
        const j = i + diag

        // Compute sum for Parlett recurrence
        let sum = complex(0, 0)
        for (let k = i + 1; k < j; k++) {
          const term1 = multiply(result[i][k], T.get([k, j]))
          const term2 = multiply(T.get([i, k]), result[k][j])
          sum = add(sum, subtract(term1, term2))
        }

        const tii = T.get([i, i])
        const tjj = T.get([j, j])
        const tij = T.get([i, j])
        const fii = result[i][i]
        const fjj = result[j][j]

        // Check if diagonal elements are close
        const diff = subtract(tii, tjj)
        const absDiff = abs(diff)
        const fdiff = subtract(fii, fjj)

        // If eigenvalues are well-separated, use standard Parlett formula
        if (Number(absDiff) > 1e-10) {
          // F[i,j] = (T[i,j] * (F[i,i] - F[j,j]) - sum) / (T[i,i] - T[j,j])
          const numerator = subtract(
            multiply(tij, fdiff),
            sum
          )
          result[i][j] = divide(numerator, diff)
        } else {
          // Eigenvalues nearly equal - use L'Hôpital's rule
          if (Number(abs(sum)) < 1e-14) {
            // Simple case: F[i,j] = T[i,j] / T[i,i]
            result[i][j] = divide(tij, tii)
          } else {
            // With sum terms, use average
            const avgDiag = multiply(0.5, add(tii, tjj))
            result[i][j] = divide(subtract(tij, sum), avgDiag)
          }
        }
      }
    }

    return matrix(result)
  }
})
