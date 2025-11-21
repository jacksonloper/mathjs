import { isSparseMatrix } from '../../utils/is.js'
import { format } from '../../utils/string.js'
import { factory } from '../../utils/factory.js'

const name = 'logm'
const dependencies = ['typed', 'matrix', 'log', 'multiply', 'subtract', 'add', 'divide', 'abs', 'identity', 'schur', 'transpose', 'sqrtm']

export const createLogm = /* #__PURE__ */ factory(name, dependencies, ({ typed, matrix, log, multiply, subtract, add, divide, abs, identity, schur, transpose, sqrtm }) => {
  /**
   * Calculate the matrix logarithm of a square matrix. The matrix logarithm is
   * the inverse of the matrix exponential. Not to be confused with log(a),
   * which performs element-wise logarithm.
   *
   * This function uses the Schur-Parlett algorithm as described in:
   * - "Functions of Matrices: Theory and Computation" by N. J. Higham (2008)
   * - Al-Mohy and Higham (2011), "Improved Inverse Scaling and Squaring Algorithms"
   *
   * The algorithm:
   * 1. Computes the Schur decomposition A = U*T*U^T
   * 2. Computes log(T) using Parlett recurrence on the triangular matrix
   * 3. Transforms back: log(A) = U*log(T)*U^T
   *
   * This approach mirrors scipy.linalg.logm and provides better numerical stability
   * for matrices with complex eigenvalues, negative eigenvalues, and ill-conditioned cases.
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
   *     expm, log, sqrtm, schur
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
   * Internal implementation of matrix logarithm using Schur-Parlett algorithm
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

    try {
      // Try Schur-Parlett algorithm (best for general case)
      const schurResult = schur(A)
      const U = schurResult.U
      const T = schurResult.T

      // Compute log(T) using Parlett recurrence on upper triangular matrix
      const logT = _logTriangular(T, n)

      // Transform back: log(A) = U * log(T) * U^T
      const UT = transpose(U)
      let result = multiply(U, logT)
      result = multiply(result, UT)

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

  /**
   * Compute logarithm of upper triangular/quasi-triangular matrix using
   * Parlett recurrence (Algorithm 11.9 from Higham 2008) with 2x2 block handling
   */
  function _logTriangular (T, n) {
    const result = []

    // Initialize result matrix
    for (let i = 0; i < n; i++) {
      result[i] = []
      for (let j = 0; j < n; j++) {
        result[i][j] = 0
      }
    }

    // Compute diagonal blocks (1x1 or 2x2)
    // Detect 2x2 blocks by checking subdiagonal elements
    let i = 0
    while (i < n) {
      if (i < n - 1 && Math.abs(Number(T.get([i + 1, i]))) > 1e-10) {
        // 2x2 block detected at [i:i+2, i:i+2]
        const block = [
          [T.get([i, i]), T.get([i, i + 1])],
          [T.get([i + 1, i]), T.get([i + 1, i + 1])]
        ]
        const logBlock = _log2x2(block)

        // Store the result
        result[i][i] = logBlock[0][0]
        result[i][i + 1] = logBlock[0][1]
        result[i + 1][i] = logBlock[1][0]
        result[i + 1][i + 1] = logBlock[1][1]

        i += 2 // Skip next diagonal element (part of this block)
      } else {
        // 1x1 block (real eigenvalue)
        const tii = T.get([i, i])
        result[i][i] = log(tii)
        i += 1
      }
    }

    // Use Parlett's recurrence for off-diagonal elements
    // Must skip over 2x2 blocks on diagonal
    for (let diag = 1; diag < n; diag++) {
      for (let i = 0; i < n - diag; i++) {
        const j = i + diag

        // Skip if this element is part of a diagonal 2x2 block
        if (Math.abs(Number(T.get([i + 1, i]))) > 1e-10 && j === i + 1) {
          continue // Already computed as part of 2x2 block
        }

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

        // Check if diagonal elements (eigenvalues) are close
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
          // So F[i,j] = T[i,j] / T[i,i] when T[i,i] = T[j,j] and no sum terms
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
   * Compute logarithm of a 2x2 matrix with complex eigenvalues
   * Uses the formula: log(A) = α*I + β*A
   * where α and β are computed from eigenvalues
   */
  function _log2x2 (block) {
    const a = block[0][0]
    const b = block[0][1]
    const c = block[1][0]
    const d = block[1][1]

    // Compute eigenvalues using quadratic formula
    // For [[a,b],[c,d]]: λ = (a+d ± sqrt((a+d)²-4(ad-bc)))/2
    const trace = add(a, d)
    const det = subtract(multiply(a, d), multiply(b, c))
    const discriminant = subtract(multiply(trace, trace), multiply(4, det))

    const discNum = Number(discriminant)

    if (discNum >= 0) {
      // Real eigenvalues
      const sqrtDisc = Math.sqrt(discNum)
      const lambda1 = multiply(0.5, add(trace, sqrtDisc))
      const lambda2 = multiply(0.5, subtract(trace, sqrtDisc))

      const logLambda1 = log(lambda1)
      const logLambda2 = log(lambda2)

      // Compute coefficients for log(A) = α*I + β*A
      const lambdaDiff = subtract(lambda1, lambda2)

      if (Number(abs(lambdaDiff)) > 1e-10) {
        // β = (log(λ₁) - log(λ₂)) / (λ₁ - λ₂)
        const beta = divide(subtract(logLambda1, logLambda2), lambdaDiff)
        // α = log(λ₁) - β*λ₁ = log(λ₂) - β*λ₂
        const alpha = subtract(logLambda1, multiply(beta, lambda1))

        // log(A) = α*I + β*A
        return [
          [add(alpha, multiply(beta, a)), multiply(beta, b)],
          [multiply(beta, c), add(alpha, multiply(beta, d))]
        ]
      } else {
        // Equal eigenvalues - use simpler formula
        const logLambda = multiply(0.5, add(logLambda1, logLambda2))
        return [
          [logLambda, 0],
          [0, logLambda]
        ]
      }
    } else {
      // Complex conjugate eigenvalues: λ = α ± iβ
      // where α = trace/2, β = sqrt(-discriminant)/2
      const alpha = multiply(0.5, trace)
      const beta = multiply(0.5, Math.sqrt(-discNum))

      // For complex λ = α + iβ:
      // log(λ) = log|λ| + i*arg(λ) = log(sqrt(α²+β²)) + i*atan2(β, α)
      const magnitude = Math.sqrt(Number(multiply(alpha, alpha)) + beta * beta)
      const logMag = Math.log(magnitude)
      const angle = Math.atan2(beta, Number(alpha))

      // log(λ₁) = logMag + i*angle
      // log(λ₂) = logMag - i*angle (conjugate)

      // For the formula log(A) = α*I + β*A with complex eigenvalues:
      // We need: (log(λ₁) - log(λ₂)) / (λ₁ - λ₂) and related terms
      // λ₁ - λ₂ = 2iβ
      // log(λ₁) - log(λ₂) = 2i*angle

      // β_coeff = (log(λ₁) - log(λ₂)) / (λ₁ - λ₂) = (2i*angle) / (2iβ) = angle/β
      const betaCoeff = angle / beta

      // α_coeff = log(λ₁) - β_coeff*λ₁
      // = (logMag + i*angle) - (angle/β)*(α + iβ)
      // = logMag + i*angle - α*angle/β - i*angle
      // = logMag - α*angle/β
      const alphaCoeff = logMag - Number(alpha) * angle / beta

      // log(A) = α_coeff*I + β_coeff*A (result is real)
      return [
        [alphaCoeff + betaCoeff * Number(a), betaCoeff * Number(b)],
        [betaCoeff * Number(c), alphaCoeff + betaCoeff * Number(d)]
      ]
    }
  }
})
