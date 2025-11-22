import { factory } from '../../../utils/factory.js'

const name = 'schur'
const dependencies = [
  'typed',
  'matrix',
  'identity',
  'multiply',
  'qr',
  'subtract',
  'norm',
  'abs',
  'add'
]

export const createSchur = /* #__PURE__ */ factory(name, dependencies, (
  {
    typed,
    matrix,
    identity,
    multiply,
    qr,
    subtract,
    norm,
    abs,
    add
  }
) => {
  /**
   *
   * Performs a real Schur decomposition of the real matrix A = UTU' where U is orthogonal
   * and T is upper quasi-triangular.
   * https://en.wikipedia.org/wiki/Schur_decomposition
   *
   * Syntax:
   *
   *     math.schur(A)
   *
   * Examples:
   *
   *     const A = [[1, 0], [-4, 3]]
   *     math.schur(A) // returns {T: [[3, 4], [0, 1]], U: [[0, 1], [-1, 0]]}
   *
   * See also:
   *
   *     sylvester, lyap, qr
   *
   * @param {Array | Matrix} A  Matrix A
   * @return {{U: Array | Matrix, T: Array | Matrix}} Object containing both matrix U and T of the Schur Decomposition A=UTU'
   */
  return typed(name, {
    Array: function (X) {
      const r = _schur(matrix(X))
      return {
        U: r.U.valueOf(),
        T: r.T.valueOf()
      }
    },

    Matrix: function (X) {
      return _schur(X)
    }
  })

  function _schur (X) {
    const n = X.size()[0]
    let A = X
    let U = identity(n)
    const maxIter = 100 * n // More iterations for larger matrices
    const eps = 1e-13

    for (let k = 0; k < maxIter; k++) {
      // Wilkinson shift
      let shift
      if (n >= 2) {
        const a = A.get([n - 2, n - 2])
        const b = A.get([n - 2, n - 1])
        const c = A.get([n - 1, n - 2])
        const d = A.get([n - 1, n - 1])

        const tr = a + d
        const det = a * d - b * c
        const disc = tr * tr / 4 - det

        if (disc >= 0) {
          const sqrtDisc = Math.sqrt(disc)
          const lambda1 = tr / 2 + sqrtDisc
          const lambda2 = tr / 2 - sqrtDisc
          shift = Math.abs(lambda1 - d) < Math.abs(lambda2 - d) ? lambda1 : lambda2
        } else {
          shift = d
        }
      } else {
        shift = A.get([n - 1, n - 1])
      }

      // Shifted QR iteration
      const shiftedA = subtract(A, multiply(shift, identity(n)))
      const QR = qr(shiftedA)
      const Q = QR.Q
      const R = QR.R
      A = add(multiply(R, Q), multiply(shift, identity(n)))
      U = multiply(U, Q)

      // Check convergence
      if (k % 10 === 0) {
        let maxSubdiag = 0
        for (let i = 2; i < n; i++) {
          for (let j = 0; j < i - 1; j++) {
            maxSubdiag = Math.max(maxSubdiag, Math.abs(A.get([i, j])))
          }
        }
        if (maxSubdiag < eps) {
          break
        }
      }
    }

    // Clean up small values
    const T = A.valueOf()
    for (let i = 1; i < n; i++) {
      for (let j = 0; j < i - 1; j++) {
        if (Math.abs(T[i][j]) < eps) {
          T[i][j] = 0
        }
      }
      // Also clean up very small first subdiagonal for quasi-triangular form
      if (Math.abs(T[i][i - 1]) < eps * 10) {
        T[i][i - 1] = 0
      }
    }

    return { U, T: matrix(T) }
  }
})
