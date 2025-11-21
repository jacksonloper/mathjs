import { factory } from '../../../utils/factory.js'

const name = 'rsf2csf'
const dependencies = [
  'typed',
  'matrix',
  'addScalar',
  'subtract',
  'multiply',
  'divideScalar',
  'sqrt',
  'abs',
  'conj',
  'larger',
  'dotMultiply',
  'complex'
]

export const createRsf2csf = /* #__PURE__ */ factory(name, dependencies, (
  { typed, matrix, addScalar, subtract, multiply, divideScalar, sqrt, abs, conj, larger, complex }
) => {
  /**
   * Convert real Schur form to complex Schur form.
   *
   * Convert a quasi-diagonal real-valued Schur form to the upper-triangular
   * complex-valued Schur form.
   *
   * Syntax:
   *
   *    math.rsf2csf(T, U)
   *
   * Example:
   *
   *    const A = [[0, 2, 2], [0, 1, 2], [1, 0, 1]]
   *    const {T, U} = math.schur(A)
   *    const {T: Tc, U: Uc} = math.rsf2csf(T, U)
   *    // Tc is now upper triangular (complex)
   *
   * @param {Matrix | Array} T  Real Schur form (quasi-triangular)
   * @param {Matrix | Array} U  Schur transformation matrix
   * @return {{T: Matrix | Array, U: Matrix | Array}}  Complex Schur form
   */
  return typed(name, {
    'Matrix, Matrix': function (T, U) {
      const result = _rsf2csf(T.toArray(), U.toArray())
      return {
        T: matrix(result.T),
        U: matrix(result.U)
      }
    },

    'Array, Array': function (T, U) {
      return _rsf2csf(T, U)
    }
  })

  function _rsf2csf (T, U) {
    const n = T.length
    const eps = 2.220446049250313e-16 // machine epsilon

    // Validate inputs
    if (!T || !U || T.length !== U.length) {
      throw new Error('Input matrices must be square and have the same size')
    }

    for (let i = 0; i < n; i++) {
      if (T[i].length !== n || U[i].length !== n) {
        throw new Error('Input matrices must be square')
      }
    }

    // Create complex copies
    const Tc = []
    const Uc = []
    for (let i = 0; i < n; i++) {
      Tc[i] = []
      Uc[i] = []
      for (let j = 0; j < n; j++) {
        Tc[i][j] = complex(T[i][j], 0)
        Uc[i][j] = complex(U[i][j], 0)
      }
    }

    // Process from bottom-right to top-left
    for (let m = n - 1; m >= 1; m--) {
      const subdiag = abs(Tc[m][m - 1])
      const diag1 = abs(Tc[m - 1][m - 1])
      const diag2 = abs(Tc[m][m])
      const diagSum = addScalar(diag1, diag2)

      // Check if we have a 2x2 block
      if (larger(subdiag, multiply(eps, diagSum))) {
        // Compute eigenvalues of the 2x2 block
        const a = Tc[m - 1][m - 1]
        const b = Tc[m - 1][m]
        const c = Tc[m][m - 1]
        const d = Tc[m][m]

        // trace and determinant
        const tr = addScalar(a, d)
        const det = subtract(multiply(a, d), multiply(b, c))

        // discriminant: tr^2/4 - det
        const trHalf = divideScalar(tr, 2)
        const disc = subtract(multiply(trHalf, trHalf), det)

        // Compute eigenvalues: lambda = tr/2 ± sqrt(disc)
        const sqrtDisc = sqrt(disc)
        const lambda1 = addScalar(trHalf, sqrtDisc)
        // const lambda2 = subtract(tr_half, sqrtDisc)

        // mu = lambda1 - T[m,m]
        const mu = subtract(lambda1, Tc[m][m])

        // r = sqrt(|mu|^2 + |T[m, m-1]|^2)
        const muAbsSq = addScalar(
          multiply(mu.re, mu.re),
          multiply(mu.im, mu.im)
        )
        const cAbsSq = multiply(Tc[m][m - 1].re, Tc[m][m - 1].re)
        const r = sqrt(addScalar(muAbsSq, cAbsSq))

        // Check for numerical instability (r near zero)
        const EPS_R = 1e-15 // Small threshold for r
        if (Number(r.re) < EPS_R) {
          // Skip this block if r is too small to avoid division by near-zero
          Tc[m][m - 1] = complex(0, 0)
          continue
        }

        // Givens rotation parameters: c = mu / r, s = T[m, m-1] / r
        const cGivens = divideScalar(mu, r)
        const sGivens = divideScalar(Tc[m][m - 1], r)

        // Construct Givens rotation: G = [[conj(c), s], [-s, c]]
        const G = [
          [conj(cGivens), sGivens],
          [multiply(-1, sGivens), cGivens]
        ]

        // Apply G on the left: T[m-1:m+1, m-1:] = G @ T[m-1:m+1, m-1:]
        for (let j = m - 1; j < n; j++) {
          const t1 = Tc[m - 1][j]
          const t2 = Tc[m][j]

          Tc[m - 1][j] = addScalar(multiply(G[0][0], t1), multiply(G[0][1], t2))
          Tc[m][j] = addScalar(multiply(G[1][0], t1), multiply(G[1][1], t2))
        }

        // Construct G^H = [[c, -s], [s, conj(c)]]
        const GH = [
          [cGivens, multiply(-1, sGivens)],
          [sGivens, conj(cGivens)]
        ]

        // Apply G^H on the right: T[:m+1, m-1:m+1] = T[:m+1, m-1:m+1] @ G^H
        for (let i = 0; i <= m; i++) {
          const t1 = Tc[i][m - 1]
          const t2 = Tc[i][m]

          Tc[i][m - 1] = addScalar(multiply(t1, GH[0][0]), multiply(t2, GH[1][0]))
          Tc[i][m] = addScalar(multiply(t1, GH[0][1]), multiply(t2, GH[1][1]))
        }

        // Apply G^H on U: U[:, m-1:m+1] = U[:, m-1:m+1] @ G^H
        for (let i = 0; i < n; i++) {
          const u1 = Uc[i][m - 1]
          const u2 = Uc[i][m]

          Uc[i][m - 1] = addScalar(multiply(u1, GH[0][0]), multiply(u2, GH[1][0]))
          Uc[i][m] = addScalar(multiply(u1, GH[0][1]), multiply(u2, GH[1][1]))
        }
      }

      // Zero out the subdiagonal element
      Tc[m][m - 1] = complex(0, 0)
    }

    return { T: Tc, U: Uc }
  }
})
