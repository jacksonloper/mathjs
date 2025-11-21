export const logmDocs = {
  name: 'logm',
  category: 'Matrix',
  syntax: [
    'logm(A)'
  ],
  description: 'Calculate the matrix logarithm of a square matrix. ' +
    'The matrix logarithm is the inverse of the matrix exponential. ' +
    'Not to be confused with log(a), which performs element-wise logarithm. ' +
    'Implements Higham\'s Schur-Parlett algorithm (Algorithm 11.9) with direct 2x2 block handling for numerical stability.',
  examples: [
    'logm([[1, 2], [0, 1]])',
    'A = expm([[1, 2], [3, 4]])',
    'logm(A)'
  ],
  seealso: [
    'expm', 'log', 'sqrtm', 'schur'
  ]
}
