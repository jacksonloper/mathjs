export const logmTriuDocs = {
  name: 'logm_triu',
  category: 'Matrix',
  syntax: [
    'logm_triu(T)'
  ],
  description:
    'Calculate the matrix logarithm of an upper triangular matrix. ' +
    'This function follows scipy\'s approach: if all diagonal entries are ' +
    'non-negative real, it computes a real logarithm; otherwise, it converts ' +
    'to complex and computes a complex logarithm. Uses Parlett recurrence ' +
    '(Algorithm 11.9 from Higham 2008).',
  examples: [
    'logm_triu([[2, 1], [0, 3]])',
    'logm_triu([[-1, 1], [0, 2]])'
  ],
  seealso: [
    'logm', 'expm', 'schur', 'rsf2csf'
  ]
}
