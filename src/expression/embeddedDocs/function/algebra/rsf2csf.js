export const rsf2csfDocs = {
  name: 'rsf2csf',
  category: 'Algebra',
  syntax: [
    'rsf2csf(T, U)'
  ],
  description: 'Convert real Schur form to complex Schur form. Converts a quasi-diagonal real-valued Schur form to the upper-triangular complex-valued Schur form.',
  examples: [
    'A = [[0, -1], [1, 0]]',
    's = schur(A)',
    'c = rsf2csf(s.T, s.U)'
  ],
  seealso: ['schur', 'eigs', 'qr']
}
