/**
 * Type-safe matrix library with compile-time dimension checking.
 *
 * Uses branded types to ensure matrix operations are only performed
 * on matrices with compatible dimensions.
 */

// Branded type for compile-time dimension tracking
declare const __rows: unique symbol;
declare const __cols: unique symbol;

/**
 * A matrix with compile-time tracked dimensions.
 * @typeParam R - Number of rows (literal number type)
 * @typeParam C - Number of columns (literal number type)
 */
export interface Matrix<R extends number, C extends number> {
  readonly [__rows]: R;
  readonly [__cols]: C;
  readonly rows: R;
  readonly cols: C;
  readonly data: ReadonlyArray<ReadonlyArray<number>>;
}

/**
 * Internal helper to construct a Matrix value.
 */
function createMatrix<R extends number, C extends number>(
  rows: R,
  cols: C,
  data: number[][],
): Matrix<R, C> {
  return { rows, cols, data } as Matrix<R, C>;
}

/**
 * Create a matrix from a 2D array of numbers.
 * Validates that all rows have the same length.
 */
export function matrix<R extends number, C extends number>(
  rows: R,
  cols: C,
  data: number[][],
): Matrix<R, C> {
  if (data.length !== rows) {
    throw new Error(`Expected ${rows} rows but got ${data.length}`);
  }
  for (let i = 0; i < data.length; i++) {
    if (data[i].length !== cols) {
      throw new Error(
        `Expected ${cols} columns in row ${i} but got ${data[i].length}`,
      );
    }
  }
  return createMatrix(rows, cols, data.map((row) => [...row]));
}

/**
 * Create a matrix filled with zeros.
 */
export function zeros<R extends number, C extends number>(
  rows: R,
  cols: C,
): Matrix<R, C> {
  const data = Array.from({ length: rows }, () => Array(cols).fill(0));
  return createMatrix(rows, cols, data);
}

/**
 * Create a matrix filled with ones.
 */
export function ones<R extends number, C extends number>(
  rows: R,
  cols: C,
): Matrix<R, C> {
  const data = Array.from({ length: rows }, () => Array(cols).fill(1));
  return createMatrix(rows, cols, data);
}

/**
 * Create an identity matrix.
 */
export function identity<N extends number>(size: N): Matrix<N, N> {
  const data = Array.from({ length: size }, (_, i) =>
    Array.from({ length: size }, (_, j) => (i === j ? 1 : 0)),
  );
  return createMatrix(size, size, data);
}

/**
 * Add two matrices of the same dimensions.
 */
export function add<R extends number, C extends number>(
  a: Matrix<R, C>,
  b: Matrix<R, C>,
): Matrix<R, C> {
  const data = (a.data as number[][]).map((row, i) =>
    row.map((val, j) => val + (b.data[i][j] as number)),
  );
  return createMatrix(a.rows, a.cols, data);
}

/**
 * Subtract matrix b from matrix a. Both must have the same dimensions.
 */
export function subtract<R extends number, C extends number>(
  a: Matrix<R, C>,
  b: Matrix<R, C>,
): Matrix<R, C> {
  const data = (a.data as number[][]).map((row, i) =>
    row.map((val, j) => val - (b.data[i][j] as number)),
  );
  return createMatrix(a.rows, a.cols, data);
}

/**
 * Multiply two matrices. The number of columns in a must equal
 * the number of rows in b (enforced at the type level).
 */
export function multiply<R extends number, N extends number, C extends number>(
  a: Matrix<R, N>,
  b: Matrix<N, C>,
): Matrix<R, C> {
  const result: number[][] = [];
  for (let i = 0; i < a.rows; i++) {
    result[i] = [];
    for (let j = 0; j < b.cols; j++) {
      let sum = 0;
      for (let k = 0; k < a.cols; k++) {
        sum += (a.data[i][k] as number) * (b.data[k][j] as number);
      }
      result[i][j] = sum;
    }
  }
  return createMatrix(a.rows as R, b.cols as C, result);
}

/**
 * Multiply every element by a scalar.
 */
export function scale<R extends number, C extends number>(
  m: Matrix<R, C>,
  scalar: number,
): Matrix<R, C> {
  const data = (m.data as number[][]).map((row) =>
    row.map((val) => val * scalar),
  );
  return createMatrix(m.rows, m.cols, data);
}

/**
 * Transpose a matrix, swapping rows and columns.
 */
export function transpose<R extends number, C extends number>(
  m: Matrix<R, C>,
): Matrix<C, R> {
  const data: number[][] = [];
  for (let j = 0; j < m.cols; j++) {
    data[j] = [];
    for (let i = 0; i < m.rows; i++) {
      data[j][i] = m.data[i][j] as number;
    }
  }
  return createMatrix(m.cols as C, m.rows as R, data);
}

/**
 * Get a single element from the matrix.
 */
export function get<R extends number, C extends number>(
  m: Matrix<R, C>,
  row: number,
  col: number,
): number {
  if (row < 0 || row >= m.rows || col < 0 || col >= m.cols) {
    throw new Error(
      `Index (${row}, ${col}) out of bounds for ${m.rows}x${m.cols} matrix`,
    );
  }
  return m.data[row][col] as number;
}

/**
 * Compute the determinant of a square matrix.
 */
export function determinant<N extends number>(m: Matrix<N, N>): number {
  if (m.rows !== m.cols) {
    throw new Error("Determinant is only defined for square matrices");
  }
  const n = m.rows;
  if (n === 1) return m.data[0][0] as number;
  if (n === 2) {
    return (
      (m.data[0][0] as number) * (m.data[1][1] as number) -
      (m.data[0][1] as number) * (m.data[1][0] as number)
    );
  }

  let det = 0;
  for (let j = 0; j < n; j++) {
    const minor = getMinor(m.data as number[][], 0, j);
    det += (j % 2 === 0 ? 1 : -1) * (m.data[0][j] as number) * det2(minor);
  }
  return det;
}

function getMinor(data: number[][], skipRow: number, skipCol: number): number[][] {
  return data
    .filter((_, i) => i !== skipRow)
    .map((row) => row.filter((_, j) => j !== skipCol));
}

function det2(data: number[][]): number {
  const n = data.length;
  if (n === 1) return data[0][0];
  if (n === 2) return data[0][0] * data[1][1] - data[0][1] * data[1][0];
  let result = 0;
  for (let j = 0; j < n; j++) {
    const minor = getMinor(data, 0, j);
    result += (j % 2 === 0 ? 1 : -1) * data[0][j] * det2(minor);
  }
  return result;
}

/**
 * Check if two matrices are equal (element-wise).
 */
export function equals<R extends number, C extends number>(
  a: Matrix<R, C>,
  b: Matrix<R, C>,
  epsilon: number = 1e-10,
): boolean {
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      if (Math.abs((a.data[i][j] as number) - (b.data[i][j] as number)) > epsilon) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Convert a matrix to a human-readable string.
 */
export function toString<R extends number, C extends number>(
  m: Matrix<R, C>,
): string {
  return (m.data as number[][])
    .map((row) => row.map((v) => v.toString()).join("\t"))
    .join("\n");
}
