use std::fmt;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

use crate::vector::Vector3d;
use crate::vector::Vector4d;

///////////////////////////////////////////////////////////////////////////////
// Matrix3                                                                   //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy)]
pub struct Matrix3d {
  pub data: [f64; 9],
}

// Implementing Default trait for Matrix3d
impl Default for Matrix3d {
  fn default() -> Self {
    Matrix3d { data: [0.0; 9] }
  }
}

impl Matrix3d {
  pub fn new(data: [f64; 9]) -> Self {
    Matrix3d { data }
  }

  pub fn at(&self, row: usize, col: usize) -> &f64 {
    assert!(row < 3 && col < 3, "Index out of bounds");
    &self.data[row * 3 + col]
  }

  pub fn set(&mut self, row: usize, col: usize, value: f64) {
    assert!(row < 3 && col < 3, "Index out of bounds");
    self.data[row * 3 + col] = value;
  }

  pub fn eye() -> Self {
    let mut data = [0.0; 9];
    data[0] = 1.0;
    data[4] = 1.0;
    data[8] = 1.0;
    Self { data }
  }

  pub fn trace(&self) -> f64 {
    self.data[0] + self.data[4] + self.data[8]
  }

  #[allow(non_snake_case)]
  pub fn transpose(&self) -> Self {
    let A = self.data;
    let mut B = [0.0; 9];

    B[0] = A[0];
    B[1] = A[3];
    B[2] = A[6];

    B[3] = A[1];
    B[4] = A[4];
    B[5] = A[7];

    B[6] = A[2];
    B[7] = A[5];
    B[8] = A[8];

    Self { data: B }
  }

  pub fn add(&self, rhs: &Matrix3d) -> Self {
    let mut result = Matrix3d::default();
    for i in 0..9 {
      result.data[i] = self.data[i] + rhs.data[i];
    }
    result
  }

  pub fn sub(&self, rhs: &Matrix3d) -> Self {
    let mut result = Matrix3d::default();
    for i in 0..9 {
      result.data[i] = self.data[i] - rhs.data[i];
    }
    result
  }

  pub fn scale(&self, rhs: f64) -> Self {
    let mut result = Matrix3d::default();
    for i in 0..9 {
      result.data[i] = rhs * self.data[i];
    }
    result
  }

  #[allow(non_snake_case)]
  pub fn dot(&self, rhs: &Matrix3d) -> Self {
    let A = self.data;
    let B = rhs.data;
    let mut C = [0.0; 9];

    C[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
    C[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
    C[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];

    C[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
    C[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
    C[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];

    C[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
    C[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
    C[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];

    Matrix3d { data: C }
  }

  #[allow(non_snake_case)]
  pub fn dot_vec(&self, rhs: &Vector3d) -> Vector3d {
    let M = self.data;
    let v = rhs.data;
    let mut y = [0.0; 3];

    y[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2];
    y[1] = M[3] * v[0] + M[4] * v[1] + M[5] * v[2];
    y[2] = M[6] * v[0] + M[7] * v[1] + M[8] * v[2];

    Vector3d { data: y }
  }

  pub fn is_close(&self, rhs: &Self, epsilon: f64) -> bool {
    self.data.iter().zip(rhs.data.iter()).all(|(&a, &b)| {
      let diff = if a > b { a - b } else { b - a };
      diff < epsilon
    })
  }
}

// -- Matrix3d + Matrix3d
impl Add<&Matrix3d> for &Matrix3d {
  type Output = Matrix3d;

  fn add(self, rhs: &Matrix3d) -> Self::Output {
    self.add(rhs)
  }
}

// -- Matrix3d - Matrix3d
impl Sub<&Matrix3d> for &Matrix3d {
  type Output = Matrix3d;

  fn sub(self, rhs: &Matrix3d) -> Self::Output {
    self.sub(rhs)
  }
}

// -- Matrix3d * f64
impl Mul<f64> for &Matrix3d {
  type Output = Matrix3d;

  fn mul(self, rhs: f64) -> Self::Output {
    self.scale(rhs)
  }
}

// -- f64 * Matrix3d
impl Mul<&Matrix3d> for f64 {
  type Output = Matrix3d;

  fn mul(self, rhs: &Matrix3d) -> Self::Output {
    rhs * self
  }
}

// -- Matrix3d * Vector3d
impl Mul<&Vector3d> for &Matrix3d {
  type Output = Vector3d;

  fn mul(self, rhs: &Vector3d) -> Self::Output {
    self.dot_vec(rhs)
  }
}

// -- Matrix3d * Matrix3d
impl Mul<&Matrix3d> for &Matrix3d {
  type Output = Matrix3d;

  fn mul(self, rhs: &Matrix3d) -> Self::Output {
    self.dot(rhs)
  }
}

// --- Neg
impl Neg for Matrix3d {
  type Output = Matrix3d;

  #[allow(non_snake_case)]
  fn neg(self) -> Self {
    let mut M = [0.0; 9];

    M[0] = -self.data[0];
    M[1] = -self.data[1];
    M[2] = -self.data[2];
    M[3] = -self.data[3];
    M[4] = -self.data[4];
    M[5] = -self.data[5];
    M[6] = -self.data[6];
    M[7] = -self.data[7];
    M[8] = -self.data[8];

    Matrix3d { data: M }
  }
}

#[cfg(test)]
mod matrix3_tests {
  // Note this useful idiom: importing names from outer (for mod tests) scope.
  use super::*;

  #[test]
  #[allow(non_snake_case)]
  fn test_matrix3() {
    // Create and use a matrix
    #[rustfmt::skip]
    let A = Matrix3d::new([
      1.0, 2.0, 3.0,
      4.0, 5.0, 6.0,
      7.0, 8.0, 9.0
    ]);
    let b = Vector3d::new([1.0, 2.0, 3.0]);

    #[rustfmt::skip]
    let expected_add = Matrix3d::new([
      2.0, 4.0, 6.0,
      8.0, 10.0, 12.0,
      14.0, 16.0, 18.0
    ]);

    #[rustfmt::skip]
    let expected_sub = Matrix3d::new([
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
      0.0, 0.0, 0.0,
    ]);

    #[rustfmt::skip]
    let expected_scale = Matrix3d::new([
      2.0, 4.0, 6.0,
      8.0, 10.0, 12.0,
      14.0, 16.0, 18.0
    ]);

    let expected_dot_vec = Vector3d::new([14.0, 32.0, 50.0]);

    #[rustfmt::skip]
    let expected_dot = Matrix3d::new([
      30.0, 36.0, 42.0,
      66.0, 81.0, 96.0,
      102.0, 126.0, 150.0
    ]);

    assert!((&A + &A).is_close(&expected_add, 1e-10));
    assert!((&A - &A).is_close(&expected_sub, 1e-10));
    assert!((2.0 * &A).is_close(&expected_scale, 1e-10));
    assert!((&A * 2.0).is_close(&expected_scale, 1e-10));
    assert!((&A * &b).is_close(&expected_dot_vec, 1e-10));
    assert!((&A * &A).is_close(&expected_dot, 1e-10));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Matrix4                                                                   //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy)]
pub struct Matrix4d {
  pub data: [f64; 16],
}

// Implementing Default trait for Matrix4d
impl Default for Matrix4d {
  fn default() -> Self {
    Self { data: [0.0; 16] }
  }
}

impl Matrix4d {
  pub fn new(data: [f64; 16]) -> Self {
    Self { data }
  }

  pub fn at(&self, row: usize, col: usize) -> &f64 {
    assert!(row < 4 && col < 4, "Index out of bounds");
    &self.data[row * 4 + col]
  }

  pub fn set(&mut self, row: usize, col: usize, value: f64) {
    assert!(row < 4 && col < 4, "Index out of bounds");
    self.data[row * 4 + col] = value;
  }

  fn eye() -> Self {
    let mut data = [0.0; 16];
    data[0] = 1.0;
    data[5] = 1.0;
    data[10] = 1.0;
    data[15] = 1.0;
    Self { data }
  }

  #[allow(non_snake_case)]
  fn transpose(&self) -> Self {
    let A = self.data;

    #[rustfmt::skip]
    let B = [
        A[0], A[4], A[8], A[12],
        A[1], A[5], A[9], A[13],
        A[2], A[6], A[10], A[14],
        A[3], A[7], A[11], A[15],
    ];

    Self { data: B }
  }

  fn add(&self, rhs: &Matrix4d) -> Self {
    let mut result = Matrix4d::default();
    for i in 0..16 {
      result.data[i] = self.data[i] + rhs.data[i];
    }
    result
  }

  fn sub(&self, rhs: &Matrix4d) -> Self {
    let mut result = Matrix4d::default();
    for i in 0..16 {
      result.data[i] = self.data[i] - rhs.data[i];
    }
    result
  }

  fn scale(&self, rhs: f64) -> Self {
    let mut result = Matrix4d::default();
    for i in 0..16 {
      result.data[i] = rhs * self.data[i];
    }
    result
  }

  #[allow(non_snake_case)]
  pub fn dot(&self, rhs: &Matrix4d) -> Self {
    let A = self.data;
    let B = rhs.data;
    let mut C = [0.0; 16];

    // Row 0
    C[0] = A[0] * B[0] + A[1] * B[4] + A[2] * B[8] + A[3] * B[12];
    C[1] = A[0] * B[1] + A[1] * B[5] + A[2] * B[9] + A[3] * B[13];
    C[2] = A[0] * B[2] + A[1] * B[6] + A[2] * B[10] + A[3] * B[14];
    C[3] = A[0] * B[3] + A[1] * B[7] + A[2] * B[11] + A[3] * B[15];

    // Row 1
    C[4] = A[4] * B[0] + A[5] * B[4] + A[6] * B[8] + A[7] * B[12];
    C[5] = A[4] * B[1] + A[5] * B[5] + A[6] * B[9] + A[7] * B[13];
    C[6] = A[4] * B[2] + A[5] * B[6] + A[6] * B[10] + A[7] * B[14];
    C[7] = A[4] * B[3] + A[5] * B[7] + A[6] * B[11] + A[7] * B[15];

    // Row 2
    C[8] = A[8] * B[0] + A[9] * B[4] + A[10] * B[8] + A[11] * B[12];
    C[9] = A[8] * B[1] + A[9] * B[5] + A[10] * B[9] + A[11] * B[13];
    C[10] = A[8] * B[2] + A[9] * B[6] + A[10] * B[10] + A[11] * B[14];
    C[11] = A[8] * B[3] + A[9] * B[7] + A[10] * B[11] + A[11] * B[15];

    // Row 3
    C[12] = A[12] * B[0] + A[13] * B[4] + A[14] * B[8] + A[15] * B[12];
    C[13] = A[12] * B[1] + A[13] * B[5] + A[14] * B[9] + A[15] * B[13];
    C[14] = A[12] * B[2] + A[13] * B[6] + A[14] * B[10] + A[15] * B[14];
    C[15] = A[12] * B[3] + A[13] * B[7] + A[14] * B[11] + A[15] * B[15];

    Self { data: C }
  }

  #[allow(non_snake_case)]
  pub fn dot_vec(&self, rhs: &Vector4d) -> Vector4d {
    let M = self.data;
    let v = rhs.data;
    let mut y = [0.0; 4];

    y[0] = M[0] * v[0] + M[1] * v[1] + M[2] * v[2] + M[3] * v[3];
    y[1] = M[4] * v[0] + M[5] * v[1] + M[6] * v[2] + M[7] * v[3];
    y[2] = M[8] * v[0] + M[9] * v[1] + M[10] * v[2] + M[11] * v[3];
    y[3] = M[12] * v[0] + M[13] * v[1] + M[14] * v[2] + M[15] * v[3];

    Vector4d { data: y }
  }

  pub fn is_close(&self, rhs: &Self, epsilon: f64) -> bool {
    self.data.iter().zip(rhs.data.iter()).all(|(&a, &b)| {
      let diff = if a > b { a - b } else { b - a };
      diff < epsilon
    })
  }
}

// -- Matrix4d + Matrix4d
impl Add<&Matrix4d> for &Matrix4d {
  type Output = Matrix4d;

  fn add(self, rhs: &Matrix4d) -> Self::Output {
    self.add(rhs)
  }
}

// -- Matrix3d * f64
impl Mul<f64> for &Matrix4d {
  type Output = Matrix4d;

  fn mul(self, rhs: f64) -> Self::Output {
    self.scale(rhs)
  }
}

// -- f64 * Matrix4d
impl Mul<&Matrix4d> for f64 {
  type Output = Matrix4d;

  fn mul(self, rhs: &Matrix4d) -> Self::Output {
    rhs * self
  }
}

// -- Matrix4d - Matrix4d
impl Sub<&Matrix4d> for &Matrix4d {
  type Output = Matrix4d;

  fn sub(self, rhs: &Matrix4d) -> Self::Output {
    self.sub(rhs)
  }
}

// -- Matrix4d * Vector4d
impl Mul<&Vector4d> for &Matrix4d {
  type Output = Vector4d;
  fn mul(self, rhs: &Vector4d) -> Self::Output {
    self.dot_vec(rhs)
  }
}

// -- Matrix4d * Matrix4d
impl Mul<&Matrix4d> for &Matrix4d {
  type Output = Matrix4d;
  fn mul(self, rhs: &Matrix4d) -> Self::Output {
    self.dot(rhs)
  }
}

// --- Neg
impl Neg for Matrix4d {
  type Output = Matrix4d;

  #[allow(non_snake_case)]
  fn neg(self) -> Self {
    let mut M = [0.0; 16];

    M[0] = -self.data[0];
    M[1] = -self.data[1];
    M[2] = -self.data[2];
    M[3] = -self.data[3];
    M[4] = -self.data[4];
    M[5] = -self.data[5];
    M[6] = -self.data[6];
    M[7] = -self.data[7];
    M[8] = -self.data[8];
    M[9] = -self.data[9];
    M[10] = -self.data[10];
    M[11] = -self.data[11];
    M[12] = -self.data[12];
    M[13] = -self.data[13];
    M[14] = -self.data[14];
    M[15] = -self.data[15];

    Self { data: M }
  }
}

#[cfg(test)]
mod matrix4_tests {
  // Note this useful idiom: importing names from outer (for mod tests) scope.
  use super::*;

  #[test]
  #[allow(non_snake_case)]
  fn test_matrix4_new() {
    // Create and use a matrix
    #[rustfmt::skip]
    let A = Matrix4d::new([
      1.0, 2.0, 3.0, 4.0,
      5.0, 6.0, 7.0, 8.0,
      9.0, 10.0, 11.0, 12.0,
      13.0, 14.0, 15.0, 16.0,
    ]);
    let b = Vector4d::new([1.0, 2.0, 3.0, 4.0]);

    #[rustfmt::skip]
    let expected_add = Matrix4d::new([
      2.0, 4.0, 6.0, 8.0,
      10.0, 12.0, 14.0, 16.0,
      18.0, 20.0, 22.0, 24.0,
      26.0, 28.0, 30.0, 32.0,
    ]);

    #[rustfmt::skip]
    let expected_sub = Matrix4d::new([
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0,
    ]);

    #[rustfmt::skip]
    let expected_scale = Matrix4d::new([
      2.0, 4.0, 6.0, 8.0,
      10.0, 12.0, 14.0, 16.0,
      18.0, 20.0, 22.0, 24.0,
      26.0, 28.0, 30.0, 32.0,
    ]);

    #[rustfmt::skip]
    let expected_dot_vec = Vector4d::new([30.0, 70.0, 110.0, 150.0]);

    #[rustfmt::skip]
    let expected_dot = Matrix4d::new([
      90.0, 100.0, 110.0, 120.0,
      202.0, 228.0, 254.0, 280.0,
      314.0, 356.0, 398.0, 440.0,
      426.0, 484.0, 542.0, 600.0,
    ]);

    assert!((&A + &A).is_close(&expected_add, 1e-10));
    assert!((&A - &A).is_close(&expected_sub, 1e-10));
    assert!((2.0 * &A).is_close(&expected_scale, 1e-10));
    assert!((&A * 2.0).is_close(&expected_scale, 1e-10));
    assert!((&A * &b).is_close(&expected_dot_vec, 1e-10));
    assert!((&A * &A).is_close(&expected_dot, 1e-10));
  }
}

///////////////////////////////////////////////////////////////////////////////
// Matrix<T>                                                                 //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct Matrix<T> {
  pub rows: usize,
  pub cols: usize,
  pub data: Vec<T>,
}

// Matrix<T>::new(rows, cols, data)
impl<T> Matrix<T> {
  pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
    assert!(data.len() == rows * cols);
    Self { rows, cols, data }
  }
}

// Matrix<T> formatter
impl<T: fmt::Display> fmt::Display for Matrix<T> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for i in 0..self.rows {
      write!(f, "[ ")?;
      for j in 0..self.cols {
        write!(f, "{:>6} ", self.data[i * self.cols + j])?;
      }
      writeln!(f, "]")?;
    }
    Ok(())
  }
}

// Matrix<T> add
impl<T> Add<&Matrix<T>> for &Matrix<T>
where
  T: Add<Output = T> + Copy,
{
  type Output = Matrix<T>;

  fn add(self, rhs: &Matrix<T>) -> Self::Output {
    debug_assert!(self.rows == rhs.rows);
    debug_assert!(self.cols == rhs.cols);

    let result_data = self
      .data
      .iter()
      .zip(rhs.data.iter())
      .map(|(&a, &b)| a + b)
      .collect();

    Matrix::new(self.rows, self.cols, result_data)
  }
}

// Matrix<T> dot product
impl<T> Mul<&Matrix<T>> for &Matrix<T>
where
  T: Default + Copy + Add<Output = T> + Mul<Output = T> + AddAssign,
{
  type Output = Matrix<T>;

  fn mul(self, rhs: &Matrix<T>) -> Self::Output {
    debug_assert!(self.rows == rhs.cols);
    debug_assert!(self.cols == rhs.rows);

    let mut result_data = vec![T::default(); self.rows * rhs.cols];
    for i in 0..self.rows {
      for j in 0..rhs.cols {
        let mut sum = T::default();
        for k in 0..self.cols {
          sum += self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j];
        }
        result_data[i * rhs.cols + j] = sum;
      }
    }

    Matrix::new(self.rows, rhs.cols, result_data)
  }
}

// Matrix<T>.is_close()
impl<T> Matrix<T>
where
  T: Sub<Output = T> + PartialOrd + Copy,
{
  pub fn is_close(&self, rhs: &Self, epsilon: T) -> bool {
    if self.rows != rhs.rows || self.cols != rhs.cols {
      return false;
    }

    self.data.iter().zip(rhs.data.iter()).all(|(&a, &b)| {
      let diff = if a > b { a - b } else { b - a };
      diff < epsilon
    })
  }
}

// Matrix<T>::block(row, col, nrows, ncols)
// Matrix<T>::set_block(row, col, Matrix<T>)
impl<T: Clone> Matrix<T> {
  pub fn block(
    &self,
    row: usize,
    col: usize,
    nrows: usize,
    ncols: usize,
  ) -> Matrix<T> {
    assert!(row + nrows <= self.rows);
    assert!(col + ncols <= self.cols);

    let mut data = Vec::with_capacity(nrows * ncols);
    for r in row..row + nrows {
      for c in col..col + ncols {
        data.push(self.data[r * self.cols + c].clone());
      }
    }

    Matrix {
      rows: nrows,
      cols: ncols,
      data,
    }
  }

  pub fn set_block(&mut self, row: usize, col: usize, other: &Matrix<T>) {
    assert!(row + other.rows <= self.rows);
    assert!(col + other.cols <= self.cols);

    for r in 0..other.rows {
      for c in 0..other.cols {
        self.data[(row + r) * self.cols + (col + c)] =
          other.data[r * other.cols + c].clone();
      }
    }
  }
}

// Convenient short hands for accessing matrix sub-blocks
//
// Matrix<T>.top_left_corner(rows, cols)
// Matrix<T>.top_right_corner(rows, cols)
// Matrix<T>.bottom_left_corner(rows, cols)
// Matrix<T>.bottom_right_corner(rows, cols)
// Matrix<T>.row(i)
// Matrix<T>.col(i)
//
impl<T: Clone> Matrix<T> {
  pub fn top_left_corner(&self, nrows: usize, ncols: usize) -> Matrix<T> {
    self.block(0, 0, nrows, ncols)
  }

  pub fn top_right_corner(&self, nrows: usize, ncols: usize) -> Matrix<T> {
    self.block(0, self.cols - ncols, nrows, ncols)
  }

  pub fn bottom_left_corner(&self, nrows: usize, ncols: usize) -> Matrix<T> {
    self.block(self.rows - nrows, 0, nrows, ncols)
  }

  pub fn bottom_right_corner(&self, nrows: usize, ncols: usize) -> Matrix<T> {
    self.block(self.rows - nrows, self.cols - ncols, nrows, ncols)
  }

  pub fn row(&self, i: usize) -> Matrix<T> {
    self.block(i, 0, 1, self.cols)
  }

  pub fn col(&self, j: usize) -> Matrix<T> {
    self.block(0, j, self.rows, 1)
  }
}

// Matrix<T>::triu()
// Matrix<T>::tril()
// Matrix<T>::set_triu(Matrix<T>)
// Matrix<T>::set_tril(Matrix<T>)
impl<T: Clone + Default> Matrix<T> {
  pub fn triu(&self) -> Matrix<T> {
    let mut data = vec![T::default(); self.rows * self.cols];
    for r in 0..self.rows {
      for c in r..self.cols {
        // c >= r
        data[r * self.cols + c] = self.data[r * self.cols + c].clone();
      }
    }
    Matrix {
      rows: self.rows,
      cols: self.cols,
      data,
    }
  }

  pub fn tril(&self) -> Matrix<T> {
    let mut data = vec![T::default(); self.rows * self.cols];
    for r in 0..self.rows {
      for c in 0..=r {
        // c <= r
        data[r * self.cols + c] = self.data[r * self.cols + c].clone();
      }
    }
    Matrix {
      rows: self.rows,
      cols: self.cols,
      data,
    }
  }

  pub fn set_triu(&mut self, other: &Matrix<T>) {
    assert_eq!(self.rows, other.rows);
    assert_eq!(self.cols, other.cols);
    for r in 0..self.rows {
      for c in r..self.cols {
        self.data[r * self.cols + c] = other.data[r * self.cols + c].clone();
      }
    }
  }

  pub fn set_tril(&mut self, other: &Matrix<T>) {
    assert_eq!(self.rows, other.rows);
    assert_eq!(self.cols, other.cols);
    for r in 0..self.rows {
      for c in 0..=r {
        self.data[r * self.cols + c] = other.data[r * self.cols + c].clone();
      }
    }
  }
}

// Matrix::eye(num_rows, num_cols)
impl<T> Matrix<T>
where
  T: Default + Clone + From<u8>,
{
  pub fn eye(num_rows: usize, num_cols: usize) -> Self {
    let mut data = vec![T::default(); num_rows * num_cols];
    for i in 0..num_rows {
      for j in 0..num_cols {
        if i == j {
          data[i * num_cols + j] = T::from(1u8);
        } else {
          data[i * num_cols + j] = T::from(0u8);
        }
      }
    }

    Matrix {
      rows: num_rows,
      cols: num_cols,
      data,
    }
  }
}

// Matrix::ones(num_rows, num_cols)
impl<T> Matrix<T>
where
  T: Default + Clone + From<u8>,
{
  pub fn ones(num_rows: usize, num_cols: usize) -> Self {
    let mut data = vec![T::default(); num_rows * num_cols];
    for i in 0..num_rows {
      for j in 0..num_cols {
        data[i * num_cols + j] = T::from(1u8);
      }
    }

    Matrix {
      rows: num_rows,
      cols: num_cols,
      data,
    }
  }
}

// Matrix::zeros(num_rows, num_cols)
impl<T> Matrix<T>
where
  T: Default + Clone + From<u8>,
{
  pub fn zeros(num_rows: usize, num_cols: usize) -> Self {
    let mut data = vec![T::default(); num_rows * num_cols];
    for i in 0..num_rows {
      for j in 0..num_cols {
        data[i * num_cols + j] = T::from(0u8);
      }
    }

    Matrix {
      rows: num_rows,
      cols: num_cols,
      data,
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// BlockView<T>                                                              //
///////////////////////////////////////////////////////////////////////////////

pub struct BlockView<'a, T> {
  matrix: &'a Matrix<T>,
  row_offset: usize,
  col_offset: usize,
  rows: usize,
  cols: usize,
}

pub struct BlockViewMut<'a, T> {
  matrix: &'a mut Matrix<T>,
  row_offset: usize,
  col_offset: usize,
  rows: usize,
  cols: usize,
}

impl<'a, T> BlockView<'a, T> {
  pub fn get(&self, r: usize, c: usize) -> &T {
    assert!(r < self.rows && c < self.cols);
    &self.matrix.data
      [(self.row_offset + r) * self.matrix.cols + self.col_offset + c]
  }
}

impl<'a, T> BlockViewMut<'a, T> {
  pub fn get_mut(&mut self, r: usize, c: usize) -> &mut T {
    assert!(r < self.rows && c < self.cols);
    let cols = self.matrix.cols;
    &mut self.matrix.data[(self.row_offset + r) * cols + self.col_offset + c]
  }

  pub fn set(&mut self, r: usize, c: usize, val: T) {
    *self.get_mut(r, c) = val;
  }
}

impl<T> Matrix<T> {
  pub fn block_view(
    &self,
    row: usize,
    col: usize,
    nrows: usize,
    ncols: usize,
  ) -> BlockView<'_, T> {
    assert!(row + nrows <= self.rows && col + ncols <= self.cols);
    BlockView {
      matrix: self,
      row_offset: row,
      col_offset: col,
      rows: nrows,
      cols: ncols,
    }
  }

  pub fn block_view_mut(
    &mut self,
    row: usize,
    col: usize,
    nrows: usize,
    ncols: usize,
  ) -> BlockViewMut<'_, T> {
    assert!(row + nrows <= self.rows && col + ncols <= self.cols);
    BlockViewMut {
      matrix: self,
      row_offset: row,
      col_offset: col,
      rows: nrows,
      cols: ncols,
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// TriView<T>                                                                //
///////////////////////////////////////////////////////////////////////////////

pub struct TriView<'a, T> {
  matrix: &'a Matrix<T>,
  upper: bool,
}

pub struct TriViewMut<'a, T> {
  matrix: &'a mut Matrix<T>,
  upper: bool,
}

impl<'a, T: Default + Clone> TriView<'a, T> {
  pub fn get(&self, r: usize, c: usize) -> T {
    let in_triangle = if self.upper { c >= r } else { r >= c };
    if in_triangle {
      self.matrix.data[r * self.matrix.cols + c].clone()
    } else {
      T::default()
    }
  }
}

impl<'a, T: Default + Clone> TriViewMut<'a, T> {
  pub fn set(&mut self, r: usize, c: usize, val: T) {
    let in_triangle = if self.upper { c >= r } else { r >= c };
    if in_triangle {
      self.matrix.data[r * self.matrix.cols + c] = val;
    }
  }

  pub fn get(&self, r: usize, c: usize) -> T {
    let in_triangle = if self.upper { c >= r } else { r >= c };
    if in_triangle {
      self.matrix.data[r * self.matrix.cols + c].clone()
    } else {
      T::default()
    }
  }
}

impl<T> Matrix<T> {
  pub fn upper_tri_view(&self) -> TriView<'_, T> {
    TriView {
      matrix: self,
      upper: true,
    }
  }

  pub fn lower_tri_view(&self) -> TriView<'_, T> {
    TriView {
      matrix: self,
      upper: false,
    }
  }

  pub fn upper_tri_view_mut(&mut self) -> TriViewMut<'_, T> {
    TriViewMut {
      matrix: self,
      upper: true,
    }
  }

  pub fn lower_tri_view_mut(&mut self) -> TriViewMut<'_, T> {
    TriViewMut {
      matrix: self,
      upper: false,
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// UNITTESTS                                                                 //
///////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
mod tests {
  // Note this useful idiom: importing names from outer (for mod tests) scope.
  use super::*;

  #[test]
  #[allow(non_snake_case)]
  fn test_matrix_eye() {
    let A: Matrix<f64> = Matrix::eye(2, 2);
    let expected = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
    assert!(A.is_close(&expected, 1e-10));
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_matrix_zeros() {
    let A: Matrix<f64> = Matrix::zeros(2, 2);
    let expected = Matrix::new(2, 2, vec![0.0, 0.0, 0.0, 0.0]);
    assert!(A.is_close(&expected, 1e-10));
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_matrix_ones() {
    let A: Matrix<f64> = Matrix::ones(2, 2);
    let expected = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
    assert!(A.is_close(&expected, 1e-10));
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_matrix_add() {
    let A = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let B = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
    let expected = Matrix::new(2, 2, vec![6.0, 8.0, 10.0, 12.0]);
    let sum = &A + &B;
    assert!(sum.is_close(&expected, 1e-10));
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_matrix_block() {
    #[rustfmt::skip]
    let A = Matrix::new(4, 4, vec![
      1.0, 1.0, 2.0, 2.0,
      1.0, 1.0, 2.0, 2.0,
      3.0, 3.0, 4.0, 4.0,
      3.0, 3.0, 4.0, 4.0,
    ]);

    let expected1 = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
    let expected2 = Matrix::new(2, 2, vec![2.0, 2.0, 2.0, 2.0]);
    let expected3 = Matrix::new(2, 2, vec![3.0, 3.0, 3.0, 3.0]);
    let expected4 = Matrix::new(2, 2, vec![4.0, 4.0, 4.0, 4.0]);

    assert!(A.block(0, 0, 2, 2).is_close(&expected1, 1e-10));
    assert!(A.block(0, 2, 2, 2).is_close(&expected2, 1e-10));
    assert!(A.block(2, 0, 2, 2).is_close(&expected3, 1e-10));
    assert!(A.block(2, 2, 2, 2).is_close(&expected4, 1e-10));
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_matrix_set_block() {
    let mut result: Matrix<f64> = Matrix::zeros(4, 4);
    let A = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
    let B = Matrix::new(2, 2, vec![2.0, 2.0, 2.0, 2.0]);
    let C = Matrix::new(2, 2, vec![3.0, 3.0, 3.0, 3.0]);
    let D = Matrix::new(2, 2, vec![4.0, 4.0, 4.0, 4.0]);

    result.set_block(0, 0, &A);
    result.set_block(0, 2, &B);
    result.set_block(2, 0, &C);
    result.set_block(2, 2, &D);

    #[rustfmt::skip]
    let expected = Matrix::new(4, 4, vec![
      1.0, 1.0, 2.0, 2.0,
      1.0, 1.0, 2.0, 2.0,
      3.0, 3.0, 4.0, 4.0,
      3.0, 3.0, 4.0, 4.0,
    ]);

    assert!(result.is_close(&expected, 1e-10));
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_matrix_convenience() {
    #[rustfmt::skip]
    let A = Matrix::new(4, 4, vec![
      1.0, 1.0, 2.0, 2.0,
      1.0, 1.0, 2.0, 2.0,
      3.0, 3.0, 4.0, 4.0,
      3.0, 3.0, 4.0, 4.0,
    ]);

    let expected_tl = Matrix::new(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
    let expected_tr = Matrix::new(2, 2, vec![2.0, 2.0, 2.0, 2.0]);
    let expected_bl = Matrix::new(2, 2, vec![3.0, 3.0, 3.0, 3.0]);
    let expected_br = Matrix::new(2, 2, vec![4.0, 4.0, 4.0, 4.0]);
    assert!(A.top_left_corner(2, 2).is_close(&expected_tl, 1e-10));
    assert!(A.top_right_corner(2, 2).is_close(&expected_tr, 1e-10));
    assert!(A.bottom_left_corner(2, 2).is_close(&expected_bl, 1e-10));
    assert!(A.bottom_right_corner(2, 2).is_close(&expected_br, 1e-10));

    let expected_row = Matrix::new(1, 4, vec![1.0, 1.0, 2.0, 2.0]);
    let expected_col = Matrix::new(4, 1, vec![2.0, 2.0, 4.0, 4.0]);
    assert!(A.row(0).is_close(&expected_row, 1e-10));
    assert!(A.col(3).is_close(&expected_col, 1e-10));
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_matrix_mul() {
    let A = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let B = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
    let expected = Matrix::new(2, 2, vec![19.0, 22.0, 43.0, 50.0]);
    let product = &A * &B;
    assert!(product.is_close(&expected, 1e-10));
  }
}
