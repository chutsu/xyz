use std::fmt;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::Sub;

///////////////////////////////////////////////////////////////////////////////
// Matrix<T>                                                                 //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub struct Matrix<T> {
  pub rows: usize,
  pub cols: usize,
  pub data: Vec<T>,
}

// Matrix<T>
impl<T> Matrix<T> {
  pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Self {
    assert!(data.len() == rows * cols);
    Self { rows, cols, data }
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

impl<T: Clone> Matrix<T> {
  /// Get a sub-matrix block starting at (row, col) with given size.
  /// Mimics Eigen's m.block(i, j, rows, cols)
  pub fn get_block(
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

  /// Set a sub-matrix block starting at (row, col).
  /// Mimics Eigen's m.block(i, j, rows, cols) = other
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
// Matrix<T>.top_rows(n)
// Matrix<T>.bottom_rows(n)
// Matrix<T>.left_cols(n)
// Matrix<T>.right_cols(n)
//
impl<T: Clone> Matrix<T> {
  pub fn top_left_corner(&self, nrows: usize, ncols: usize) -> Matrix<T> {
    self.get_block(0, 0, nrows, ncols)
  }

  pub fn top_right_corner(&self, nrows: usize, ncols: usize) -> Matrix<T> {
    self.get_block(0, self.cols - ncols, nrows, ncols)
  }

  pub fn bottom_left_corner(&self, nrows: usize, ncols: usize) -> Matrix<T> {
    self.get_block(self.rows - nrows, 0, nrows, ncols)
  }

  pub fn bottom_right_corner(&self, nrows: usize, ncols: usize) -> Matrix<T> {
    self.get_block(self.rows - nrows, self.cols - ncols, nrows, ncols)
  }

  pub fn row(&self, i: usize) -> Matrix<T> {
    self.get_block(i, 0, 1, self.cols)
  }

  pub fn col(&self, j: usize) -> Matrix<T> {
    self.get_block(0, j, self.rows, 1)
  }

  pub fn top_rows(&self, n: usize) -> Matrix<T> {
    self.get_block(0, 0, n, self.cols)
  }

  pub fn bottom_rows(&self, n: usize) -> Matrix<T> {
    self.get_block(self.rows - n, 0, n, self.cols)
  }

  pub fn left_cols(&self, n: usize) -> Matrix<T> {
    self.get_block(0, 0, self.rows, n)
  }

  pub fn right_cols(&self, n: usize) -> Matrix<T> {
    self.get_block(0, self.cols - n, self.rows, n)
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
  fn test_matrix_get_block() {
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

    assert!(A.get_block(0, 0, 2, 2).is_close(&expected1, 1e-10));
    assert!(A.get_block(0, 2, 2, 2).is_close(&expected2, 1e-10));
    assert!(A.get_block(2, 0, 2, 2).is_close(&expected3, 1e-10));
    assert!(A.get_block(2, 2, 2, 2).is_close(&expected4, 1e-10));
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
  fn test_matrix_mul() {
    let A = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let B = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
    let expected = Matrix::new(2, 2, vec![19.0, 22.0, 43.0, 50.0]);
    let product = &A * &B;
    assert!(product.is_close(&expected, 1e-10));
  }
}
