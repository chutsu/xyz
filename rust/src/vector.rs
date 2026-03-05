use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

///////////////////////////////////////////////////////////////////////////////
// Vector3                                                                   //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy)]
pub struct Vector3d {
  pub data: [f64; 3],
}

// Implementing Default trait for Vector3d
impl Default for Vector3d {
  fn default() -> Self {
    Vector3d { data: [0.0; 3] }
  }
}

impl Vector3d {
  pub fn new(data: [f64; 3]) -> Self {
    Vector3d { data }
  }

  pub fn x(&self) -> f64 {
    self.data[0]
  }

  pub fn y(&self) -> f64 {
    self.data[1]
  }

  pub fn z(&self) -> f64 {
    self.data[2]
  }

  pub fn at(&self, i: usize) -> &f64 {
    assert!(i < 3, "Index out of bounds");
    &self.data[i]
  }

  pub fn set(&mut self, i: usize, value: f64) {
    assert!(i < 3, "Index out of bounds");
    self.data[i] = value;
  }

  pub fn add(&self, rhs: &Vector3d) -> Vector3d {
    let mut result = Vector3d::default();
    result.data[0] = self.data[0] + rhs.data[0];
    result.data[1] = self.data[1] + rhs.data[1];
    result.data[2] = self.data[2] + rhs.data[2];
    result
  }

  pub fn sub(&self, rhs: &Vector3d) -> Vector3d {
    let mut result = Vector3d::default();
    result.data[0] = self.data[0] - rhs.data[0];
    result.data[1] = self.data[1] - rhs.data[1];
    result.data[2] = self.data[2] - rhs.data[2];
    result
  }

  fn scale(&self, rhs: f64) -> Vector3d {
    let mut res = [0.0; 3];
    res[0] = rhs * self.data[0];
    res[1] = rhs * self.data[1];
    res[2] = rhs * self.data[2];
    Vector3d::new(res)
  }

  fn dot(&self, rhs: &Vector3d) -> f64 {
    let mut res = 0.0;
    res += rhs.data[0] * self.data[0];
    res += rhs.data[1] * self.data[1];
    res += rhs.data[2] * self.data[2];
    res
  }

  fn cross(&self, rhs: &Vector3d) -> Vector3d {
    let data = [
      self.y() * rhs.z() - self.z() * rhs.y(),
      self.z() * rhs.x() - self.x() * rhs.z(),
      self.x() * rhs.y() - self.y() * rhs.x(),
    ];
    Vector3d { data }
  }

  fn norm(&self) -> f64 {
    self.dot(self).sqrt()
  }

  fn normalize(&self) -> Vector3d {
    *self * (1.0 / self.norm())
  }

  pub fn is_close(&self, rhs: &Self, epsilon: f64) -> bool {
    self.data.iter().zip(rhs.data.iter()).all(|(&a, &b)| {
      let diff = if a > b { a - b } else { b - a };
      diff < epsilon
    })
  }
}

// --- Vector3d + Vector3d
impl Add<&Vector3d> for &Vector3d {
  type Output = Vector3d;
  fn add(self, rhs: &Vector3d) -> Self::Output {
    self.add(rhs)
  }
}

// --- Vector3d - Vector3d
impl Sub<&Vector3d> for &Vector3d {
  type Output = Vector3d;
  fn sub(self, rhs: &Vector3d) -> Self::Output {
    self.sub(rhs)
  }
}

// --- Vector3d * f64
impl Mul<f64> for Vector3d {
  type Output = Vector3d;
  fn mul(self, rhs: f64) -> Vector3d {
    self.scale(rhs)
  }
}

// --- f64 * Vector3d
impl Mul<Vector3d> for f64 {
  type Output = Vector3d;
  fn mul(self, rhs: Vector3d) -> Vector3d {
    rhs * self
  }
}

// --- Vector3d * Vector3d
impl Mul<&Vector3d> for &Vector3d {
  type Output = f64;

  fn mul(self, rhs: &Vector3d) -> f64 {
    self.dot(rhs)
  }
}

// --- Neg ---
impl Neg for Vector3d {
  type Output = Vector3d;
  fn neg(self) -> Vector3d {
    let data = [-self.x(), -self.y(), -self.z()];
    Vector3d::new(data)
  }
}

// Formatter
impl std::fmt::Display for Vector3d {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "({:.2}, {:.2}, {:.2})", self.x(), self.y(), self.z())
  }
}

///////////////////////////////////////////////////////////////////////////////
// Vector4                                                                   //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy)]
pub struct Vector4d {
  pub data: [f64; 4],
}

// Implementing Default trait for Vector4d
impl Default for Vector4d {
  fn default() -> Self {
    Vector4d { data: [0.0; 4] }
  }
}

impl Vector4d {
  pub fn new(data: [f64; 4]) -> Self {
    Vector4d { data }
  }

  pub fn x(&self) -> f64 {
    self.data[0]
  }

  pub fn y(&self) -> f64 {
    self.data[1]
  }

  pub fn z(&self) -> f64 {
    self.data[2]
  }

  pub fn w(&self) -> f64 {
    self.data[3]
  }

  pub fn at(&self, i: usize) -> &f64 {
    assert!(i < 4, "Index out of bounds");
    &self.data[i]
  }

  pub fn set(&mut self, i: usize, value: f64) {
    assert!(i < 4, "Index out of bounds");
    self.data[i] = value;
  }

  pub fn add(&self, rhs: &Vector4d) -> Vector4d {
    let mut result = Vector4d::default();
    result.data[0] = self.data[0] + rhs.data[0];
    result.data[1] = self.data[1] + rhs.data[1];
    result.data[2] = self.data[2] + rhs.data[2];
    result.data[3] = self.data[3] + rhs.data[3];
    result
  }

  pub fn sub(&self, rhs: &Vector4d) -> Vector4d {
    let mut result = Vector4d::default();
    result.data[0] = self.data[0] - rhs.data[0];
    result.data[1] = self.data[1] - rhs.data[1];
    result.data[2] = self.data[2] - rhs.data[2];
    result.data[3] = self.data[3] - rhs.data[3];
    result
  }

  fn scale(&self, rhs: f64) -> Vector4d {
    let mut res = [0.0; 4];
    res[0] = rhs * self.data[0];
    res[1] = rhs * self.data[1];
    res[2] = rhs * self.data[2];
    res[3] = rhs * self.data[3];
    Vector4d::new(res)
  }

  fn dot(&self, rhs: &Vector4d) -> f64 {
    let mut res = 0.0;
    res += rhs.data[0] * self.data[0];
    res += rhs.data[1] * self.data[1];
    res += rhs.data[2] * self.data[2];
    res += rhs.data[3] * self.data[3];
    res
  }

  fn norm(&self) -> f64 {
    self.dot(self).sqrt()
  }

  fn normalize(&self) -> Vector4d {
    *self * (1.0 / self.norm())
  }

  pub fn is_close(&self, rhs: &Self, epsilon: f64) -> bool {
    self.data.iter().zip(rhs.data.iter()).all(|(&a, &b)| {
      let diff = if a > b { a - b } else { b - a };
      diff < epsilon
    })
  }
}

// --- Vector4d + Vector4d
impl Add<&Vector4d> for &Vector4d {
  type Output = Vector4d;

  fn add(self, rhs: &Vector4d) -> Self::Output {
    self.add(rhs)
  }
}

// --- Vector4d - Vector4d
impl Sub<&Vector4d> for &Vector4d {
  type Output = Vector4d;

  fn sub(self, rhs: &Vector4d) -> Self::Output {
    self.sub(rhs)
  }
}

// --- Vector4d * f64
impl Mul<f64> for Vector4d {
  type Output = Vector4d;

  fn mul(self, rhs: f64) -> Vector4d {
    self.scale(rhs)
  }
}

// --- f64 * Vector4d
impl Mul<Vector4d> for f64 {
  type Output = Vector4d;

  fn mul(self, rhs: Vector4d) -> Vector4d {
    rhs * self
  }
}

// --- Vector4d * Vector4d
impl Mul<&Vector4d> for &Vector4d {
  type Output = f64;

  fn mul(self, rhs: &Vector4d) -> f64 {
    self.dot(rhs)
  }
}

// --- Neg
impl Neg for Vector4d {
  type Output = Vector4d;
  fn neg(self) -> Vector4d {
    let data = [-self.x(), -self.y(), -self.z(), -self.w()];
    Vector4d::new(data)
  }
}

// Formatter
impl std::fmt::Display for Vector4d {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(
      f,
      "({:.2}, {:.2}, {:.2} {:.2})",
      self.x(),
      self.y(),
      self.z(),
      self.w()
    )
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
  fn test_vector3d() {
    let a = Vector3d::new([1.0, 2.0, 3.0]);
    let b = Vector3d::new([4.0, 5.0, 6.0]);
    let expected_add = Vector3d::new([5.0, 7.0, 9.0]);
    let expected_sub = Vector3d::new([-3.0, -3.0, -3.0]);
    let expected_dot_scalar = Vector3d::new([2.0, 4.0, 6.0]);
    let expected_cross = Vector3d::new([-3.0, 6.0, -3.0]);
    let expected_neg = Vector3d::new([-1.0, -2.0, -3.0]);
    let expected_normalize = Vector3d::new([0.27, 0.53, 0.80]);

    assert!(a.x() == 1.0);
    assert!(a.y() == 2.0);
    assert!(a.z() == 3.0);

    assert!(b.x() == 4.0);
    assert!(b.y() == 5.0);
    assert!(b.z() == 6.0);

    assert!(*a.at(0) == 1.0);
    assert!(*a.at(1) == 2.0);
    assert!(*a.at(2) == 3.0);

    assert!(a.add(&b).is_close(&expected_add, 1e-10));
    assert!(a.sub(&b).is_close(&expected_sub, 1e-10));
    assert!(a.dot(&b) == 32.0);
    assert!((a * 2.0).is_close(&expected_dot_scalar, 1e-10));
    assert!((2.0 * a).is_close(&expected_dot_scalar, 1e-10));
    assert!((&a * &b) == 32.0);
    assert!((-a).is_close(&expected_neg, 1e-10));
    assert!(a.cross(&b).is_close(&expected_cross, 1e-10));
    assert!(a.norm() == 3.7416573867739413);
    assert!(a.normalize().is_close(&expected_normalize, 1e-2));
  }

  #[test]
  #[allow(non_snake_case)]
  fn test_vector4d() {
    let a = Vector4d::new([1.0, 2.0, 3.0, 4.0]);
    let b = Vector4d::new([5.0, 6.0, 7.0, 8.0]);
    let expected_add = Vector4d::new([6.0, 8.0, 10.0, 12.0]);
    let expected_sub = Vector4d::new([-4.0, -4.0, -4.0, -4.0]);
    let expected_dot_scalar = Vector4d::new([2.0, 4.0, 6.0, 8.0]);
    let expected_neg = Vector4d::new([-1.0, -2.0, -3.0, -4.0]);
    // let expected_normalize = Vector4d::new([0.27, 0.53, 0.80]);

    assert!(a.x() == 1.0);
    assert!(a.y() == 2.0);
    assert!(a.z() == 3.0);
    assert!(a.w() == 4.0);

    assert!(b.x() == 5.0);
    assert!(b.y() == 6.0);
    assert!(b.z() == 7.0);
    assert!(b.w() == 8.0);

    assert!(*a.at(0) == 1.0);
    assert!(*a.at(1) == 2.0);
    assert!(*a.at(2) == 3.0);
    assert!(*a.at(3) == 4.0);

    assert!(a.add(&b).is_close(&expected_add, 1e-10));
    assert!(a.sub(&b).is_close(&expected_sub, 1e-10));
    assert!(a.dot(&b) == 70.0);
    assert!((a * 2.0).is_close(&expected_dot_scalar, 1e-10));
    assert!((2.0 * a).is_close(&expected_dot_scalar, 1e-10));
    assert!((&a * &b) == 70.0);
    assert!((-a).is_close(&expected_neg, 1e-10));
    // assert!(a.norm() == 3.7416573867739413);
    // assert!(a.normalize().is_close(&expected_normalize, 1e-2));
  }
}
