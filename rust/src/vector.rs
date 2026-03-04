use std::ops::Add;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;

///////////////////////////////////////////////////////////////////////////////
// Vector3                                                                   //
///////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy)]
pub struct Vector3d {
  data: [f64; 3],
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

  fn is_close(&self, rhs: &Vector3d, epsilon: f64) -> bool {
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
    let mut result = Vector3d::default();
    result.data[0] = self.data[0] + rhs.data[0];
    result.data[1] = self.data[1] + rhs.data[1];
    result.data[2] = self.data[2] + rhs.data[2];
    result
  }
}

// --- Vector3d - Vector3d
impl Sub<&Vector3d> for &Vector3d {
  type Output = Vector3d;

  fn sub(self, rhs: &Vector3d) -> Self::Output {
    let mut result = Vector3d::default();
    result.data[0] = self.data[0] - rhs.data[0];
    result.data[1] = self.data[1] - rhs.data[1];
    result.data[2] = self.data[2] - rhs.data[2];
    result
  }
}

// --- Vector3d * f64
impl Mul<f64> for Vector3d {
  type Output = Vector3d;

  fn mul(self, rhs: f64) -> Vector3d {
    let mut res = [0.0; 3];
    res[0] = rhs * self.data[0];
    res[1] = rhs * self.data[1];
    res[2] = rhs * self.data[2];
    Vector3d::new(res)
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
}
