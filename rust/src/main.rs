use xyz_rust::Matrix;

fn main() {
  let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
  let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
  let m3 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
  let m4 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
  let sum = &m1 + &m2;
  let mul = &m3 * &m4;

  println!("{}", sum);
  println!("{}", mul);
}
