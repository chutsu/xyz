function q = axisangle2quat(axis, angle)
  ax = axis(1);
  ay = axis(2);
  az = axis(3);

  qw = cos(angle / 2.0);
  qx = ax * sin(angle / 2.0);
  qy = ay * sin(angle / 2.0);
  qz = az * sin(angle / 2.0);

  q = [qw; qx; qy; qz];
endfunction
