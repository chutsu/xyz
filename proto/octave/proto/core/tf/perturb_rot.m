function [retval] = perturb_rot(T, step_size, i)
  assert(size(T) == [4, 4] || size(T) == [3, 3]);
  rvec = eye(3) * step_size;

  if (size(T) == [3, 3])
    q_diff = quat_mul(rot2quat(T), quat_delta(rvec(1:3, i)));
    q_diff = quat_normalize(q_diff);
    C_diff = quat2rot(q_diff);
    retval = C_diff;

  elseif (size(T) == [4, 4])
    C = tf_rot(T);
    r = tf_trans(T);

    % Note: using right-multiply
    q_diff = quat_mul(tf_quat(T), quat_delta(rvec(1:3, i)));
    q_diff = quat_normalize(q_diff);
    C_diff = quat2rot(q_diff);

    retval = tf(C_diff, r);
  endif
endfunction