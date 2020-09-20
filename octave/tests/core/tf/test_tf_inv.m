addpath(genpath("proto"));

C = [1.0, 0.0, 0.0;
     0.0, 1.0, 0.0;
     0.0, 0.0, 1.0];
r = [1.0; 2.0; 3.0];
T = tf(C, r);
T_inv = tf_inv(T);
assert(isequal(T_inv, inv(T)) == 1);
