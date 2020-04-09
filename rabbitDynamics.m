syms x y r q1r q2r q1l q2l real 
syms dx dy dr dq1r dq2r dq1l dq2l real 
syms L1 L2 L3 % torso length, thigh length, shin length

toe_r = [x + L2 * sin(r+q1r) + L3 * sin(r+q1r+q2r) ;
         y + L2 * cos(r+q1r) + L3 * cos(r+q1r+q2r) ];
     
toe_l = [x + L2 * sin(r+q1l) + L3 * sin(r+q1l+q2l) ;
         y + L2 * cos(r+q1l) + L3 * cos(r+q1l+q2l) ];

q = [x; y; r; q1r; q2r; q1l; q2l;];
dq = [dx; dy; dr; dq1r; dq2r; dq1l; dq2l;];

gc = [toe_r ; toe_l];

J_gc = jacobian(gc,q);



Jpst_vec = J_gc(:);
dJpst_vec = jacobian(Jpst_vec,q)*dq;
dJ_gc = reshape(dJpst_vec,size(J_gc))

% [ 0, 0, - dq1r*(L2*sin(q1r + r) + L3*sin(q1r + q2r + r)) - dr*(L2*sin(q1r + r) + L3*sin(q1r + q2r + r)) - L3*dq2r*sin(q1r + q2r + r), - dq1r*(L2*sin(q1r + r) + L3*sin(q1r + q2r + r)) - dr*(L2*sin(q1r + r) + L3*sin(q1r + q2r + r)) - L3*dq2r*sin(q1r + q2r + r), - L3*dq1r*sin(q1r + q2r + r) - L3*dq2r*sin(q1r + q2r + r) - L3*dr*sin(q1r + q2r + r),                                                                                                                            0,                                                                                    0]
% [ 0, 0, - dq1r*(L2*cos(q1r + r) + L3*cos(q1r + q2r + r)) - dr*(L2*cos(q1r + r) + L3*cos(q1r + q2r + r)) - L3*dq2r*cos(q1r + q2r + r), - dq1r*(L2*cos(q1r + r) + L3*cos(q1r + q2r + r)) - dr*(L2*cos(q1r + r) + L3*cos(q1r + q2r + r)) - L3*dq2r*cos(q1r + q2r + r), - L3*dq1r*cos(q1r + q2r + r) - L3*dq2r*cos(q1r + q2r + r) - L3*dr*cos(q1r + q2r + r),                                                                                                                            0,                                                                                    0]
% [ 0, 0, - dq1l*(L2*sin(q1l + r) + L3*sin(q1l + q2l + r)) - dr*(L2*sin(q1l + r) + L3*sin(q1l + q2l + r)) - L3*dq2l*sin(q1l + q2l + r),                                                                                                                            0,                                                                                    0, - dq1l*(L2*sin(q1l + r) + L3*sin(q1l + q2l + r)) - dr*(L2*sin(q1l + r) + L3*sin(q1l + q2l + r)) - L3*dq2l*sin(q1l + q2l + r), - L3*dq1l*sin(q1l + q2l + r) - L3*dq2l*sin(q1l + q2l + r) - L3*dr*sin(q1l + q2l + r)]
% [ 0, 0, - dq1l*(L2*cos(q1l + r) + L3*cos(q1l + q2l + r)) - dr*(L2*cos(q1l + r) + L3*cos(q1l + q2l + r)) - L3*dq2l*cos(q1l + q2l + r),                                                                                                                            0,                                                                                    0, - dq1l*(L2*cos(q1l + r) + L3*cos(q1l + q2l + r)) - dr*(L2*cos(q1l + r) + L3*cos(q1l + q2l + r)) - L3*dq2l*cos(q1l + q2l + r), - L3*dq1l*cos(q1l + q2l + r) - L3*dq2l*cos(q1l + q2l + r) - L3*dr*cos(q1l + q2l + r)]

matlabFunction(dJ_gc, 'File', 'tmp/auto_dJ','vars',{L1,L2,L3,q,dq});