q1 = [cos(pi/8);0;0;sin(pi/8)]; %rotation about Z axis by 45 degrees
q2 = [cos(pi/4);sin(pi/4);0;0]; %rotation about X axis by 90 degrees

qa = quatmultiply(q1,q2)
qb = quatmultiply(q2,q1)

x = [0;0;1]; %vector in final frame of reference

%vector in global frame of reference for Case a
xa = quatmultiply(qa,quatmultiply([0;x],[qa(1);-qa(2);-qa(3);-qa(4)]))

%vector in global frame of reference for Case b
xb = quatmultiply(qb,quatmultiply([0;x],[qb(1);-qb(2);-qb(3);-qb(4)]))

function [ Q ] = quatmultiply( q,p )
%Performs Quaternion multiplication q o p
        q0=q(1);
        q1=q(2);
        q2=q(3);
        q3=q(4);

        Q1 = [q0 -q1 -q2 -q3;
        q1  q0 -q3  q2;
        q2  q3  q0 -q1;
        q3 -q2  q1  q0]; 
        Q = Q1*p;
end		