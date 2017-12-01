function d = quatDist(q, QJ)
    nj = size(QJ,1);
    d = zeros(nj,1);
    for n = 1:nj
        d(n) = qDist(q, QJ(n,:));
    end
end

function d = qDist(q1,q2)
    %disp(q1);
    %disp(q2);
    r1 = quat2dcm(q1);
    r2 = quat2dcm(q2);
    r_rel = r1'*r2;
    d = norm(logm(r_rel),'fro')/sqrt(2);
end