% Cucker-Smale generalized 3rd order
% Indirect Z-control through velocities

clear all
close all
clc

% Initial conditions
load('motiv_exd2N10_dati.mat','x')
% load('dati_d2_N50.mat','x')

% Parameters
d = 2;           % agent dimension
N = 10;          % number of agents 
lam = 1; %0.5;
ht = 0.01;
t0 = 0; 
tol_consensus = 1e-15; 

beta = 1;
K = 1;                                            % stesso G del tuo codice
phi  = @(r) (K ./ (1 + r.^2).^beta) / N;          % a_ij(x)
bfun = @(r) (-2*beta*K ./ (1 + r.^2).^(beta+1)) / N;  % b_ij(x)

% Initialization
xnew = cell(N,1); 
vnew = cell(N,1); 
znew = cell(N,1); 
u = cell(N,1);
for i=1:N
    xnew{i} = zeros(d,1e5);
    vnew{i} = zeros(d,1e5);
    znew{i} = zeros(d,1e5);  
    u{i}    = zeros(d,1e5);
    xnew{i}(:,1) = x{i}(:,1);
    vnew{i}(:,1) = x{i}(:,1);    
    znew{i}(:,1) = x{i}(:,1);
end

X_k = zeros(d, N);
V_k = zeros(d, N);
Z_k = zeros(d, N);
for i = 1:N
    X_k(:,i) = xnew{i}(:,1);
    V_k(:,i) = vnew{i}(:,1);
    Z_k(:,i) = znew{i}(:,1);
end

x_mean(:,1) = mean(X_k, 2);
v_mean(:,1) = mean(V_k, 2);
z_mean(:,1) = mean(Z_k, 2);                               
zbar0 = z_mean(:,1)';

% Initial consensus parameter
Xt = 0;
for i = 1:N
    Xt = Xt + norm(znew{i}(:,1) - z_mean(:,1))^2;
end
Xt = Xt / (N^2);
X_t(1) = Xt;

k = 1;
while Xt > tol_consensus
    Xt
    k = k+1;
    X = zeros(N,d); 
    V = zeros(N,d); 
    Z = zeros(N,d);
    for i=1:N 
        X(i,:) = xnew{i}(:,k-1)'; 
        V(i,:) = vnew{i}(:,k-1)'; 
        Z(i,:) = znew{i}(:,k-1)'; 
    end
    % L_B(x,z)
    LB = zeros(N*d,N*d);
    for i=1:N
        ii = (i-1)*d + (1:d);
        % Off-diagonal:  −b_ij (zj−zi)(xi−xj)^T  =  + b_ij (zij)(xi−xj)^T
        for j=1:N
            if j==i, continue; end
            jj  = (j-1)*d + (1:d);
            xij = X(i,:) - X(j,:);
            zij = Z(i,:) - Z(j,:);      % zi - zj
            bij = bfun(norm(xij));
            LB(ii,jj) = + bij * (zij' * xij);
        end
        % Diagonal:  sum_k b_ik (zk−zi)(xi−xk)^T  =  − sum_k b_ik (zik)(xi−xk)^T
        blk = zeros(d,d);
        for k2=1:N
            if k2==i, continue; end
            xik = X(i,:) - X(k2,:);
            zik = Z(i,:) - Z(k2,:);     % zi - zk
            bik = bfun(norm(xik));
            blk = blk - bik * (zik' * xik);
        end
        LB(ii,ii) = blk;
    end
    % Pre-sum scalar (on-the-fly)
    ri   = zeros(N,1);   % r_i = sum_k a_ik
    Si   = zeros(N,1);   % S_i = sum_j b_ij s_ij  (= rdot_i)
    for i=1:N
        for k2=1:N
            if k2==i, continue; end
            xik = X(i,:) - X(k2,:);
            vik = V(i,:) - V(k2,:);
            a_ik = phi(norm(xik));
            b_ik = bfun(norm(xik));
            s_ik = dot(xik, vik);
            ri(i) = ri(i) + a_ik;
            Si(i) = Si(i) + b_ik * s_ik;
        end
    end
    rdot = Si;   % \dot r_i = sum_k b_ik s_ik
    % R_i
    R = zeros(N,d);
    for i=1:N
        Ri = zeros(1,d);
        % T1: sum_j ddot a_ij^(0) (zj - zi) = - sum_j ddot0 * zij
        for j=1:N
            if j==i, continue; end
            xij = X(i,:) - X(j,:);
            vij = V(i,:) - V(j,:);
            zij = Z(i,:) - Z(j,:);
            rho = 1 + dot(xij,xij);
            sij = dot(xij, vij);
            qij = dot(vij, vij);
            rij = dot(xij, zij);   % (xi-xj)^T(zi-zj)
            ddot0 = (1/N) * ( 4*beta*(beta+1)*K * rho^(-beta-2) * (sij^2) ...
                            - 2*beta*K * rho^(-beta-1) * (qij + rij) );
            Ri = Ri - ddot0 * zij;
        end
        % T2: 2 sum_{j,k} (b_ij s_ij) (a_jk - a_ik) z_k
        for j=1:N
            if j==i, continue; end
            xij = X(i,:) - X(j,:);
            vij = V(i,:) - V(j,:);
            sij = dot(xij, vij);
            bij = bfun(norm(xij));
            coeff = 2 * bij * sij;
            for k2=1:N
                xjk = X(j,:) - X(k2,:);
                xik = X(i,:) - X(k2,:);
                ajk = phi(norm(xjk));
                aik = phi(norm(xik));
                Ri = Ri + coeff * (ajk - aik) * Z(k2,:);
            end
        end
        % T3: 2 r_i * (sum_j b_ij s_ij) * z_i
        Ri = Ri + 2 * ri(i) * Si(i) * Z(i,:);
        % T4: -2 sum_j (b_ij s_ij) r_j z_j
        for j=1:N
            if j==i, continue; end
            xij = X(i,:) - X(j,:);
            vij = V(i,:) - V(j,:);
            sij = dot(xij, vij);
            bij = bfun(norm(xij));
            Ri = Ri - 2 * (bij*sij) * ri(j) * Z(j,:);
        end
        % T5: sum_{j,k} a_ij (dot a_jk - dot a_ik) z_k, con dot a = b*s
        for j=1:N
            if j==i, continue; end
            aij = phi(norm(X(i,:) - X(j,:)));
            for k2=1:N
                xjk = X(j,:) - X(k2,:);  vjk = V(j,:) - V(k2,:);
                xik = X(i,:) - X(k2,:);  vik = V(i,:) - V(k2,:);
                sjk = dot(xjk, vjk);     sik = dot(xik, vik);
                bjk = bfun(norm(xjk));   bik = bfun(norm(xik));
                Ri  = Ri + aij * (bjk*sjk - bik*sik) * Z(k2,:);
            end
        end
        % T6: sum_{j,k,l} a_ij a_kℓ (a_jk - a_ik) (z_k - z_ℓ)
        for j=1:N
            if j==i, continue; end
            aij = phi(norm(X(i,:) - X(j,:)));
            for k2=1:N
                ajk = phi(norm(X(j,:) - X(k2,:)));
                aik = phi(norm(X(i,:) - X(k2,:)));
                for ell=1:N
                    akl = phi(norm(X(k2,:) - X(ell,:)));
                    zkl = Z(k2,:) - Z(ell,:);
                    Ri  = Ri + aij * akl * (ajk - aik) * zkl;
                end
            end
        end
        % T7: - sum_j a_ij * rdot_j * z_j
        for j=1:N
            if j==i, continue; end
            aij = phi(norm(X(i,:) - X(j,:)));
            Ri  = Ri - aij * rdot(j) * Z(j,:);
        end
        % T8: - sum_{j,k} a_ij a_jk r_j (z_k - z_j)
        for j=1:N
            if j==i, continue; end
            aij = phi(norm(X(i,:) - X(j,:)));
            for k2=1:N
                ajk = phi(norm(X(j,:) - X(k2,:)));
                zkj = Z(k2,:) - Z(j,:);
                Ri  = Ri - aij * ajk * ri(j) * zkj;
            end
        end
        % T9: + r_i * rdot_i * z_i
        Ri = Ri + ri(i) * rdot(i) * Z(i,:);
        % T10: + sum_j a_ij r_i^2 (z_j - z_i) = - sum_j a_ij r_i^2 * zij
        for j=1:N
            if j==i, continue; end
            aij = phi(norm(X(i,:) - X(j,:)));
            zij = Z(i,:) - Z(j,:);
            Ri  = Ri - (ri(i)^2) * aij * zij;
        end
        % T11: + 3λ sum_j (b_ij s_ij) (z_j - z_i) = - 3λ sum_j (b_ij s_ij) * zij
        for j=1:N
            if j==i, continue; end
            xij = X(i,:) - X(j,:);
            vij = V(i,:) - V(j,:);
            sij = dot(xij, vij);
            bij = bfun(norm(xij));
            zij = Z(i,:) - Z(j,:);
            Ri  = Ri - 3*lam * (bij*sij) * zij;
        end
        % T12: + 3λ sum_{j,k} a_ij (a_jk - a_ik) z_k
        for j=1:N
            if j==i, continue; end
            aij = phi(norm(X(i,:) - X(j,:)));
            for k2=1:N
                ajk = phi(norm(X(j,:) - X(k2,:)));
                aik = phi(norm(X(i,:) - X(k2,:)));
                Ri  = Ri + 3*lam * aij * (ajk - aik) * Z(k2,:);
            end
        end
        % T13: + 3λ r_i^2 z_i
        Ri = Ri + 3*lam * (ri(i)^2) * Z(i,:);
        % T14: - 3λ sum_j a_ij r_j z_j
        for j=1:N
            if j==i, continue; end
            aij = phi(norm(X(i,:) - X(j,:)));
            Ri  = Ri - 3*lam * aij * ri(j) * Z(j,:);
        end
        % T15: + 3λ^2 sum_j a_ij (z_j - z_i) = - 3λ^2 sum_j a_ij * zij
        for j=1:N
            if j==i, continue; end
            aij = phi(norm(X(i,:) - X(j,:)));
            zij = Z(i,:) - Z(j,:);
            Ri  = Ri - 3*(lam^2) * aij * zij;
        end
        % T16: + λ^3 (z_i - zbar)
        Ri = Ri + (lam^3) * (Z(i,:) - zbar0);
        R(i,:) = Ri;
    end
    R_vec = reshape(R',N*d,1);
    U_vec = lsqminnorm(LB, -R_vec);
    U = reshape(U_vec,d,N)';
    % Dynamics:  x' = v,  v' = z + u,  z' = Σ aij (zj − zi)
    for i=1:N
        u{i}(:,k-1) = U(i,:)';
        sum_inter_z = zeros(d,1);
        for j=1:N
            if j==i, continue; end
            aij = phi(norm(xnew{i}(:,k-1) - xnew{j}(:,k-1)));
            zij = znew{i}(:,k-1) - znew{j}(:,k-1);
            sum_inter_z = sum_inter_z - aij * zij;
        end
        znew{i}(:,k) = znew{i}(:,k-1) + ht*sum_inter_z;
        vnew{i}(:,k) = vnew{i}(:,k-1) + ht*( znew{i}(:,k-1) + u{i}(:,k-1) );
        xnew{i}(:,k) = xnew{i}(:,k-1) + ht* vnew{i}(:,k-1);
        X_k(:,i) = xnew{i}(:,k);
        V_k(:,i) = vnew{i}(:,k);
        Z_k(:,i) = znew{i}(:,k);
    end
    x_mean(:,k) = mean(X_k,2);
    v_mean(:,k) = mean(V_k,2);
    z_mean(:,k) = mean(Z_k,2);
    Xt = 0;
    for i = 1:N
        Xt = Xt + norm(znew{i}(:,k) - z_mean(:,k))^2;
    end
    Xt = Xt / (N^2);
    X_t(k) = Xt;
end

tspan = (0:k-1)*ht;
tf = k*ht;
nt = length(tspan);

% Compute control at the last time step
X = zeros(N,d); 
V = zeros(N,d); 
Z = zeros(N,d);
for i=1:N
    X(i,:) = xnew{i}(:,nt)'; 
    V(i,:) = vnew{i}(:,nt)'; 
    Z(i,:) = znew{i}(:,nt)'; 
end
% zbar = mean(Z,1);

LB = zeros(N*d,N*d);
for i=1:N
    ii=(i-1)*d+(1:d);
    for j=1:N
        if j==i, continue; end
        jj=(j-1)*d+(1:d);
        xij = X(i,:) - X(j,:);
        zij = Z(i,:) - Z(j,:);
        bij = bfun(norm(xij));
        LB(ii,jj) = + bij * (zij' * xij);
    end
    blk=zeros(d,d);
    for k2=1:N
        if k2==i, continue; end
        xik = X(i,:) - X(k2,:);
        zik = Z(i,:) - Z(k2,:);
        bik = bfun(norm(xik));
        blk = blk - bik * (zik' * xik);
    end
    LB(ii,ii) = blk;
end

ri   = zeros(N,1); Si = zeros(N,1);
for i=1:N
    for k2=1:N
        if k2==i, continue; end
        xik = X(i,:) - X(k2,:);
        vik = V(i,:) - V(k2,:);
        a_ik = phi(norm(xik));
        b_ik = bfun(norm(xik));
        s_ik = dot(xik, vik);
        ri(i) = ri(i) + a_ik;
        Si(i) = Si(i) + b_ik * s_ik;
    end
end
rdot = Si;
R = zeros(N,d);
for i=1:N
    Ri = zeros(1,d);
    % T1
    for j=1:N
        if j==i, continue; end
        xij = X(i,:) - X(j,:);
        vij = V(i,:) - V(j,:);
        zij = Z(i,:) - Z(j,:);
        rho = 1 + dot(xij,xij);
        sij = dot(xij, vij);
        qij = dot(vij, vij);
        rij = dot(xij, zij);
        ddot0 = (1/N) * ( 4*beta*(beta+1)*K * rho^(-beta-2) * (sij^2) ...
                        - 2*beta*K * rho^(-beta-1) * (qij + rij) );
        Ri = Ri - ddot0 * zij;
    end
    % T2
    for j=1:N
        if j==i, continue; end
        xij = X(i,:) - X(j,:);
        vij = V(i,:) - V(j,:);
        sij = dot(xij, vij);
        bij = bfun(norm(xij));
        coeff = 2 * bij * sij;
        for k2=1:N
            xjk = X(j,:) - X(k2,:);
            xik = X(i,:) - X(k2,:);
            ajk = phi(norm(xjk));
            aik = phi(norm(xik));
            Ri = Ri + coeff * (ajk - aik) * Z(k2,:);
        end
    end
    % T3
    Ri = Ri + 2 * ri(i) * Si(i) * Z(i,:);
    % T4
    for j=1:N
        if j==i, continue; end
        xij = X(i,:) - X(j,:);
        vij = V(i,:) - V(j,:);
        sij = dot(xij, vij);
        bij = bfun(norm(xij));
        Ri = Ri - 2 * (bij*sij) * ri(j) * Z(j,:);
    end
    % T5
    for j=1:N
        if j==i, continue; end
        aij = phi(norm(X(i,:) - X(j,:)));
        for k2=1:N
            xjk = X(j,:) - X(k2,:);  vjk = V(j,:) - V(k2,:);
            xik = X(i,:) - X(k2,:);  vik = V(i,:) - V(k2,:);
            sjk = dot(xjk, vjk);     sik = dot(xik, vik);
            bjk = bfun(norm(xjk));   bik = bfun(norm(xik));
            Ri  = Ri + aij * (bjk*sjk - bik*sik) * Z(k2,:);
        end
    end
    % T6
    for j=1:N
        if j==i, continue; end
        aij = phi(norm(X(i,:) - X(j,:)));
        for k2=1:N
            ajk = phi(norm(X(j,:) - X(k2,:)));
            aik = phi(norm(X(i,:) - X(k2,:)));
            for ell=1:N
                akl = phi(norm(X(k2,:) - X(ell,:)));
                zkl = Z(k2,:) - Z(ell,:);
                Ri  = Ri + aij * akl * (ajk - aik) * zkl;
            end
        end
    end
    % T7
    for j=1:N
        if j==i, continue; end
        aij = phi(norm(X(i,:) - X(j,:)));
        Ri  = Ri - aij * rdot(j) * Z(j,:);
    end
    % T8
    for j=1:N
        if j==i, continue; end
        aij = phi(norm(X(i,:) - X(j,:)));
        for k2=1:N
            ajk = phi(norm(X(j,:) - X(k2,:)));
            zkj = Z(k2,:) - Z(j,:);
            Ri  = Ri - aij * ajk * ri(j) * zkj;
        end
    end
    % T9
    Ri = Ri + ri(i) * rdot(i) * Z(i,:);
    % T10
    for j=1:N
        if j==i, continue; end
        aij = phi(norm(X(i,:) - X(j,:)));
        zij = Z(i,:) - Z(j,:);
        Ri  = Ri - (ri(i)^2) * aij * zij;
    end
    % T11
    for j=1:N
        if j==i, continue; end
        xij = X(i,:) - X(j,:);
        vij = V(i,:) - V(j,:);
        sij = dot(xij, vij);
        bij = bfun(norm(xij));
        zij = Z(i,:) - Z(j,:);
        Ri  = Ri - 3*lam * (bij*sij) * zij;
    end
    % T12
    for j=1:N
        if j==i, continue; end
        aij = phi(norm(X(i,:) - X(j,:)));
        for k2=1:N
            ajk = phi(norm(X(j,:) - X(k2,:)));
            aik = phi(norm(X(i,:) - X(k2,:)));
            Ri  = Ri + 3*lam * aij * (ajk - aik) * Z(k2,:);
        end
    end
    % T13
    Ri = Ri + 3*lam * (ri(i)^2) * Z(i,:);
    % T14
    for j=1:N
        if j==i, continue; end
        aij = phi(norm(X(i,:) - X(j,:)));
        Ri  = Ri - 3*lam * aij * ri(j) * Z(j,:);
    end
    % T15
    for j=1:N
        if j==i, continue; end
        aij = phi(norm(X(i,:) - X(j,:)));
        zij = Z(i,:) - Z(j,:);
        Ri  = Ri - 3*(lam^2) * aij * zij;
    end
    % T16
    Ri = Ri + (lam^3) * (Z(i,:) - zbar0);
    R(i,:) = Ri;
end
R_vec = reshape(R',N*d,1);
U_vec = lsqminnorm(LB, -R_vec);
U = reshape(U_vec,d,N)'; 
for i=1:N
    u{i}(:,nt) = U(i,:)'; 
end

for i = 1:N
    for k = 1:nt
        mediax(i,k) = mean(xnew{i}(:,k));
        mediav(i,k) = mean(vnew{i}(:,k));
        mediaa(i,k) = mean(znew{i}(:,k));
        mediau(i,k) = mean(u{i}(:,k));
    end
end

% Plots
figure
hold on
for i = 1:N
    plot(tspan, mediax(i,:), 'LineWidth', 1.5)
end
xlim([0 tf])
xlabel('Time'); ylabel('Mean Position')
title('Trajectories')
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
box on

figure
hold on
for i = 1:N
    plot(tspan, mediav(i,:), 'LineWidth', 1.5)
end
xlim([0 tf])
xlabel('Time'); ylabel('Mean Velocity')
title('Velocities')
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
box on

figure
hold on
for i = 1:N
    plot(tspan, mediaa(i,:), 'LineWidth', 1.5)
end
xlim([0 tf])
ylim([0 8])
xlabel('Time'); ylabel('Mean Acceleration')
title('Acceleration')
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
box on

figure
hold on
for i = 1:N
    plot(tspan, mediau(i,:), 'LineWidth', 1.5)
end
xlim([0 tf])
xlabel('Time'); ylabel('Mean Control Input')
title('Controls')
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
box on

figure
semilogy(tspan, X_t(1:k), 'LineWidth', 2)
xlabel('Time'); ylabel('\Gamma(t)')
title('Consensus')
xlim([0 tf])
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
box on
