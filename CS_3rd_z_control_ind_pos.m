% Cucker-Smale generalized 3rd order
% Indirect Z-control through positions
clear all
close all
clc

% Initial conditions
% load('motiv_exd2N10_dati.mat','x')
load('dati_d3_N150.mat','x')

% Parameters
d = 3;           % agent dimension
N = 150;          % number of agents 
lam = 1.1; 
ht = 0.01;
t0 = 0;
tol_consensus = 1e-15; 

% Interaction function
beta = 1; 
K = 1;
phi  = @(r) K ./ (1 + r.^2).^beta/N;
bfun = @(r) -2*beta*K ./ (1 + r.^2).^(beta + 1)/N;

% Initialization
xnew = cell(N,1); 
vnew = cell(N,1); 
znew = cell(N,1); 
u = cell(N,1);
for i = 1:N
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
z_mean(:,1) = mean(Z_k, 2);                                  % \bar z(0)
zbar0 = z_mean(:,1)';

% Initial consensus parameter
Xt = 0;
for i = 1:N
    Xt = Xt + norm(znew{i}(:,1) - z_mean(:,1))^2;
end
Xt = Xt / (N^2);
X_t(1) = Xt;

k = 1;
tmax = 1200;
extime = 0;
while Xt > tol_consensus && extime < tmax
    Xt
    tic;
    k = k+1;
    X = zeros(N,d); 
    V = zeros(N,d); 
    Z = zeros(N,d);
    for i = 1:N
        X(i,:) = xnew{i}(:,k-1)';
        V(i,:) = vnew{i}(:,k-1)';
        Z(i,:) = znew{i}(:,k-1)';   % (4.13)
    end
    % Block matrix L_B(x,z):  off = -b_ij (z_j - z_i)(x_i - x_j)^T
    LB = zeros(N*d, N*d);
    for i = 1:N
        ii = (i-1)*d + (1:d);
        for j = 1:N
            if j == i, continue; end
            jj  = (j-1)*d + (1:d);
            xij = X(i,:) - X(j,:);
            zji = Z(j,:) - Z(i,:);
            bij = bfun(norm(xij));
            LB(ii, jj) = - bij * (zji' * xij);     % d x d
        end
        blk = zeros(d,d);
        for kk = 1:N
            if kk == i, continue; end
            xik = X(i,:) - X(kk,:);
            zki = Z(kk,:) - Z(i,:);
            bik = bfun(norm(xik));
            blk = blk + bik * (zki' * xik);        % d x d
        end
        LB(ii, ii) = blk;
    end
    % R(x,v,z;lambda)
    R = zeros(N,d);
    for i = 1:N
        r1 = zeros(1,d); r2 = zeros(1,d); r3 = zeros(1,d); r4 = zeros(1,d); r5 = zeros(1,d);
        % r_i = sum_k a_ik
        ri = 0;
        for kk = 1:N
            ri = ri + phi(norm(X(i,:) - X(kk,:)));
        end
        for j = 1:N
            aij = phi(norm(X(i,:) - X(j,:)));
            if j ~= i
                xij = X(i,:) - X(j,:);
                vij = V(i,:) - V(j,:);    % (v_i - v_j)
                zji = Z(j,:) - Z(i,:);    % (z_j - z_i)
                bij = bfun(norm(xij));
                % 1) sum_j b_ij * (x_i - x_j)^T (v_i - v_j) * (z_j - z_i)
                r1 = r1 + bij * dot(xij, vij) * (zji);
            end
            % 2) sum_{j,k} a_ij (a_jk - a_ik) z_k
            rj = 0;
            for k2 = 1:N
                ajk = phi(norm(X(j,:) - X(k2,:)));
                aik = phi(norm(X(i,:) - X(k2,:)));
                zk2 = Z(k2,:);
                r2  = r2 + aij * (ajk - aik) * zk2;  % *** segno corretto ***
                rj  = rj + ajk;                      % r_j
            end
            % 3) - sum_j a_ij r_j z_j
            r5 = r5 - aij * rj * Z(j,:);
            % 4) 2*lambda * sum_j a_ij (z_j - z_i)
            zji = Z(j,:) - Z(i,:);
            r3 = r3 + (2*lam) * aij * zji;
        end
        % 3): + r_i^2 z_i
        r5 = r5 + (ri^2) * Z(i,:);
        % 5) lambda^2 (z_i - zbar0)
        r4 = lam^2 * (Z(i,:) - zbar0);
        R(i,:) = r1 + r2 + r3 + r4 + r5;
    end
    R_vec = reshape(R', N*d, 1);
    U_vec = lsqminnorm(LB, -R_vec);
    U = reshape(U_vec, d, N)';
    % Update dynamics: x'=v+u, v'=z, z'=sum a_ij (z_j - z_i)
    for i = 1:N
        u{i}(:,k-1) = U(i,:)';
        sum_inter_z = zeros(d,1);
        for j = 1:N
            if j ~= i
                xi = xnew{i}(:,k-1); 
                xj = xnew{j}(:,k-1);
                zi = znew{i}(:,k-1); 
                zj = znew{j}(:,k-1);
                aij = phi(norm(xi - xj));
                sum_inter_z = sum_inter_z + aij*(zj - zi);
            end
        end
        znew{i}(:,k) = znew{i}(:,k-1) + ht*sum_inter_z;
        vnew{i}(:,k) = vnew{i}(:,k-1) + ht*znew{i}(:,k-1);
        xnew{i}(:,k) = xnew{i}(:,k-1) + ht*(vnew{i}(:,k-1) + u{i}(:,k-1));
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
    tempo = toc;
    extime = extime + tempo;
end

tspan = (0:k-1)*ht;
tf = k*ht;
nt = length(tspan);

% Compute control at the last time step
X = zeros(N,d); 
V = zeros(N,d); 
Z = zeros(N,d);
for i = 1:N
    X(i,:) = xnew{i}(:,nt)';
    V(i,:) = vnew{i}(:,nt)';
    Z(i,:) = znew{i}(:,nt)';
end
LB = zeros(N*d, N*d);
for i = 1:N
    ii = (i-1)*d + (1:d);
    for j = 1:N
        if j == i, continue; end
        jj  = (j-1)*d + (1:d);
        xij = X(i,:) - X(j,:);
        zji = Z(j,:) - Z(i,:);
        bij = bfun(norm(xij));
        LB(ii, jj) = - bij * (zji' * xij);
    end
    blk = zeros(d,d);
    for kk = 1:N
        if kk == i, continue; end
        xik = X(i,:) - X(kk,:);
        zki = Z(kk,:) - Z(i,:);
        bik = bfun(norm(xik));
        blk = blk + bik * (zki' * xik);
    end
    LB(ii, ii) = blk;
end
R = zeros(N,d);
for i = 1:N
    r1 = zeros(1,d); r2 = zeros(1,d); r3 = zeros(1,d); r4 = zeros(1,d); r5 = zeros(1,d);
    ri = 0; 
    for kk = 1:N 
        ri = ri + phi(norm(X(i,:) - X(kk,:))); 
    end
    for j = 1:N
        aij = phi(norm(X(i,:) - X(j,:)));
        if j ~= i
            xij = X(i,:) - X(j,:);
            vij = V(i,:) - V(j,:);
            zji = Z(j,:) - Z(i,:);
            bij = bfun(norm(xij));
            r1 = r1 + bij * dot(xij, vij) * (zji);
        end
        rj = 0;
        for k2 = 1:N
            ajk = phi(norm(X(j,:) - X(k2,:)));
            aik = phi(norm(X(i,:) - X(k2,:)));
            zk2 = Z(k2,:);
            r2  = r2 + aij * (ajk - aik) * zk2; % segno corretto
            rj  = rj + ajk;
        end
        r5 = r5 - aij * rj * Z(j,:);
        zji = Z(j,:) - Z(i,:);
        r3 = r3 + (2*lam) * aij * zji;
    end
    r5 = r5 + (ri^2) * Z(i,:);
    r4 = lam^2 * (Z(i,:) - zbar0);
    R(i,:) = r1 + r2 + r3 + r4 + r5;
end
R_vec = reshape(R', N*d, 1);
U_vec = lsqminnorm(LB, -R_vec);
U = reshape(U_vec, d, N)';
for i = 1:N
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