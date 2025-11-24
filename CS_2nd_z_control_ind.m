% Z-Controlled Multi-Agent Indirect Control -
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
tol_consensus = 1e-15; %threshold for consensus

% Interaction function
K = 1;
beta = 1;
phi  = @(r) K ./ (1 + r.^2).^beta/N;
bfun = @(r) -2*beta*K ./ (1 + r.^2).^(beta + 1)/N;

% Initialization
xnew = cell(N,1);
vnew = cell(N,1);
u = cell(N,1);
for i = 1:N
    xnew{i} = zeros(d,1e5);
    vnew{i} = zeros(d,1e5);
    u{i} = zeros(d,1e5);
    xnew{i}(:,1) = x{i}(:,1);
    vnew{i}(:,1) = x{i}(:,1);
end

X_k = zeros(d, N);
V_k = zeros(d, N);
for i = 1:N
    X_k(:,i) = xnew{i}(:,1);
    V_k(:,i) = vnew{i}(:,1);
end

x_mean(:,1) = mean(X_k, 2);
v_mean(:,1) = mean(V_k, 2);
vbar0 = v_mean(:,1)';

% Initial consensus parameter
Xt = 0;
for i = 1:N
    Xt = Xt + norm(vnew{i}(:,1) - v_mean(:,1))^2;
end
Xt = Xt / (N^2);
X_t(1) = Xt;

% Evolution dynamic
k = 1;
% tmax = 600;
% extime = 0;
errore = 1;
tol_errore = 1e-20;
while Xt > tol_consensus && errore > tol_errore
    Xt
    % tic;
    k = k+1;
    X = zeros(N,d);
    V = zeros(N,d);
    for i = 1:N
        X(i,:) = xnew{i}(:,k-1)';
        V(i,:) = vnew{i}(:,k-1)';
    end
    % Block matrix LB Nd x Nd
    LB = zeros(N*d, N*d);
    for i = 1:N
        ii = (i-1)*d + (1:d);  % indici righe/colonne del blocco (i,i)
        % Blocks outside diagonal -(b_ij) (v_j - v_i)(x_i - x_j)^T
        for j = 1:N
            if j == i, continue; end
            jj  = (j-1)*d + (1:d);
            xij = X(i,:) - X(j,:);
            vji = V(j,:) - V(i,:);
            bij = bfun(norm(xij));
            LB(ii, jj) = - bij * (vji' * xij);   % d x d
        end
        % Diagonal Blocks sum_{k != i} b_ik (v_k - v_i)(x_i - x_k)^T
        blk = zeros(d,d);
        for kk = 1:N
            if kk == i, continue; end
            xik = X(i,:) - X(kk,:);
            vki = V(kk,:) - V(i,:);
            bik = bfun(norm(xik));
            blk = blk + bik * (vki' * xik);      % d x d
        end
        LB(ii, ii) = blk;
    end
    % R vector
    R = zeros(N,d);
    for i = 1:N
        r1 = zeros(1,d); r2 = zeros(1,d); r3 = zeros(1,d); r4 = zeros(1,d); r5 = zeros(1,d);
        % r_i = sum_k a_ik
        ri = 0;
        for kk = 1:N
            ri = ri + phi(norm(X(i,:) - X(kk,:)));
        end
        for j = 1:N
            if j ~= i
                xij = X(i,:) - X(j,:);
                vji = V(j,:) - V(i,:);
                bij = bfun(norm(xij));
                r1 = r1 + bij * dot(xij, -vji) * (vji);
            end
            % a_ij
            aij = phi(norm(X(i,:) - X(j,:)));
            % sum_k a_ij (a_jk - a_ik) v_k
            % and r_j = sum_k a_jk
            rj = 0;
            for k2 = 1:N
                ajk = phi(norm(X(j,:) - X(k2,:)));
                aik = phi(norm(X(i,:) - X(k2,:)));
                vk2 = V(k2,:);
                r2 = r2 + aij * (ajk - aik) * vk2;
                rj = rj + ajk;           % accumula r_j
            end
            % - sum_j a_ij r_j v_j    (+ r_i^2 v_i outside cycle su j)
            r5 = r5 - aij * rj * V(j,:);
            % 2*lambda * sum_j a_ij (v_j - v_i)
            vji = V(j,:) - V(i,:);
            r3 = r3 + (2*lam) * aij * vji;
        end
        %  + r_i^2 v_i
        r5 = r5 + (ri^2) * V(i,:);
        % lambda^2 (v_i - v_bar)
        % r4 = -lam^2 * (V(i,:) - v_mean);
        r4 = lam^2 * (V(i,:) - vbar0);
        R(i,:) = r1 + r2 + r3 + r4 + r5;
    end
    R_vec = reshape(R', N*d, 1);
    U_vec = lsqminnorm(LB, -R_vec);
    U = reshape(U_vec, d, N)';
    % Update dynamics
    for i = 1:N
        u{i}(:,k-1) = U(i,:)';
        sum_inter = zeros(d,1);
        for j = 1:N
            if j ~= i
                xi = xnew{i}(:,k-1);
                xj = xnew{j}(:,k-1);
                vi = vnew{i}(:,k-1);
                vj = vnew{j}(:,k-1);
                aij = phi(norm(xi-xj));
                sum_inter = sum_inter + aij*(vj - vi);
            end
        end
        vnew{i}(:,k) = vnew{i}(:,k-1) + ht*sum_inter;
        xnew{i}(:,k) = xnew{i}(:,k-1) + ht*(vnew{i}(:,k-1) + u{i}(:,k-1));
        X_k(:,i) = xnew{i}(:,k);
        V_k(:,i) = vnew{i}(:,k);
    end
    x_mean(:,k) = mean(X_k,2);
    v_mean(:,k) = mean(V_k,2);
    % Update consensus parameter
    Xt = 0;
    for i = 1:N
        Xt = Xt + norm(vnew{i}(:,k) - v_mean(:,k))^2;
    end
    Xt = Xt / (N^2);
    X_t(k) = Xt;
    % tempo = toc;
    % extime = extime + tempo;
    errore = abs(Xt-X_t(k-1));
end

tspan = (0:k-1)*ht;
tf = k*ht;
nt = length(tspan);

% Compute control at the last time step
X = zeros(N,d); 
V = zeros(N,d);
for i = 1:N
    X(i,:) = xnew{i}(:,nt)';
    V(i,:) = vnew{i}(:,nt)';
end

LB = zeros(N*d, N*d);
for i = 1:N
    ii = (i-1)*d + (1:d);  % index row/column block (i,i)
    for j = 1:N
        if j == i, continue; end
        jj  = (j-1)*d + (1:d);
        xij = X(i,:) - X(j,:);
        vji = V(j,:) - V(i,:);
        bij = bfun(norm(xij));
        LB(ii, jj) = - bij * (vji' * xij);   % d x d
    end
    % Diag block:  sum_{k != i} b_ik (v_k - v_i)(x_i - x_k)^T
    blk = zeros(d,d);
    for kk = 1:N
        if kk == i, continue; end
        xik = X(i,:) - X(kk,:);
        vki = V(kk,:) - V(i,:);
        bik = bfun(norm(xik));
        blk = blk + bik * (vki' * xik);      % d x d
    end
    LB(ii, ii) = blk;
end
R = zeros(N,d);
v_mean = mean(V,1);
for i = 1:N
    r1 = zeros(1,d); r2 = zeros(1,d); r3 = zeros(1,d); r4 = zeros(1,d); r5 = zeros(1,d);
    % r_i = sum_k a_ik
    ri = 0;
    for kk = 1:N
        ri = ri + phi(norm(X(i,:) - X(kk,:)));
    end
    for j = 1:N
        if j ~= i
            xij = X(i,:) - X(j,:);
            vji = V(j,:) - V(i,:);
            bij = bfun(norm(xij));
            r1 = r1 + bij * dot(xij, -vji) * (vji);
        end
        % a_ij
        aij = phi(norm(X(i,:) - X(j,:)));
        % sum_k a_ij (a_jk - a_ik) v_k
        % and r_j = sum_k a_jk
        rj = 0;
        for k2 = 1:N
            ajk = phi(norm(X(j,:) - X(k2,:)));
            aik = phi(norm(X(i,:) - X(k2,:)));
            vk2 = V(k2,:);
            r2 = r2 + aij * (ajk - aik) * vk2;
            rj = rj + ajk;          
        end
        %  - sum_j a_ij r_j v_j   (+ r_i^2 v_i outside j cycle)
        r5 = r5 - aij * rj * V(j,:);
        % 2*lambda * sum_j a_ij (v_j - v_i)
        vji = V(j,:) - V(i,:);
        r3 = r3 + (2*lam) * aij * vji;
    end
    % + r_i^2 v_i
    r5 = r5 + (ri^2) * V(i,:);
    % lambda^2 (v_i - v_bar)
    % r4 = -lam^2 * (V(i,:) - v_mean);
    r4 = lam^2 * (V(i,:) - vbar0);
    R(i,:) = r1 + r2 + r3 + r4 + r5;
end
R_vec = reshape(R', N*d, 1);
U_vec = lsqminnorm(LB,-R_vec);
U = reshape(U_vec, d, N)';
for i = 1:N
    u{i}(:,nt) = U(i,:)';
end

% Mean values
for i = 1:N
    for k = 1:nt
        mediax(i,k) = mean(xnew{i}(:,k));
        mediav(i,k) = mean(vnew{i}(:,k));
        mediau(i,k) = mean(u{i}(:,k));
    end
end

% Plots
figure
hold on
for i = 1:N
    plot(tspan, mediax(i,:), 'LineWidth',1.5)
end
set(gca, 'XScale','log','FontSize',14,'FontWeight','bold','LineWidth',1.5)
xlabel('Time') 
ylabel('Mean Position');
xlim([ht,tf])
title('Trajectories') 
box on

figure
hold on
for i = 1:N
    plot(tspan, mediav(i,:), 'LineWidth',1.5)
end
set(gca, 'XScale','log','FontSize',14,'FontWeight','bold','LineWidth',1.5)
xlabel('Time')
ylabel('Mean Velocity'); 
xlim([ht,tf])
% ylim([0 8])
title('Velocities')
box on

figure
hold on
for i = 1:N
    plot(tspan, mediau(i,:), 'LineWidth',1.5)
end
set(gca, 'XScale','log','FontSize',14,'FontWeight','bold','LineWidth',1.5)
xlabel('Time')
ylabel('Mean Control Input') 
xlim([ht,tf])
title('Controls')
box on


figure
semilogy(tspan, X_t, 'LineWidth', 2)
xlabel('Time')
ylabel('\Gamma(t)')
xlim([ht,tf])
title('Consensus')
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
box on