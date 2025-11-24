% Opinion dynamics without control

clear all
close all
clc

% Initial conditions
load('motiv_exd2N10_dati.mat','x')

% Parameters
d = 2;           % agent dimension
N = 10;          % number of agents
alpha = 300;     % parameter in the opinion dynamics
beta = 0.8;

ht = 0.01;
t0    = 0;
tf = 100000;
tspan = t0:ht:tf;
nt    = length(tspan);

% Interaction function
sig = @(y) 1./(1 + exp(-y));
phi = @(s) (1 - sig(alpha*(s - 1))) / (1 - sig(-alpha));

% Nuova matrice
kmat = zeros(N,N);
for i = 2 : N-1
    kmat(i,i+1) = 1;
    kmat(i,i-1) = -1;
end
kmat(1,2) = 1;
kmat(1,N) = -1;
kmat(N,1) = 1;
kmat(N,N-1) = -1;

% Initialization
xnew = cell(N,1);
for i = 1:N
    xnew{i} = zeros(d, nt);
    xnew{i}(:,1) = x{i}(:,1);
end


x_mean(:,1) = mean(xnew{i}(:,1),2);
Xt = zeros(1, nt);
% Initial consensus parameter
Xt(1) = 0;
for i = 1:N
    Xt(1) = Xt(1) + norm(xnew{i}(:,1) - x_mean(:,1))^2;
end
Xt(1) = Xt(1) / (N^2);

% Evolution dynamics
for k = 2:nt
    Phi = zeros(N,N);
    Phitilde = zeros(N,N);
    Xkm1 = zeros(d,N);
    for i = 1:N
        Xkm1(:,i) = xnew{i}(:,k-1);
    end
    eps_vec = zeros(N,1);
    for i = 1 : N
        if i==1
            idx_prev = N;
        else
            idx_prev = i-1;
        end
        dij = norm(Xkm1(:,i) - Xkm1(:,idx_prev));
        eps_vec(i) = phi(dij);
    end
    epsilon = beta*min(eps_vec);
    for i = 1:N
        for j = 1:N
            if j==i, Phi(i,j) = 0; continue; end
            dij = norm(Xkm1(:,i) - Xkm1(:,j));
            Phi(i,j) = max(phi(dij), 0);
            Phitilde(i,j) = Phi(i,j) + epsilon*kmat(i,j);
        end
    end
    Phitilde = Phitilde/N;
    for i = 1:N
        xi = Xkm1(:,i);
        Si = zeros(d,1);
        for j = 1:N
            if j==i, continue; end
            aij = Phitilde(i,j);
            Si  = Si + aij * (Xkm1(:,j) - xi);
        end
        xnew{i}(:,k) = xnew{i}(:,k-1) + ht* Si;
    end
    % Consensus parameter
    X_k = zeros(d,N);
    for i = 1:N
        X_k(:,i) = xnew{i}(:,k);
    end
    x_mean(:,k) = mean(X_k, 2);
    Xt(k) = 0;
    for i = 1:N
        Xt(k) = Xt(k) + norm(xnew{i}(:,k) - x_mean(:,k))^2;
    end
    Xt(k) = Xt(k) / (N^2);
end

% Compute mean value
mediax = zeros(N, nt);
for i = 1:N
    for k = 1:nt
        mediax(i,k) = mean(xnew{i}(:,k));
    end
end

% Plots
% figure
hold on
for i = 1:N
    plot(tspan, mediax(i,:), 'LineWidth', 1.5)
end
xlim([0 tf])
ylim([0 8])
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
xlabel('Time')
ylabel('Opinions')
title(['Opinion Dynamics — \alpha = ', num2str(alpha)])
box on

figure
semilogy(tspan, Xt, 'LineWidth', 2)
xlim([0 tf])
xlabel('Time')
ylabel('X_t')
title(['Consensus Parameter – \alpha = ', num2str(alpha)])
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold')
box on
