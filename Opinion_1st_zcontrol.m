% Opinion dynamics
% Z-control
clear all
close all
clc

% Initial conditions
% load('motiv_exd2N10_dati.mat','x')
load('dati_d150_N150.mat', 'x')

% Parameters
d = 150;           % agent dimension
N = 150;          % number of agents
alpha = 1.6;     % parameter in the opinion dynamics
ht = 1e-2;
beta = 0.8;

tol_consensus = 1e-15;  % threshold for consensus
lam = 10; % lambda value for Z-control

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

% Interaction function
sig = @(y) 1./(1 + exp(-y));
phi = @(s) (1 - sig(alpha*(s - 1))) / (1 - sig(-alpha));

% Initialization
xnew = cell(N,1);
u = cell(N,1);
for i = 1:N
    xnew{i} = zeros(d,1e5);  % preallocation
    u{i} = zeros(d,1e5);
    xnew{i}(:,1) = x{i}(:,1);
end
X_k = zeros(d, N);
for i = 1:N
    X_k(:,i) = xnew{i}(:,1);
end
x_mean(:,1) = mean(X_k, 2);

% Initial consensus parameter
Xt0 = 0;
for i = 1:N
    Xt0 = Xt0 + norm(xnew{i}(:,1) - x_mean(:,1))^2;
end
Xt0 = Xt0/(N^2);
X_t(1) = Xt0;

% Evolution dynamics
k = 1;
while Xt0 > tol_consensus
    Xt0
    k = k + 1;
    Phi = zeros(N,N);
    X_km = zeros(d,N);
    for i = 1:N
        X_km(:,i) = xnew{i}(:,k-1);
    end
    eps_vec = zeros(N,1);
    for i = 1 : N
        if i==1
            idx_prev = N;
        else
            idx_prev = i-1;
        end
        dij = norm(X_km(:,i) - X_km(:,idx_prev));
        eps_vec(i) = phi(dij);
    end
    epsilon = beta*min(eps_vec);
    for i = 1:N
        for j = 1:N
            if j==i, Phi(i,j) = 0; continue; end
            dij = norm(X_km(:,i) - X_km(:,j));
            Phi(i,j) = max(phi(dij), 0);
            Phitilde(i,j) = Phi(i,j) + epsilon*kmat(i,j);
        end
    end
    Phitilde = Phitilde/N;
    for i = 1:N
        xi = X_km(:,i);
        Si = zeros(d,1);
        for j = 1:N
            if j==i, continue; end
            aij = Phitilde(i,j);
            Si  = Si + aij * (X_km(:,j) - xi);
        end
        u{i}(:,k-1) = -lam*(xnew{i}(:,k-1)-x_mean(:,k-1))-Si;
        xnew{i}(:,k) = xnew{i}(:,k-1) + ht * (Si + u{i}(:,k-1));
        X_k(:,i) = xnew{i}(:,k);
    end
    x_mean(:,k) = mean(X_k, 2);
    Xt0 = 0;
    for i = 1:N
        Xt0 = Xt0 + norm(xnew{i}(:,k) - x_mean(:,k))^2;
    end
    Xt0 = Xt0/(N^2);
    X_t(k) = Xt0;
end
tspan = ht * (0:k-1);
tf = ht*k;
nt = length(tspan);

% Compute mean value
mediax = zeros(N, k);
mediau = zeros(N, k-1);
for i = 1:N
    for j = 1:k
        mediax(i,j) = mean(xnew{i}(:,j));
    end
    for j = 1:k-1
        mediau(i,j) = mean(u{i}(:,j));
    end
end

% Plots
figure
hold on
for i = 1:N
    plot(tspan, mediax(i,:), 'LineWidth', 1.5)
end
xlim([0 tf])
ylim([0 8])
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
xlabel('Time')
ylabel('Opinions')
title(['Z-controlled Opinion Dynamics — \alpha = ', num2str(alpha)])
box on

figure
hold on
for i = 1:N
    plot(tspan(1:end-1), mediau(i,:), 'LineWidth', 1.5)
end
xlim([0 tf])
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
xlabel('Time')
ylabel('Controls ')
title(['Control Dynamics – \alpha = ', num2str(alpha)])
box on

figure
semilogy(tspan, X_t(1:k), 'LineWidth', 2)
xlim([0 tf])
xlabel('Time')
ylabel('\Gamma(t)')
title(['Consensus Parameter – \alpha = ', num2str(alpha)])
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold')
box on
