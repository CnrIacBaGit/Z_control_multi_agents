% Cucker-Smale generalized 3rd order
% Direct Z-control
clear all
close all
clc

% Initial conditions
% load('motiv_exd2N10_dati.mat','x')
load('dati_d150_N150.mat', 'x')

% Parameters
d = 150;           % agent dimension
N = 150;          % number of agents
lam = 0.1; %0.3;
ht = 0.01;
t0 = 0;
tol_consensus = 1e-15; %threshold for consensus

% Interaction function
beta = 1;
K = 1;
phi = @(s) K * (1 + s^2)^(-beta);

% Initialization
xnew = cell(N,1);
vnew = cell(N,1);
anew = cell(N,1);
u = cell(N,1);
for i = 1:N
    xnew{i} = zeros(d,1e5);
    vnew{i} = zeros(d,1e5);
    anew{i} = zeros(d,1e5);
    u{i} = zeros(d,1e5);
    xnew{i}(:,1) = x{i}(:,1);
    vnew{i}(:,1) = x{i}(:,1);
    anew{i}(:,1) = x{i}(:,1); 
end

X_k = zeros(d,N);
V_k = zeros(d,N);
A_k = zeros(d,N);
for i = 1:N
    X_k(:,i) = xnew{i}(:,1);
    V_k(:,i) = vnew{i}(:,1);
    A_k(:,i) = anew{i}(:,1);
end
x_mean(:,1) = mean(X_k,2);
v_mean(:,1) = mean(V_k,2);
a_mean(:,1) = mean(A_k,2);

% Initial consensus parameter
Xt = 0;
for i = 1:N
    Xt = Xt + norm(anew{i}(:,1) - a_mean(:,1))^2;
end
Xt = Xt / (N^2);
X_t(1) = Xt;

k = 1;
while Xt > tol_consensus
    Xt
    k = k + 1;
    for i = 1:N
        sum_interaction = zeros(d,1);
        for j = 1:N
            if j ~= i
                dist = norm(xnew{j}(:,k-1) - xnew{i}(:,k-1));
                sum_interaction = sum_interaction + ...
                    phi(dist) * (anew{j}(:,k-1) - anew{i}(:,k-1));
            end
        end
        u{i}(:,k-1) = -lam * (anew{i}(:,k-1) - a_mean(:,k-1));
        xnew{i}(:,k) = xnew{i}(:,k-1) + ht * vnew{i}(:,k-1);
        vnew{i}(:,k) = vnew{i}(:,k-1) + ht * anew{i}(:,k-1);
        anew{i}(:,k) = anew{i}(:,k-1) + ht * (sum_interaction/N + u{i}(:,k-1));
        X_k(:,i) = xnew{i}(:,k);
        V_k(:,i) = vnew{i}(:,k);
        A_k(:,i) = anew{i}(:,k);
    end
    x_mean(:,k) = mean(X_k,2);
    v_mean(:,k) = mean(V_k,2);
    a_mean(:,k) = mean(A_k,2);
    Xt = 0;
    for i = 1:N
        Xt = Xt + norm(anew{i}(:,k) - a_mean(:,k))^2;
    end
    Xt = Xt / (N^2);
    X_t(k) = Xt;
end

tspan = (0:k-1)*ht;
nt = length(tspan);
tf = k*ht;
for i = 1:N
    for k = 1:nt
        mediax(i,k) = mean(xnew{i}(:,k));
        mediav(i,k) = mean(vnew{i}(:,k));
        mediaa(i,k) = mean(anew{i}(:,k));
        mediau(i,k) = mean(u{i}(:,k));
    end
end

% Plot
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
xlabel('Time'); ylabel('Control Input')
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