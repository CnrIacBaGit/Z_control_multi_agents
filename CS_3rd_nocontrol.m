% Cucker-Smale generalized 3rd order
% Dynamics without control
clear all
close all
clc

% Initial conditions
load('motiv_exd2N10_dati.mat','x')

% Parameters
d = 2;           % agent dimension
N = 10;          % number of agents
ht = 0.01;
t0 = 0;
tf = 10000;
tspan = [t0:ht:tf];
nt = length(tspan);

% Interaction function
beta = 0.1;
K = 1;
phi = @(s) K * (1 + s^2)^(-beta);

% Initialization
xnew = cell(N,1);
vnew = cell(N,1);
anew = cell(N,1);
for i = 1:N
    xnew{i} = zeros(d,nt);
    vnew{i} = zeros(d,nt);
    anew{i} = zeros(d,nt);
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

for k = 2 : nt
    for i = 1:N
        sum_interaction = zeros(d,1);
        for j = 1:N
            if j ~= i
                dist = norm(xnew{j}(:,k-1) - xnew{i}(:,k-1));
                sum_interaction = sum_interaction + ...
                    phi(dist) * (anew{j}(:,k-1) - anew{i}(:,k-1));
            end
        end
        xnew{i}(:,k) = xnew{i}(:,k-1) + ht * vnew{i}(:,k-1);
        vnew{i}(:,k) = vnew{i}(:,k-1) + ht * anew{i}(:,k-1);
        anew{i}(:,k) = anew{i}(:,k-1) + ht * (sum_interaction/N);
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

for i = 1:N
    for k = 1:nt
        mediax(i,k) = mean(xnew{i}(:,k));
        mediav(i,k) = mean(vnew{i}(:,k));
        mediaa(i,k) = mean(anew{i}(:,k));
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
semilogy(tspan, X_t(1:k), 'LineWidth', 2)
xlabel('Time'); ylabel('\Gamma(t)')
title('Consensus')
xlim([0 tf])
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
box on