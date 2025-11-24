% Cucker-Smale 2nd order model
% Dynamics without control
clear all
close all
clc

% Initial conditions
load('motiv_exd2N10_dati.mat','x')
load('dati_d2_N150.mat','x')

% Parameters
d = 2;           % agent dimension
N = 150;          % number of agents
ht = 0.01;
tf = 10000;
t0 = 0;
tspan = t0:ht:tf;
nt = length(tspan);

% Interaction function
beta = 1; 
K = 1;
phi = @(s) K * (1 + s^2)^(-beta);

% Initialization
xnew = cell(N,1);
vnew = cell(N,1);
for i = 1:N
    xnew{i} = zeros(d, nt);
    vnew{i} = zeros(d, nt);
    xnew{i}(:,1) = x{i}(:,1);
    vnew{i}(:,1) = x{i}(:,1); % inizializzazione arbitraria
end


x_mean(:,1) = mean(xnew{i}(:,1),2);
v_mean(:,1) = mean(vnew{i}(:,1),2);

% Initial consensus parameter
Xt = zeros(1, nt);
Xt(1) = 0;
for i = 1:N
    Xt(1) = Xt(1) + norm(vnew{i}(:,1) - v_mean(:,1))^2;
end
Xt(1) = Xt(1) / (N^2);

% Evolution dynamics
for k = 2:nt
    Xkm1 = zeros(d,N);
    Vkm1 = zeros(d,N);
    for i = 1:N
        Xkm1(:,i) = xnew{i}(:,k-1);
        Vkm1(:,i) = vnew{i}(:,k-1);
    end
    Phi = zeros(N,N);
    for i = 1:N
        for j = 1:N
            if j == i, continue; end
            dij = norm(Xkm1(:,i) - Xkm1(:,j));
            Phi(i,j) = phi(dij) / N;
        end
    end
    for i = 1:N
        xi = Xkm1(:,i);
        vi = Vkm1(:,i);
        Si = zeros(d,1);
        for j = 1:N
            if j==i, continue; end
            aij = Phi(i,j) / N;
            Si  = Si + aij * (Vkm1(:,j) - vi);
        end
        vnew{i}(:,k) = vnew{i}(:,k-1) + ht * Si;
        xnew{i}(:,k) = xnew{i}(:,k-1) + ht * vnew{i}(:,k-1);
    end
    x_mean(:,k) = mean(xnew{i}(:,k),2);
    v_mean(:,k) = mean(vnew{i}(:,k),2);
    Xt(k) = 0;
    for i = 1:N
        Xt(k) = Xt(k) + norm(vnew{i}(:,k) - v_mean(:,k))^2;
    end
    Xt(k) = Xt(k) / (2 * N^2);
end

% Compute mean value
mediax = zeros(N, nt);
mediav = zeros(N, nt);
for i = 1:N
    for k = 1:nt
        mediax(i,k) = mean(xnew{i}(:,k));
        mediav(i,k) = mean(vnew{i}(:,k));
    end
end

% Plots
figure
hold on
for i = 1:N
    plot(tspan, mediax(i,:), 'LineWidth', 1.5)
end
xlabel('Time')
ylabel('Mean Position')
title('Trajectories')
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
box on

figure
hold on
for i = 1:N
    plot(tspan, mediav(i,:), 'LineWidth', 1.5)
end
xlabel('Time')
ylabel('Mean Velocity')
title('Velocities')
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold', 'LineWidth', 1.5)
box on

figure
semilogy(tspan, Xt, 'LineWidth', 2)
xlabel('Time')
ylabel('\Gamma(t)')
title('Consensus')
set(gca, 'XScale', 'log', 'FontSize', 14, 'FontWeight', 'bold')
box on