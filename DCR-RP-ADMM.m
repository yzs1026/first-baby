# f%% 计及电能共享的基于非对称纳什谈判的多微网运行优化策略
clc
clear

%% ADMM参数设置
maxIter = 100;        % 最大迭代次数
tol = 1e-4;          % 收敛容差
rho = 1.0;           % ADMM惩罚参数
alpha = 1.0;         % 初始松弛参数
alpha_min = 0.8;     % 最小松弛参数
alpha_max = 1.2;     % 最大松弛参数
window_size = 5;     % 残差预测窗口大小

% 拉格朗日乘子初始化
lambda_e_12 = zeros(1,24); 
lambda_e_13 = zeros(1,24);
lambda_q_12 = zeros(1,24);
lambda_q_13 = zeros(1,24);
lambda_g_12 = zeros(1,24);
lambda_g_13 = zeros(1,24);

lambda_e_21 = zeros(1,24);
lambda_e_23 = zeros(1,24);
lambda_q_21 = zeros(1,24);
lambda_q_23 = zeros(1,24);
lambda_g_21 = zeros(1,24);
lambda_g_23 = zeros(1,24);

lambda_e_31 = zeros(1,24);
lambda_e_32 = zeros(1,24);
lambda_q_31 = zeros(1,24);
lambda_q_32 = zeros(1,24);
lambda_g_31 = zeros(1,24);
lambda_g_32 = zeros(1,24);

% 交易量记录矩阵初始化
P_12 = zeros(maxIter+1,24); P_21 = zeros(maxIter+1,24);
P_13 = zeros(maxIter+1,24); P_31 = zeros(maxIter+1,24);
P_23 = zeros(maxIter+1,24); P_32 = zeros(maxIter+1,24);

Q_12 = zeros(maxIter+1,24); Q_21 = zeros(maxIter+1,24);
Q_13 = zeros(maxIter+1,24); Q_31 = zeros(maxIter+1,24);
Q_23 = zeros(maxIter+1,24); Q_32 = zeros(maxIter+1,24);

G_12 = zeros(maxIter+1,24); G_21 = zeros(maxIter+1,24);
G_13 = zeros(maxIter+1,24); G_31 = zeros(maxIter+1,24);
G_23 = zeros(maxIter+1,24); G_32 = zeros(maxIter+1,24);

% 增加目标函数值存储
Obj_MG1 = zeros(1, maxIter);
Obj_MG2 = zeros(1, maxIter);
Obj_MG3 = zeros(1, maxIter);

% 残差和松弛参数历史记录
residual_history = zeros(maxIter, 1);
alpha_history = zeros(maxIter, 1);

%% 主迭代循环
iter = 1;
while 1
    if iter == maxIter
        disp('达到最大迭代次数，算法停止');
        break;
    end
    
    % 1. 更新各微网变量
    if iter == 1
        % 第一次迭代的特殊处理
        [P_12(2,:), P_13(2,:), Q_12(2,:), Q_13(2,:), G_12(2,:), Obj_MG1(iter), various_MG1] = ...
            Fun_MG1(P_21(iter,:), P_31(iter,:), Q_21(iter,:), Q_31(iter,:), G_21(iter,:), ...
            lambda_e_12, lambda_e_13, lambda_q_12, lambda_q_13, lambda_g_12);
        
        [P_21(2,:), P_23(2,:), Q_21(2,:), Q_23(2,:), G_21(2,:), Obj_MG2(iter), various_MG2] = ...
            Fun_MG2(P_12(2,:), P_32(iter,:), Q_12(2,:), Q_32(iter,:), G_12(2,:), ...
            lambda_e_21, lambda_e_23, lambda_q_21, lambda_q_23, lambda_g_21);
        
        [P_31(2,:), P_32(2,:), Q_31(2,:), Q_32(2,:), Obj_MG3(iter), various_MG3] = ...
            Fun_MG3(P_13(2,:), P_23(2,:), Q_13(2,:), Q_23(2,:), ...
            lambda_e_31, lambda_e_32, lambda_q_31, lambda_q_32);
    else
        % 使用动态松弛参数进行变量更新
        [P_12(iter+1,:), P_13(iter+1,:), Q_12(iter+1,:), Q_13(iter+1,:), G_12(iter+1,:), Obj_MG1(iter), various_MG1] = ...
            Fun_MG1(P_21(iter,:), P_31(iter,:), Q_21(iter,:), Q_31(iter,:), G_21(iter,:), ...
            lambda_e_12, lambda_e_13, lambda_q_12, lambda_q_13, lambda_g_12);
        
        % 应用动态松弛
        P_12(iter+1,:) = alpha * P_12(iter+1,:) + (1-alpha) * P_12(iter,:);
        P_13(iter+1,:) = alpha * P_13(iter+1,:) + (1-alpha) * P_13(iter,:);
        Q_12(iter+1,:) = alpha * Q_12(iter+1,:) + (1-alpha) * Q_12(iter,:);
        Q_13(iter+1,:) = alpha * Q_13(iter+1,:) + (1-alpha) * Q_13(iter,:);
        G_12(iter+1,:) = alpha * G_12(iter+1,:) + (1-alpha) * G_12(iter,:);
        
        [P_21(iter+1,:), P_23(iter+1,:), Q_21(iter+1,:), Q_23(iter+1,:), G_21(iter+1,:), Obj_MG2(iter), various_MG2] = ...
            Fun_MG2(P_12(iter+1,:), P_32(iter,:), Q_12(iter+1,:), Q_32(iter,:), G_12(iter+1,:), ...
            lambda_e_21, lambda_e_23, lambda_q_21, lambda_q_23, lambda_g_21);
        
        [P_31(iter+1,:), P_32(iter+1,:), Q_31(iter+1,:), Q_32(iter+1,:), Obj_MG3(iter), various_MG3] = ...
            Fun_MG3(P_13(iter+1,:), P_23(iter+1,:), Q_13(iter+1,:), Q_23(iter+1,:), ...
            lambda_e_31, lambda_e_32, lambda_q_31, lambda_q_32);
    end
    
    % 2. 计算原始残差和对偶残差
    primal_res = norm([P_12(iter+1,:)-P_12(iter,:), P_13(iter+1,:)-P_13(iter,:), ...
        Q_12(iter+1,:)-Q_12(iter,:), Q_13(iter+1,:)-Q_13(iter,:), ...
        G_12(iter+1,:)-G_12(iter,:)]);
    
    dual_res = rho * norm([P_21(iter+1,:)-P_21(iter,:), P_31(iter+1,:)-P_31(iter,:), ...
        Q_21(iter+1,:)-Q_21(iter,:), Q_31(iter+1,:)-Q_31(iter,:), ...
        G_21(iter+1,:)-G_21(iter,:)]);
    
    % 3. 更新松弛参数
    alpha = update_relaxation(primal_res, dual_res, alpha, alpha_min, alpha_max);
    alpha_history(iter) = alpha;
    
    % 4. 更新拉格朗日乘子
    lambda_e_12 = lambda_e_12 + rho * alpha * (P_12(iter+1,:) + P_21(iter+1,:));
    lambda_e_13 = lambda_e_13 + rho * alpha * (P_13(iter+1,:) + P_31(iter+1,:));
    lambda_q_12 = lambda_q_12 + rho * alpha * (Q_12(iter+1,:) + Q_21(iter+1,:));
    lambda_q_13 = lambda_q_13 + rho * alpha * (Q_13(iter+1,:) + Q_31(iter+1,:));
    lambda_g_12 = lambda_g_12 + rho * alpha * (G_12(iter+1,:) + G_21(iter+1,:));
    
    % 5. 记录历史数据
    residual_history(iter) = primal_res + dual_res;
    
    % 6. 检查收敛性和提前终止条件
    if primal_res < tol && dual_res < tol
        disp(['算法在第 ', num2str(iter), ' 次迭代收敛']);
        break;
    end
    
    % 使用残差预测判断是否提前终止
    if predict_convergence(residual_history(1:iter), window_size, tol)
        disp(['基于残差预测在第 ', num2str(iter), ' 次迭代提前终止']);
        break;
    end
    
    iter = iter + 1;
end

%% 结果可视化
figure
subplot(2,1,1)
semilogy(residual_history(1:iter), 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('残差');
title('ADMM算法收敛性能');

subplot(2,1,2)
plot(1:iter, alpha_history(1:iter), 'LineWidth', 1.5);
xlabel('迭代次数');
ylabel('松弛参数');
title('动态松弛参数变化');

%% 保存结果
save('results_optimized.mat');

%% 辅助函数定义
function alpha_new = update_relaxation(primal_res, dual_res, alpha_old, alpha_min, alpha_max)
    ratio = primal_res / (dual_res + 1e-10);
    if ratio > 1.2
        alpha_new = min(alpha_max, alpha_old * 1.1);
    elseif ratio < 0.8
        alpha_new = max(alpha_min, alpha_old * 0.9);
    else
        alpha_new = alpha_old;
    end
end

function [should_stop] = predict_convergence(residual_history, window_size, tol)
    if length(residual_history) < window_size
        should_stop = false;
        return;
    end
    
    y = residual_history(end-window_size+1:end);
    x = (1:window_size)';
    p = polyfit(x, y, 1);
    future_steps = window_size+(1:5);
    predicted_residuals = polyval(p, future_steps);
    should_stop = all(predicted_residuals < 2*tol);
endirst-baby
