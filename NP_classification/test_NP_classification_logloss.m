% This code amis to illustrate the performance comparison 
% between  CSA, PSG, YNW, APriD and SLPMM
% for solving Neyman-Pearson classification problem.
% More detailed description on this experiment can be found at Section 5.2
% of the paper:
% % L. Zhang, Y. Zhang, J. Wu and X. Xiao. Solving Stochastic Optimization 
% with Expectation Constraints Efficiently by a Stochastic Augmented
% Lagrangian-Type Algorithm. https://arxiv.org/abs/2106.11577v3.
% To appear in INFORMS Journal on Computing.

% Version: 2022/07/09

clc; clear; close all; rng('default');
HOME = pwd;
addpath(genpath(HOME));

% save detailed information for graphic output
trace           = 1;
record          = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DATASETS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
datasets = {'gisette', 'MNIST',  'CINA'};
%datasets = { 'MNIST'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ALGORITHMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxit = 3000;
itPrint = 1000;
% SLPMM
algor1.name       = 'SLPMM';
algor1.opts       = struct('maxit',maxit,'itPrint',itPrint,'trace',trace,'record',record);


% CSA
algor2.name       = 'CSA';
algor2.opts       = struct('maxit',maxit,'itPrint',itPrint,'trace',trace,'record',record);

% PSG
algor3.name       = 'PSG';
algor3.opts       = struct('maxit',maxit,'itPrint',itPrint,'trace',trace,'record',record,'ssize_a',0);

% YNW
algor4.name       = 'YNW';
algor4.opts       = struct('maxit',maxit,'itPrint',itPrint,'trace',trace,'record',record,'ssize_a',0);

% APriD
algor5.name       = 'APriD';
algor5.opts       = struct('maxit',maxit,'itPrint',itPrint,'trace',trace,'record',record,'ssize_a',0);


%algorithms        = {algor1};
algorithms        = {algor5,algor2,algor3,algor4,algor1};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of test repetitions
nr_test           = 10;



% parameters
risk_level        = 1; % g(x) <= risk_level
%mu                = 0.01; % mu * ||x||_1

test_problem      = 'logloss';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOPS // TEST FRAMEWORK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:length(datasets)
    
    % load dataset
    dataset_name = datasets{i};

    load(strcat('./datasets/',dataset_name,'/',dataset_name,'_train_scale.mat'));
    load(strcat('./datasets/',dataset_name,'/',dataset_name,'_labels.mat'));
    
    A0 = A(b==1,:);    A1 = A(b==-1,:);  clear A  b;
    N0 = size(A0,1);    N1 = size(A1,1);
    
    % build f and g function
    switch test_problem
        case 'logloss'
        f_g_handle = make_f_g_logloss(A0,A1,risk_level);
        case 'hingeloss'
        f_g_handle = make_f_g_hingeloss(A0,A1,risk_level);    
    end

    
    % generate benchmark point by fmincon
    x_initial = zeros(size(A0,2),1);

    % fmincon
    problem.options =  optimoptions('fmincon','SpecifyObjectiveGradient',true,'SpecifyConstraintGradient',true,'Algorithm','interior-point','OptimalityTolerance',1e-2,'Display','iter');
    problem.solver = 'fmincon';
    problem.objective = @f_g_handle.get_obj;
    problem.nonlcon = @f_g_handle.get_cons;
    problem.x0 = x_initial;

    [x_bench,obj_bench] = fmincon(problem);


    



 
    % generate and save graphic output
    if trace
        results = cell(length(algorithms),1);
    end
    
    %----------------------------------------------------------------------
    % Loop for different algorithms
    %----------------------------------------------------------------------
    for j = 1:length(algorithms) 
        
        %------------------------------------------------------------------
        % print algorithmic information and set specific parameters
        %------------------------------------------------------------------
        

                switch dataset_name
                    case 'gisette'
                        algorithms{j}.opts.tau      = 0.05;
                    case 'MNIST'
                        algorithms{j}.opts.tau      = 0.1;
                    case 'CINA'
                        algorithms{j}.opts.tau      = 0.01;
                end
                temp_nr_test        = nr_test;        
                
                            

        algorithms{j}.opts.x0                       = x_initial;
        algorithms{j}.opts.obj_bench                = obj_bench;
%         algorithms{j}.opts.ssize_f                  = ceil(0.0001 * size(A0,1));
%         algorithms{j}.opts.ssize_g                  = ceil(0.0001 * size(A1,1));
        algorithms{j}.opts.ssize_f                  = 10;
        algorithms{j}.opts.ssize_g                  = 10;

        
        % save algorithmic information
        if trace
            results{j}.name     = algorithms{j}.name;
        end

        %------------------------------------------------------------------
        % Loop for number of tests and application of algorithms
        %------------------------------------------------------------------
        
        for k = 1:temp_nr_test
            
            % permute data
                f_g_handle.perm_f();
                f_g_handle.perm_g();    

            switch algorithms{j}.name
                case 'PSG'
                    [x,out] = PSG_x(f_g_handle,algorithms{j}.opts);
                case 'CSA'
                    [x,out] = CSA_x(f_g_handle,algorithms{j}.opts);
                case 'YNW'
                    [x,out] = YNW_x(f_g_handle,algorithms{j}.opts);
                case 'SLPMM'
                    [x,out] = SLPMM_x(f_g_handle,algorithms{j}.opts);
                case 'APriD'
                    [x,out] = APriD_x(f_g_handle,algorithms{j}.opts);
            end
            
            
            % save results for graphic output
            if trace && k == 1
                results{j}.obj = out.trace.obj; results{j}.cons = out.trace.cons;   
                results{j}.time    = out.trace.time;   results{j}.epoch = out.trace.epoch;       
            elseif trace
                results{j}.obj =  results{j}.obj + out.trace.obj; results{j}.cons = results{j}.cons + out.trace.cons;
                results{j}.time    = results{j}.time + out.trace.time;   results{j}.epoch = results{j}.epoch + out.trace.epoch;
            end
        end
        
        % Compute average of the results
        if trace
            results{j}.obj = results{j}.obj/temp_nr_test;  results{j}.cons = results{j}.cons/temp_nr_test; 
            results{j}.time    = results{j}.time/temp_nr_test; results{j}.epoch = results{j}.epoch/temp_nr_test;
        end
                
    end
% %   Draw figures
if trace 
    dataset_name    = datasets{i};
    draw_figures_results(dataset_name,test_problem,results,nr_test,obj_bench);
end
end



