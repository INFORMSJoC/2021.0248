% This code amis to illustrate the performance comparison 
% between   PSG, YNW, APriD and SLPMM
% for solving stochastic quadratically constrained  quadratic programs (QCQP).
% More detailed description on this experiment can be found at Section 5.3
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
% ALGORITHMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
maxit = 1000;
itPrint = 100;
% SLPMM
algor1.name       = 'SLPMM';
algor1.opts       = struct('maxit',maxit,'itPrint',itPrint,'trace',trace,'record',record);

% APriD
algor2.name       = 'APriD';
algor2.opts       = struct('maxit',maxit,'itPrint',itPrint,'trace',trace,'record',record,'ssize_a',0);


% PSG
algor3.name       = 'PSG';
algor3.opts       = struct('maxit',maxit,'itPrint',itPrint,'trace',trace,'record',record,'ssize_a',0);

% YNW
algor4.name       = 'YNW';
algor4.opts       = struct('maxit',maxit,'itPrint',itPrint,'trace',trace,'record',record,'ssize_a',0);




%algorithms        = {algor1};
algorithms        = {algor2,algor3,algor4,algor1};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of test repetitions
nr_test           = 1;



% parameters
n = 100; % dimension
p = 5; % number of quadratical constraints
R = 2;

test_problem      = 'QCQP';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOPS // TEST FRAMEWORK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_hat = -R/sqrt(n) + (2*R/sqrt(n))*rand(n,1);
    
    % build f and g function
    batch = 10; % mini-batch size
        f_g_handle = make_f_g_qcqp(n,p,R,x_hat,batch);

    % generate benchmark point by fmincon
    x_initial = ones(n,1)*sqrt(R)/sqrt(n);

    obj_bench = 0.5*norm(x_hat)^2;



 
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
        


                temp_nr_test        = nr_test;        
                
                            

        algorithms{j}.opts.x0                       = x_initial;
        algorithms{j}.opts.obj_bench            = obj_bench;


        
        % save algorithmic information
        if trace
            results{j}.name     = algorithms{j}.name;
        end

        %------------------------------------------------------------------
        % Loop for number of tests and application of algorithms
        %------------------------------------------------------------------
        
        for k = 1:temp_nr_test
            
 

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
                results{j}.time    = out.trace.time;       
            elseif trace
                results{j}.obj =  results{j}.obj + out.trace.obj; results{j}.cons = results{j}.cons + out.trace.cons;
                results{j}.time    = results{j}.time + out.trace.time;   
            end
        end
        
        % Compute average of the results
        if trace
            results{j}.obj = results{j}.obj/temp_nr_test;  results{j}.cons = results{j}.cons/temp_nr_test; 
            results{j}.time    = results{j}.time/temp_nr_test; 
        end
                
    end
% %   Draw figures
if trace 
    dataset_name    = 'random';
    draw_figures_results(dataset_name,test_problem,results,nr_test,obj_bench);
end




