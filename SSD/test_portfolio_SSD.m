% This code amis to illustrate the performance comparison 
% between   PSG, YNW, APriD, PALEM and SLPMM
% for solving Second-order stochastic dominance (SSD) constrained portfolio optimization.
% More detailed description on this experiment can be found at Section 5.4
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
datasets = {'DAX_26_3046', 'DowJones_29_3020', 'SP100_90_3020','DowJones_76_30000'};
%datasets = { 'DowJones_76_30000'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ALGORITHMS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SLPMM
algor1.name       = 'SLPMM';
algor1.print_name = 'SLPMM';
algor1.opts       = struct('maxit',3000,'itPrint',100,'trace',trace,'record',record,'ssize_a',0,'print_cons',0,'s_cons',1);

% YNWs with standard process (update $t$ for all N constraints )
algor2.name       = 'YNW';
algor2.print_name = 'YNW';
algor2.opts       = struct('maxit',3000,'itPrint',100,'trace',trace,'record',record,'ssize_a',0,'s_cons',1);

% PSGs with standard process (update $t$ for all N constraints )
algor3.name       = 'PSG';
algor3.print_name = 'PSG';
algor3.opts       = struct('maxit',500,'itPrint',50,'trace',trace,'record',record,'ssize_a',0,'s_cons',1);

% PALEM
algor4.name       = 'PALEM';
algor4.print_name = 'PALEM';
algor4.opts       = struct('maxit',30,'itPrint',10,'trace',trace,'record',record);

% APriD
algor5.name       = 'APriD';
algor5.print_name = 'APriD';
algor5.opts       = struct('maxit',1000,'itPrint',100,'trace',trace,'record',record);

%algorithms        = {algor1};
algorithms        = {algor5,algor2,algor3,algor4,algor1};
test_problem = 'SSD';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TEST SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% number of test repetitions
nr_test           = 1;

   % generate and save  output
    if trace
        results = cell(length(algorithms),1);
    end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOPS // TEST FRAMEWORK
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:length(datasets)
    
    % load dataset
    dataset_name    = datasets{i};

    R_matrix  = load(strcat('./data/data_problem_',dataset_name,'/','matrix_sde.txt'));
    Y_bench   = load(strcat('./data/data_problem_',dataset_name,'/','vector_benchmark_sd.txt'));
    
    switch dataset_name
           case 'DAX_26_3046'
                        ub                          = 0.2;
                        obj_bench                   = -0.000657006;
                        eps                         = 1e-4;
           case 'DowJones_29_3020'
                        ub                          = 0.2;
                        obj_bench                   = -0.000334689;
           case 'SP100_90_3020'
                        ub                          = 0.2;
                        obj_bench                   = -0.000865278;
           case 'DowJones_76_30000'
                        ub                          = 0.2;
                        obj_bench                   = -0.018652555968;
     end    
    

    % build f and g function
     f_g_handle = make_f_g_empirical(R_matrix,Y_bench,ub);   
     N               = f_g_handle.scenarios;



     %x_initial = ub * ones(size(R_matrix,2),1);
     x_initial = zeros(size(R_matrix,2),1);

    %----------------------------------------------------------------------
    % Loop for different algorithms
    %----------------------------------------------------------------------
    for j = 1:length(algorithms) 
        
        %------------------------------------------------------------------
        % print algorithmic information and set specific parameters
        %------------------------------------------------------------------
        

                switch dataset_name
                    case 'DAX_26_3046'
                        algorithms{j}.opts.alpha0     = 200;
                        algorithms{j}.opts.beta0      = 1;
                        algorithms{j}.opts.gamma0  = 1;
                        algorithms{j}.opts.alpha       = 1;
                        algorithms{j}.opts.sigma      = 1;
                    case 'DowJones_29_3020'
                        algorithms{j}.opts.alpha0     = 500;
                        algorithms{j}.opts.beta0      = 1;
                        algorithms{j}.opts.gamma0  = 1;
                        algorithms{j}.opts.alpha       = 1;
                        algorithms{j}.opts.sigma      = 1;                        
                    case 'SP100_90_3020'
                       algorithms{j}.opts.alpha0      = 30;
                        algorithms{j}.opts.beta0      = 1;
                        algorithms{j}.opts.gamma0  = 1;
                        algorithms{j}.opts.alpha      = 1;
                        algorithms{j}.opts.sigma      = 1;                        
                    case 'DowJones_76_30000'
                       algorithms{j}.opts.alpha0      = 1;
                        algorithms{j}.opts.beta0      = 1;
                        algorithms{j}.opts.gamma0  = 1;
                        algorithms{j}.opts.alpha      = N^(1/2);
                        algorithms{j}.opts.sigma      = N^(-1/4);
                end
                temp_nr_test        = nr_test;             


        
        algorithms{j}.opts.x0                       = x_initial;
        algorithms{j}.opts.obj_bench                = obj_bench;

        algorithms{j}.opts.ssize_f                  = max(min(N,50),floor(N/1000)); 
        algorithms{j}.opts.ssize_g                  = max(min(N,50),floor(N/1000)); 
        algorithms{j}.opts.ssize_I                  = max(min(N,50),floor(N/1000)); 
        algorithms{j}.opts.eps                      = eps;      
%        algorithms{j}.opts.print_cons               = print_cons;
        

        % save algorithmic information
        if trace
            results{j}.name     = algorithms{j}.print_name;
        end

        %------------------------------------------------------------------
        % Loop for number of tests and application of algorithms
        %------------------------------------------------------------------
        
        for k = 1:temp_nr_test
            
            % permute data
                f_g_handle.perm_f();
                f_g_handle.perm_g(); 
                f_g_handle.perm_I();

            switch algorithms{j}.name
               case 'SLPMM'
                    algorithms{j}.opts.ssize_a                  = 0; % adaptive sample size
                    [x,out] = SLPMM_SSD_x(f_g_handle,algorithms{j}.opts);
                case 'YNW'
                    algorithms{j}.opts.ssize_a                  = 0; % adaptive sample size
                    [x,out] = YNWs_std_x(f_g_handle,algorithms{j}.opts);
                case 'PSG'
                    algorithms{j}.opts.ssize_a                  = 0; % adaptive sample size
                    [x,out] = PSGs_std_x(f_g_handle,algorithms{j}.opts);
                case 'PALEM'
                    algorithms{j}.opts.ssize_a                  = 0; % adaptive sample size
                    [x,out] = PALEM_SSD_x(f_g_handle,algorithms{j}.opts);  
                case 'APriD'
                    algorithms{j}.opts.ssize_a                  = 0; % adaptive sample size
                    [x,out] = APriD_SSD_x(f_g_handle,algorithms{j}.opts);                
            end
            
            
            % save results for graphic output
            if trace && k == 1
                results{j}.obj0 = out.obj; results{j}.cons0 = out.cons;    results{j}.res0   = out.res;
                results{j}.time0    = out.time;   results{j}.epoch0 = out.epoch;       
                results{j}.obj = out.trace.obj; results{j}.cons = out.trace.cons;   
                results{j}.time    = out.trace.time; 
            elseif trace
                results{j}.obj0 =  results{j}.obj0 + out.obj; results{j}.cons0 = results{j}.cons0 + out.cons;
                results{j}.res0   = results{j}.res0 + out.res;
                results{j}.time0    = results{j}.time0 + out.time;   results{j}.epoch0 = results{j}.epoch0 + out.epoch;
                results{j}.obj =  results{j}.obj + out.trace.obj; results{j}.cons = results{j}.cons + out.trace.cons;
                results{j}.time    = results{j}.time + out.trace.time;
            end
        end
        
        % Compute average of the results
        if trace
            results{j}.obj0 = results{j}.obj0/temp_nr_test;  results{j}.cons0 = results{j}.cons0/temp_nr_test; 
            results{j}.res0 = results{j}.res0/temp_nr_test;
            results{j}.time0    = results{j}.time0/temp_nr_test; results{j}.epoch0 = results{j}.epoch0/temp_nr_test;
            results{j}.obj = results{j}.obj/temp_nr_test;  results{j}.cons = results{j}.cons/temp_nr_test; 
            results{j}.time    = results{j}.time/temp_nr_test;            
        end
   


    end
          % %   Draw figures
    if trace 
         dataset_name    = datasets{i};
        draw_figures_results(dataset_name,test_problem,results,nr_test,obj_bench);
    end
end


