function [x, out] = SLPMM_x(f_g_handle,opts)
% SLPMM_x is a code written by Xiantao Xiao
% for solving NP classificatio problem with stochastic linearized proximal
% method of multipliers (SLPMM),
% based on the paper: 
% L. Zhang, Y. Zhang, J. Wu and X. Xiao. Solving Stochastic Optimization 
% with Expectation Constraints Efficiently by a Stochastic Augmented
% Lagrangian-Type Algorithm. https://arxiv.org/abs/2106.11577v3.

% Version: 2022/07/09
% 
fprintf('\n ----------------------SLPMM begins-------------------\n');

N0          = size(f_g_handle.A0,1);
N1          = size(f_g_handle.A1,1);
p           = f_g_handle.p;
N           = N0 + N1;

tic;

%%-------------------------------------------------------------------------
if nargin < 2; opts = []; end
 if ~isfield(opts,'x0');       opts.x0  = zeros(p,1);   end
 if ~isfield(opts,'tol');      opts.tol = 1e-6;     end
 if ~isfield(opts,'record');   opts.record = 1;     end
 if ~isfield(opts,'itPrint');  opts.itPrint = 10;     end
 if ~isfield(opts,'eps');      opts.eps = 1e-2;     end 
 if ~isfield(opts,'tau');      opts.tau = 1e-1;     end 
 if ~isfield(opts,'ssize_a'); opts.ssize_a = 0; end
  if ~isfield(opts,'step_bound'); opts.step_bound = 1; end
% 
 if ~isfield(opts,'optim_obj');  opts.optim_obj = 0;     end
 if ~isfield(opts,'maxit');      opts.maxit = 500;       end
 
 if ~isfield(opts,'ssize_f');  opts.ssize_f = max(min(N0,10),floor(N0/100));   end
 if ~isfield(opts,'ssize_g');  opts.ssize_g = max(min(N1,10),floor(N1/100));   end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x           = opts.x0;

record      = opts.record;
itPrint     = opts.itPrint;

obj_bench   = opts.obj_bench;


ssize_f     = opts.ssize_f;
ssize_g     = opts.ssize_g;

maxit       = opts.maxit;

%tau         = opts.tau;

ssize_a     = opts.ssize_a;


%--------------------------------------------------------------------------
if record >= 1
    % set up print format

    stra = ['%6s', '%13s','%13s', '%12s', '%12s', '\n', ...
        repmat( '-', 1, 56 ), '\n'];
    str_head = sprintf(stra, ...
        'iter', 'obj', 'cons', 'obj-bench', 'cons_s');
    str_num = ('%4d |  %+2.4e  %+2.4e %+2.4e %+2.4e \n');
end
%--------------------------------------------------------------------------

% prepare trace in output
if opts.trace
    [trace.obj, trace.cons, trace.res, trace.time, ... 
        trace.epoch] = deal(zeros(maxit,1));
end


% prepare sampling
resample_f      = make_resample(N0);
resample_g      = make_resample(N1);
sample_time     = 0;
eval_time   = 0;

nr_epoch        = 0;

% step size
alpha = maxit^(1/2);
sigma = maxit^(-1/2);


%t = 0;
lambda = 0; % initial multiplier
g_s      = 0;

%--------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:maxit
    
    
       % calculate obj and cons
       if record >= 1
                     time_temp       = toc;
                     obj             = f_g_handle.get_f(x);
                     cons            = f_g_handle.get_g(x); 
                     eval_time       = eval_time + toc - time_temp;
       end
        % print the information
        if record>=1 && (iter == 1 || iter==maxit || mod(iter,itPrint)==0)
            if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit 
                    fprintf('\n%s\n', str_head);
            end
           fprintf(str_num, iter, obj, cons, obj - obj_bench, g_s);
        end 
   

        
        % adaptive sample size 
        if ssize_a
            if iter ==1; ssize_f_0 = ssize_f; ssize_g_0 = ssize_g; end
           ssize_f = min(ssize_f_0 * iter, N0);
           ssize_g = min(ssize_g_0 * iter, N1);
        end

        % sampling and evaluation of stochastic gradient of f
        [~,start_pos_f] = resample_f(ssize_f);
        time_temp       = toc;
        f_g_handle.resample_f(start_pos_f, ssize_f);
        sample_time     = toc - time_temp + sample_time;
        nr_epoch        = nr_epoch + ssize_f/N;
        
        v_f             = f_g_handle.get_s_grad_f(x);



        % sampling, evaluation of t and stochastic gradient of g  
        [~,start_pos_g] = resample_g(ssize_g);
        time_temp       = toc;
        f_g_handle.resample_g(start_pos_g, ssize_g);
        sample_time     = toc - time_temp + sample_time;
        nr_epoch        = nr_epoch + ssize_g/N;
        g_s             = f_g_handle.get_s_g(x);
        v_g          = f_g_handle.get_s_grad_g(x);

        
        % update iterate
        temp1 = sqrt(sigma/alpha); temp2 = sqrt(sigma*alpha);
        a = temp1*v_g; 
        b = lambda/temp2 + temp1*g_s - a'*x;
        c = v_f/alpha - x;
        
        x_old = x;
        if  b - a'*c < 0
            x = -c;
        else
            temp3 = b*a +c;
            x = -temp3 + a'*temp3*a/(1+a'*a);
        end
        
        % update multiplier
        temp4 = lambda + sigma*(g_s + v_g'*(x-x_old));
        lambda = max(0, temp4);
   
        % save information for graphic output
        if record && opts.trace 

            trace.epoch(iter)   = nr_epoch;
            trace.obj(iter)     = obj;
            trace.cons(iter)    = cons;
            trace.time(iter)    = toc - eval_time;

        end     
end %end outer loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if opts.trace 
    if record == 0; error('Set "record = 1!"'); end
    out.trace = trace; 
else
    out = []; 
end

end



