function [x, out] = PSG_x(f_g_handle,opts)
% PSG_x is a code written by Xiantao Xiao
% for solving stochastic QCQP problem with PSG proposed in the following paper: 
% Xiao X (2019) Penalized stochastic gradient methods for 
% stochastic convex optimization with expectation constraints,
% optimization-online.

% Version: 2022/07/09
% 
fprintf('\n ----------------------PSG begins-------------------\n');

n           = f_g_handle.n;
p           = f_g_handle.p;



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
 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x           = opts.x0;

record      = opts.record;
itPrint     = opts.itPrint;

obj_bench   = opts.obj_bench;



maxit       = opts.maxit;

tau         = opts.tau;

ssize_a     = opts.ssize_a;


%--------------------------------------------------------------------------
if record >= 1
    % set up print format

    stra = ['%6s', '%13s','%13s', '%12s', '%12s', '\n', ...
        repmat( '-', 1, 56 ), '\n'];
    str_head = sprintf(stra, ...
        'iter', 'obj', 'cons', 'obj-bench', 'time');
    str_num = ('%4d |  %+2.4e  %+2.4e %+2.4e %+2.4e \n');
end
%--------------------------------------------------------------------------

% prepare trace in output
if opts.trace
    [trace.obj, trace.cons,  trace.time, ... 
        ] = deal(zeros(maxit,1));
end



eval_time   = 0;


% step size
alpha0 = tau; beta0 = 1; gamma0 = tau;
vareps = 0.001;
if ssize_a
%a = 3/4 + vareps; b = 0; c = 1/2 + vareps;
a = 1/2 + 2*vareps; b = 0; c = 1/2 + vareps;
else
%a = 7/8 + vareps; b = 1/2 + vareps; c = 3/4 + vareps;
a = 3/4 + 2*vareps; b = 1/2 + vareps; c = 3/4 + vareps;
end

t = zeros(p,1);
x_old = x;
step_min = 1e-1 * tau;
step_bound = opts.step_bound;


%--------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:maxit
    
    
       % calculate obj and cons
       if record >= 1
                     time_temp       = toc;
                     obj             = f_g_handle.get_obj(x);
                     cons            = max(f_g_handle.get_cons(x)); 
                     eval_time       = eval_time + toc - time_temp;
       end
        % print the information
        if record>=1 && (iter == 1 || iter==maxit || mod(iter,itPrint)==0)
            if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit 
                    fprintf('\n%s\n', str_head);
            end
           fprintf(str_num, iter, obj, cons, obj - obj_bench, eval_time);
        end 
   
        % step size set up
        if step_bound
            alpha           = max(alpha0 / ceil(iter/10)^a,step_min/ceil(maxit/100)^a);
            beta            = max(beta0 / ceil(max(iter-1,1)/10)^b,step_min/ceil(maxit/100)^b);
            gamma           = max(gamma0 / ceil(iter/10)^c,step_min/ceil(maxit/100)^c);
        else
            alpha           = alpha0 / ceil(iter/1)^a;
            beta            = beta0 / ceil(max(iter-1,1)/1)^b;
            gamma           = gamma0 / ceil(iter/1)^c;           
        end
        


        % sampling and evaluation of stochastic gradient of f
        
        v_f             = f_g_handle.get_f_grad(x);



        % sampling, evaluation of t and stochastic gradient of g  
        % 
        if iter > 1 
            w = x_old + (x - x_old)/beta; 
        else
            w = x;
        end
        g_s             = f_g_handle.get_g_grad(w);
        t               = (1 - beta) * t + beta * g_s;
        
        % update iterate
        if iter >= 1
        y               = x - alpha * v_f;
        else
            y = x;
        end
        x_old = x;
        if t > eps
           [~,v_g]          = f_g_handle.get_f_grad(x);
           x            = y - gamma  * v_g*t; 
         
        else
           x            = y; 
        end
        x = f_g_handle.proj(x); 
   
        % save information for graphic output
        if record && opts.trace 
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



