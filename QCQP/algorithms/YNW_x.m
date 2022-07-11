function [x, out] = YNW_x(f_g_handle,opts)
% YNW_x is a code written by Xiantao Xiao
% for solving stochastic QCQP problem with an algorithm proposed in the following paper: 
% Yu H, Neely MJ, Wei X (2017) Online convex optimization with stochastic constraints. 
% Advances in Neural Information Processing Systems, 1428–1438.

% Version: 2022/07/09
% 
fprintf('\n ----------------------YNW begins-------------------\n');

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
alpha0 = tau; gamma0 = tau;
a = 1/2; c = 1;
alpha           = alpha0 / maxit^a;
gamma           = gamma0 / maxit^c;           




t = zeros(p,1);
%x_old = x;



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
   
     

        % sampling and evaluation of stochastic gradient of f
        
        v_f             = f_g_handle.get_f_grad(x);



        % sampling, evaluation of t and stochastic gradient of g  
        
        % update iterate
        if iter >= 1
        y               = x - alpha * v_f;
        else
            y = x;
        end
        x_old = x;
        [g_s,v_g]             = f_g_handle.get_g_grad(x);
        if t > eps
           x            = y - gamma  * v_g * t;          
        else
           x            = y; 
        end
        x = f_g_handle.proj(x); 
        
        % update t
        t               = t + g_s + v_g'*(x-x_old);
        t               = max(t,0);
        
   
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



