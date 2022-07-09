function [x, out] = APriD_x(f_g_handle,opts)
% APriD_x is a code written by Xiantao Xiao
% for solving stochastic QCQP problem with APriD proposed in the following paper: 
% Yan Y, Xu Y (2022) Adaptive primal-dual stochastic gradient method
% for expectation-constrained convex stochastic programs. 
% Math. Program. Comput. 14(2):319–363.

% Version: 2022/07/09
% 
fprintf('\n ----------------------APriD begins-------------------\n');

n           = f_g_handle.n;
p           = f_g_handle.p;


tic;

%%-------------------------------------------------------------------------
if nargin < 2; opts = []; end
 if ~isfield(opts,'x0');       opts.x0  = zeros(p,1);   end
 if ~isfield(opts,'tol');      opts.tol = 1e-6;     end
 if ~isfield(opts,'record');   opts.record = 1;     end
 if ~isfield(opts,'itPrint');  opts.itPrint = 10;     end

% 
 if ~isfield(opts,'optim_obj');  opts.optim_obj = 0;     end
 if ~isfield(opts,'maxit');      opts.maxit = 500;       end
 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x           = opts.x0;

m        = zeros(n,1);
v         = zeros(n,1);
v_hat   = zeros(n,1);

record      = opts.record;
itPrint     = opts.itPrint;

obj_bench   = opts.obj_bench;


maxit       = opts.maxit;




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
alpha = 1/maxit^(1/2);
rho = 1/maxit^(1/2);

% parameters
beta1 = 0.9;
beta2 = 0.99;
theta = 10;

z = zeros(p,1); % initial multiplier


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
        [g_s,v_g]             = f_g_handle.get_g_grad(x);


        
        % update iterate
        u    = v_f + v_g*z;
        w   = g_s;
        m   = beta1*m + (1-beta1)*u;
        v    = beta2*v + (1-beta2)*u.^2/max(1,norm(u)^2/theta);
        v_hat = max(v,v_hat);
    
        x = x - alpha*m./(sqrt(v_hat)+1e-15); %% +1e-15 in case v_hat=0.
        x = f_g_handle.proj(x);

        
        % update multiplier
         z = max(z + rho*w,0);
   
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



