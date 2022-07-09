function [x, out] = SLPMM_x(f_g_handle,opts)
% SLPMM_x is a code written by Xiantao Xiao
% for solving stochastic QCQP problem with stochastic linearized proximal
% method of multipliers (SLPMM),
% based on the paper: 
% L. Zhang, Y. Zhang, J. Wu and X. Xiao. Solving Stochastic Optimization 
% with Expectation Constraints Efficiently by a Stochastic Augmented
% Lagrangian-Type Algorithm. https://arxiv.org/abs/2106.11577v3.

% Version: 2022/07/09
% 
fprintf('\n ----------------------SLPMM begins-------------------\n');

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
  if ~isfield(opts,'maxit_inner');      opts.maxit_inner = 500;       end
 






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x           = opts.x0;

record      = opts.record;
itPrint     = opts.itPrint;

obj_bench   = opts.obj_bench;

maxit       = opts.maxit;
max_inner = opts.maxit_inner;


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
    [trace.obj, trace.cons, trace.time, ... 
        ] = deal(zeros(maxit,1));
end



eval_time   = 0;

% step size
alpha = maxit^(1/2);
sigma = maxit^(-1/2);


lambda = 0; % initial multiplier

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

        
        % update a, b, c
        temp1 = sqrt(sigma/alpha);
        a = temp1*v_g;
        b = lambda/temp1 + temp1*g_s - a'*x;
        c = v_f/alpha - x;

        % update iterate
        x_old = x;
        [x,iter_inner] = APG(x_old,a,b,c,f_g_handle,max_inner);
        
        % update multiplier
        temp4 = lambda + sigma*(g_s + v_g'*(x-x_old));
        lambda = max(0, temp4);
   
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

%% subfunctions
function    [x,iter] = APG(x0,a,b,c,f_g_handle,maxit_inner)
eps    = 1e-6;
eta    = 2;
yk     = x0;
L       = 1;
k       = 0;

dh     = grad_h(yk,a,b,c);
xk     = f_g_handle.proj(yk-dh/L);
res    = norm(xk-yk);

while res > eps && k < maxit_inner
    k = k + 1;
    x0 = xk;
    %% line search
    h0 = fun_h(yk,a,b,c);
    L1 = L*eta;
    it = 0;
    TLy = f_g_handle.proj(yk-dh/L1);
    diff  = TLy - yk;
    h1 = fun_h(TLy,a,b,c);
    while h1 > h0 + dh'*diff + 0.5*L1*norm(diff)^2 && it < 10
        it     = it + 1;
        L1   = L1*eta;
        TLy = f_g_handle.proj(yk-dh/L1);
        diff  = TLy - yk;
        h1   = fun_h(TLy,a,b,c);
    end
    L    = L1;
    xk  = TLy;
    res    = norm(xk-yk);
    yk = xk + k/(k+3)*(xk-x0);
    dh = grad_h(yk,a,b,c);
end

x = xk; iter = k;
end
%-------------------------------------------
function h   = fun_h(x,a,b,c)
    temp = max(0,a'*x + b);
    temp = temp.^2;
    h = 0.5*sum(temp) + 0.5*norm(x)^2 + c'*x;
end
%------------------------------------------------
function dh = grad_h(x,a,b,c)
        temp = max(0,a'*x + b);
        dh = a*temp + x + c;
end

