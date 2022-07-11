function [x, out] = PALEM_SSD_x(f_g_handle,opts)
% PALEM_SSD_x is a code written by Xiantao Xiao
% for solving SSD constrained portfolio optimization problem
% with PALEM proposed in the following paper: 
% Dentcheva D, Martinez G, Wolfhagen E (2016) 
% Augmented Lagrangian methods for solving optimization
% problems with stochastic-order constraints. 
% Oper. Res. 64(6):1451–1465.

% Version: 2022/07/09
% 
fprintf('\n ----------------------PALEM_SSD begins-------------------\n');

N           = f_g_handle.scenarios;
p           = f_g_handle.variables;


tic;

%%-------------------------------------------------------------------------
if nargin < 2; opts = []; end
 if ~isfield(opts,'x0');       opts.x0  = zeros(p,1);   end
 if ~isfield(opts,'tol');      opts.tol = 1e-4;     end
 if ~isfield(opts,'record');   opts.record = 1;     end
 if ~isfield(opts,'itPrint');  opts.itPrint = 10;     end

% 
 if ~isfield(opts,'maxit');      opts.maxit = 500;       end
 if ~isfield(opts,'maxit_inner');      opts.maxit_inner = 500;       end
  if ~isfield(opts,'delta');      opts.delta = 100;       end
    if ~isfield(opts,'max_delta');      opts.max_delta = 1e+5;       end

 %if ~isfield(opts,'ssize_a'); opts.ssize_a = 0; end
%  if ~isfield(opts,'s_cons'); opts.s_cons = 0; end
%   
%   
%  if ~isfield(opts,'ssize_f');  opts.ssize_f = max(min(N,10),floor(N/100));   end
%  if ~isfield(opts,'ssize_g');  opts.ssize_g = max(min(N,10),floor(N/100));   end
%  if ~isfield(opts,'ssize_I');  opts.ssize_I = max(min(N,10),floor(N/100));   end
% 
%  if ~isfield(opts,'alpha'); opts.alpha = N^(1/2); end
%  if ~isfield(opts,'sigma'); opts.sigma = N^(-1/4); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x           = opts.x0;


record      = opts.record;
itPrint     = opts.itPrint;

obj_bench   = opts.obj_bench;


tol = opts.tol;


maxit                = opts.maxit;
%maxit_inner       = opts.maxit_inner;


delta = opts.delta;
max_delta = opts.max_delta;
%--------------------------------------------------------------------------
if record >= 1
    % set up print format

    stra = ['%6s', '%13s','%13s', '%12s', '%12s', '\n', ...
        repmat( '-', 1, 56 ), '\n'];
    str_head = sprintf(stra, ...
        'iter', 'obj', 'max(cons)', 'obj-bench', 'inner_iter');
    str_num = ('%4d |  %+2.4e  %+2.4e %+2.4e %4d \n');
end
%--------------------------------------------------------------------------

% prepare trace in output
if opts.trace
    [trace.obj, trace.cons, trace.res, trace.time, ... 
        trace.epoch] = deal(zeros(maxit,1));
end



eval_time   = 0;

J = ones(1);
mu = ones(1);
iter_inner = 100;
nr_epoch = 0;
rho = 100;

Aeq = ones(p,1)';
beq = 1;
lb = zeros(p,1);
ub = f_g_handle.ub*ones(p,1);

con_I = zeros(N,1);
%--------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:maxit
    
    
        % calculate obj and cons
        if record >= 1
            time_temp   = toc;
            obj             = f_g_handle.get_f(x);

            for j = 1:N; con_I(j) = f_g_handle.get_g(x,j);  end

            [cons,I]            = max(con_I); 

            res             = (obj - obj_bench)/abs(obj_bench);
            eval_time   = eval_time + toc - time_temp;
            
        end
        
        % save information for graphic output
        if opts.trace 
            trace.epoch(iter)   = nr_epoch;
            trace.obj(iter)     = obj;
            trace.cons(iter)    = cons;
            trace.time(iter)    = toc - eval_time;
            trace.res(iter)     = res;
        end            
        
        if iter >= 2 && norm(mu-mu0) <= tol
%             if cons < tol
%                 break;
%             end    
            if ismember(I,J) && length(J) <= N
                delta = min(2*delta,max_delta); % update penalty parameter
            else % update mu and J
                J0 = J; mu0 = mu;
                J = ones(length(J0)+1); mu = J;
                J = [J0;I];
                mu = [mu0;0];
            end
        end
        
        
        
        % print the information
        if record>=1 && (iter == 1 || iter==maxit || mod(iter,itPrint)==0)
            if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit 
                    fprintf('\n%s\n', str_head);
            end
           fprintf(str_num, iter, obj, cons, res, iter_inner);
        end 
   
        % solve subproblem by fmincon
        options = struct('Display','off');
        fun = @(x)f_g_handle.get_AL(x,J,mu,rho);
        x_old = x;
        [x,fval,exitflag,output] = fmincon(fun,x_old,[],[],Aeq,beq,lb,ub,[],options);
        iter_inner = output.iterations;
        
        % update multipliers
        mu0 = mu;
        for j = 1:length(J)
            ind = J(j);
            tmp = f_g_handle.get_g(x,ind);
            mu(j) = max(0,mu(j)+rho*tmp);
        end
        

        



        
 
end %end outer loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OUTPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
out.time    =  toc - eval_time;

       for j = 1:N; con_I(j) = f_g_handle.get_g(x,j);  end
            cons            = max(con_I);   

if opts.trace; out.trace = trace; end
out.iter    = iter;
out.obj     = obj;
out.cons    = cons;
out.res     = res;
out.epoch   = nr_epoch;
end


