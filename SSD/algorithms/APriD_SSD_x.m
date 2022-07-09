function [x, out] = APriD_SSD_x(f_g_handle,opts)
% APriD_SSD_x is a code written by Xiantao Xiao
% for solving SSD constrained portfolio optimization problem
% with APriD proposed in the following paper: 
% Yan Y, Xu Y (2022) Adaptive primal-dual stochastic gradient method
% for expectation-constrained convex stochastic programs. 
% Math. Program. Comput. 14(2):319–363.

% Version: 2022/07/09
% 
fprintf('\n ----------------------APriD_SSD begins-------------------\n');

N           = f_g_handle.scenarios;
p           = f_g_handle.variables;


tic;

%%-------------------------------------------------------------------------
if nargin < 2; opts = []; end
 if ~isfield(opts,'x0');       opts.x0  = zeros(p,1);   end
 if ~isfield(opts,'tol');      opts.tol = 1e-6;     end
 if ~isfield(opts,'record');   opts.record = 1;     end
 if ~isfield(opts,'itPrint');  opts.itPrint = 10;     end

% 
 if ~isfield(opts,'maxit');      opts.maxit = 500;       end
 if ~isfield(opts,'maxit_inner');      opts.maxit_inner = 500;       end

 if ~isfield(opts,'ssize_a'); opts.ssize_a = 0; end
 if ~isfield(opts,'s_cons'); opts.s_cons = 0; end
  
  
 if ~isfield(opts,'ssize_f');  opts.ssize_f = max(min(N,10),floor(N/100));   end
 if ~isfield(opts,'ssize_g');  opts.ssize_g = max(min(N,10),floor(N/100));   end
 if ~isfield(opts,'ssize_I');  opts.ssize_I = max(min(N,10),floor(N/100));   end

 if ~isfield(opts,'alpha'); opts.alpha = N^(1/2); end
 if ~isfield(opts,'sigma'); opts.sigma = N^(-1/4); end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x         = opts.x0;

m         = zeros(p,1);
v         = zeros(p,1);
v_hat     = zeros(p,1);


record      = opts.record;
itPrint     = opts.itPrint;

obj_bench   = opts.obj_bench;

s_cons  = opts.s_cons;
ssize_f     = opts.ssize_f;
ssize_g     = opts.ssize_g;
if s_cons; ssize_I     = opts.ssize_I; end

maxit                = opts.maxit;


ssize_a     = opts.ssize_a;


if s_cons; lambda  = zeros(ssize_I,1); else; lambda = zeros(N,1); end
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


% prepare sampling
resample_f      = make_resample(N);
resample_g      = make_resample(N);
if s_cons; resample_I      = make_resample(N);end
sample_time     = 0;
eval_time   = 0;

nr_epoch        = 0;

%step size
alpha = 1/maxit^(1/2);
rho   = 1/maxit^(1/2);

% parameters
beta1 = 0.9;
beta2 = 0.99;
theta = 10;


con_I = zeros(N,1);
iter_inner = 0;

%--------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MAIN LOOP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for iter = 1:maxit
    
    
        % calculate obj and cons
        if record >= 1
            time_temp   = toc;
            obj             = f_g_handle.get_f(x);
            
            % Computing the values of constraints is very time-consuming,
            % so we simple let cons = 0 for saving time
                cons = 0;
%             for j = 1:N; con_I(j) = f_g_handle.get_g(x,j);  end 
%             cons            = max(con_I);  
            
            
            res             = (obj - obj_bench)/abs(obj_bench);
            eval_time   = eval_time + toc - time_temp;
            
        end
        % print the information
        if record>=1 && (iter == 1 || iter==maxit || mod(iter,itPrint)==0)
            if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit 
                    fprintf('\n%s\n', str_head);
            end
           fprintf(str_num, iter, obj, cons, res, iter_inner);
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
        


        % sampling, evaluation of  stochastic gradient of g  
        [~,start_pos_g] = resample_g(ssize_g);
        if s_cons;[~,start_pos_I] = resample_I(ssize_I);end
        time_temp       = toc;
        f_g_handle.resample_g(start_pos_g, ssize_g);
        if s_cons;sample_Ind = f_g_handle.resample_I(start_pos_I, ssize_I);end        
        sample_time     = toc - time_temp + sample_time;
        if s_cons
            nr_epoch        = nr_epoch + ssize_I * ssize_g/N;
                    v_g = zeros(p,ssize_I);
                    g_s = zeros(ssize_I,1);
        else 
            nr_epoch        = nr_epoch +  ssize_g/N;
                    v_g = zeros(p,N);
                    g_s = zeros(N,1);
        end



        if s_cons
            for j = 1:ssize_I
                ind                 = sample_Ind(j);
                g_s(j)             = f_g_handle.get_s_g(x,ind);
                v_g(:,j)           = f_g_handle.get_s_grad_g(x,ind);
            end
        else
             for j = 1:N
                g_s(j)             = f_g_handle.get_s_g(x,j);
                v_g(:,j)           = f_g_handle.get_s_grad_g(x,j);
            end           
        end


        
        % update 
        u   = v_f + v_g*lambda;
        w   = g_s;
        m   = beta1*m + (1-beta1)*u;
        v   = beta2*v + (1-beta2)*u.^2/max(1,norm(u)^2/theta);
        v_hat = max(v,v_hat);
        
        
        % update iterate
        x = x - alpha*m./(sqrt(v_hat)+1e-15); %% +1e-15 in case v_hat=0.
        x = f_g_handle.proj(x);

        % update multiplier
        lambda = max(lambda + rho*w,0);
        
        % save information for graphic output
        if opts.trace 
            trace.epoch(iter)   = nr_epoch;
            trace.obj(iter)     = obj;
            trace.cons(iter)    = cons;
            trace.time(iter)    = toc - eval_time;
            trace.res(iter)     = res;
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


