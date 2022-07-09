function [x, out] = YNWs_std_x(f_g_handle,opts)
% YNWs_std_x is a code written by Xiantao Xiao
% for solving SSD constrained portfolio optimization problem
% with an algorithm proposed in the following paper: 
% Yu H, Neely MJ, Wei X (2017) Online convex optimization with stochastic constraints. 
% Advances in Neural Information Processing Systems, 1428–1438.

% Version: 2022/07/09
% 
fprintf('\n ----------------------YNW begins-------------------\n');

N           = f_g_handle.scenarios;
p           = f_g_handle.variables;


tic;

%%-------------------------------------------------------------------------
if nargin < 2; opts = []; end
 if ~isfield(opts,'x0');       opts.x0  = zeros(p,1);   end
 if ~isfield(opts,'tol');      opts.tol = 1e-6;     end
 if ~isfield(opts,'record');   opts.record = 1;     end
 if ~isfield(opts,'itPrint');  opts.itPrint = 10;     end
 if ~isfield(opts,'eps');      opts.eps = 1e-2;     end 

% 
 if ~isfield(opts,'maxit');      opts.maxit = 500;       end
 

 if ~isfield(opts,'ssize_a'); opts.ssize_a = 0; end
    if ~isfield(opts,'s_cons'); opts.s_cons = 0; end
 
 if ~isfield(opts,'ssize_f');  opts.ssize_f = max(min(N,10),floor(N/100));   end
 if ~isfield(opts,'ssize_g');  opts.ssize_g = max(min(N,10),floor(N/100));   end
  if ~isfield(opts,'ssize_I');  opts.ssize_I = max(min(N,10),floor(N/100));   end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x           = opts.x0;

record      = opts.record;
itPrint     = opts.itPrint;

obj_bench   = opts.obj_bench;


ssize_f     = opts.ssize_f;
ssize_g     = opts.ssize_g;
s_cons      = opts.s_cons;
if s_cons 
    ssize_I     = opts.ssize_I; 
end

maxit       = opts.maxit;


ssize_a     = opts.ssize_a;


%--------------------------------------------------------------------------
if record >= 1
    % set up print format

    stra = ['%6s', '%13s','%13s', '%12s', '%12s', '\n', ...
        repmat( '-', 1, 56 ), '\n'];
    str_head = sprintf(stra, ...
        'iter', 'obj', 'cons', 'obj-bench', 't');
    str_num = ('%4d |  %+2.4e  %+2.4e %+2.4e %+2.4e \n');
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
resample_I      = make_resample(N);
sample_time     = 0;
eval_time   = 0;

nr_epoch        = 0;

% step size
alpha0 = opts.beta0; gamma0 = opts.beta0;
a = 1/2; c = 1;
alpha           = alpha0 / maxit^a;
gamma           = gamma0 / maxit^c; 



t = zeros(N,1);


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
            res             = (obj - obj_bench)/abs(obj_bench);
            % Computing the values of constraints is very time-consuming,
            % so we simple let cons = 0 for saving time
                cons = 0;
%             for j = 1:N; con_I(j) = f_g_handle.get_g(x,j);  end 
%             cons            = max(con_I);  

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
        
        % print the information
        if record>=1 && (iter == 1 || iter==maxit || mod(iter,itPrint)==0)
            if iter == 1 || mod(iter,20*itPrint) == 0 && iter~=maxit 
                    fprintf('\n%s\n', str_head);
            end
           fprintf(str_num, iter, obj, cons, res, max(t));
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
        [~,start_pos_I] = resample_I(ssize_I);
        time_temp       = toc;
        f_g_handle.resample_g(start_pos_g, ssize_g);
        sample_Ind = f_g_handle.resample_I(start_pos_I, ssize_I);
        sample_time     = toc - time_temp + sample_time;
        nr_epoch        = nr_epoch + ssize_I * ssize_g/N;

        v_g = zeros(p,1);

%         if iter == 1 
%             t = 0;
%         end

        % compute v_g
        if s_cons
            for j = 1:ssize_I
            ind             = sample_Ind(j); 
                if t(ind) > eps
                    v_g = v_g +  t(ind) *  f_g_handle.get_s_grad_g(x,ind);
                end 
            end
        else
         for j = 1:N
            ind             = j; 
            if t(ind) > eps
                v_g = v_g +  t(ind) * f_g_handle.get_s_grad_g(x,ind) ;
            end        
         end           
        end
        % update iterate
        %if  maxit - iter >= itPrint
        %if iter >= 2 && obj > obj_bench 
        y               = x - alpha * v_f;
        
        x_old = x;

        x   = y - gamma * v_g;
        
        x = f_g_handle.proj(x);
        
        % update t
        if s_cons
            for j = 1:ssize_I
                ind             = sample_Ind(j); 
                g_s             = f_g_handle.get_s_g(x_old,ind);
                v_g             = f_g_handle.get_s_grad_g(x_old,ind) ;
                t(ind)          = t(ind) + g_s + v_g'*(x-x_old);
                t(ind)          = max(t(ind),0);            
            end
        else    
            for j = 1:N   
                ind = j;
                g_s             = f_g_handle.get_s_g(x_old,ind);
                v_g             = f_g_handle.get_s_grad_g(x_old,ind) ;
                t(ind)          = t(ind) + g_s + v_g'*(x-x_old);
                t(ind)          = max(t(ind),0);
            end
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



