function [f_g_handle] = make_f_g_logloss(A0,A1,risk_level,mu)
    f_g_handle.get_f = @get_f;
    f_g_handle.get_g = @get_g;
    f_g_handle.get_s_g = @get_s_g;    

  
    f_g_handle.perm_g  = @perm_g;
    f_g_handle.perm_f  = @perm_f;
    f_g_handle.resample_g = @resample_g;
    f_g_handle.resample_f = @resample_f;
    f_g_handle.get_s_grad_f = @get_s_grad_f;
    f_g_handle.get_grad_f = @get_grad_f;
    f_g_handle.get_s_grad_g = @get_s_grad_g;
    f_g_handle.get_grad_g = @get_grad_g;
    f_g_handle.A0 = A0;
    f_g_handle.A1 = A1;
    f_g_handle.p = size(A0,2);


    f_g_handle.risk_level = risk_level;
    f_g_handle.get_obj = @get_obj;
    f_g_handle.get_cons = @get_cons;
%     f_g_handle.get_Hd = @get_Hd;
    A0_perm_f     = [];
    A1_perm_g     = [];
    A0_sample_f = [];
    A1_sample_g = [];
    
    if nargin >= 5
         f_g_handle.mu = mu; 
         f_g_handle.prox  = @prox;  
    end


    function [f, err_rate] = get_f(x)
        tmp = A0 * x;
        err_rate = sum(tmp<0)/size(A0,1);
        f = mean(log(1+exp(-tmp)));
    end

    function [g] = get_g(x)
        tmp = -A1 * x;
        g = mean(log(1+exp(-tmp))) - risk_level;
    end

    
    function [g] = get_s_g(x)
        if isempty(A1_sample_g)
            error('sample not initialized');
        end
        tmp = -A1_sample_g * x;
        g = mean(log(1+exp(-tmp))) - risk_level;
    end




   function [ss,null_ind] = prox(y)
        tmp = abs(y)-mu;
        null_ind = tmp <= 0;
        ss = sign(y).*max(tmp,0);
    end

    function [] = perm_g()
        rand_perm = randperm(size(A1,1));
        A1_perm_g = A1(rand_perm,:);
    end

    function [] = perm_f()
        rand_perm = randperm(size(A0,1));
        A0_perm_f = A0(rand_perm,:);
    end
    

    function [] = resample_g(sample_start_pos, ssize)
        if isempty(A1_perm_g)
            error('samples A1 not permutated');
        end
        A1_sample_g = A1_perm_g(sample_start_pos:sample_start_pos+ssize-1,:);
    end


    function [] = resample_f(sample_start_pos, ssize)
        if isempty(A0_perm_f)
            error('samples A0 not permutated');
        end
        A0_sample_f = A0_perm_f(sample_start_pos:sample_start_pos+ssize-1,:);
    end

    
    function [grad] = get_s_grad_g(x)
        if isempty(A1_sample_g)
            error('sample not initialized');
        end
        cache = 1 - 1./(1 + exp(A1_sample_g * x));
        grad   = (cache' * A1_sample_g)' / size(A1_sample_g,1);
    end

    function [grad] = get_grad_g(x)
        cache = 1 - 1./(1 + exp(A1 * x));
        grad   = (cache' * A1)' / size(A1,1);
    end    
    
    function [grad] = get_s_grad_f(x)
        if isempty(A0_sample_f)
            error('sample not initialized');
        end
        cache = 1 - 1./(1 + exp(-A0_sample_f * x));
        grad   = (-cache' * A0_sample_f)' / size(A0_sample_f,1);
    end

    function [grad] = get_grad_f(x)
        cache = 1 - 1./(1 + exp(-A0 * x));
        grad   = (-cache' * A0)' / size(A0,1);
    end 

    function [c,ceq,gradc,gradceq] = get_cons(x)
            c = get_g(x); ceq = [];
            gradc = get_grad_g(x); gradceq = [];
            
    end 

    function [f,gradf] = get_obj(x)
            f = get_f(x);
            gradf = get_grad_f(x);
    end
 

%     function [Jd] = get_Hd(A,b,x,d)
%         cache0 = exp(b.*(A * x));
%         cache1 = 1./(2 + cache0 + 1./cache0);
%         Jd       = ((cache1.*(A * d))' * A)' / size(A,1);
%     end
    
end