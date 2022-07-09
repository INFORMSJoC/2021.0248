function [f_g_handle] = make_f_g_empirical(R_matrix,Y_bench,ub)
    f_g_handle.get_f    = @get_f;
    f_g_handle.get_g    = @get_g;
    f_g_handle.get_AL    = @get_AL;
    f_g_handle.get_s_g  = @get_s_g;    
    f_g_handle.proj     = @proj;

    
    f_g_handle.perm_g       = @perm_g;
    f_g_handle.perm_f       = @perm_f;
    f_g_handle.perm_I       = @perm_I;    
    f_g_handle.resample_g   = @resample_g;
    f_g_handle.resample_f   = @resample_f;
    f_g_handle.resample_I   = @resample_I;    
    f_g_handle.get_s_grad_f = @get_s_grad_f;
    f_g_handle.get_grad_f   = @get_grad_f;
    f_g_handle.get_s_grad_g = @get_s_grad_g;
    f_g_handle.get_grad_g   = @get_grad_g;
    f_g_handle.get_grad     = @get_grad;

     f_g_handle.scenarios = size(R_matrix,1);
     f_g_handle.variables = size(R_matrix,2);
     f_g_handle.ub = ub;



    R_perm_f     = [];
    R_perm_g     = [];
    Y_perm_g     = [];
    perm_Ind       = [];
    R_sample_f   = [];
    R_sample_g   = [];
    Y_sample_g   = [];
    %sample_Ind     = [];
    


    function [f] = get_f(x)
        tmp = R_matrix * x;
        f = -mean(tmp);
    end

    function [g] = get_g(x,j)
        r   = R_matrix * x;
        tmp = max(0,Y_bench(j)-r) - max(0,Y_bench(j)-Y_bench);
        g = mean(tmp);
    end



  

    
    function [g] = get_s_g(x,j)
        if isempty(R_sample_g)
            error('sample not initialized');
        end
        r   = R_sample_g * x;
        tmp = max(0,Y_bench(j)-r) - max(0,Y_bench(j)-Y_sample_g);
        g = mean(tmp);
    end


    function [p] = proj(y)
        p = proj_capped_splx(y,ub);
    end



    function [] = perm_g()
        rand_perm = randperm(size(R_matrix,1));
        R_perm_g  = R_matrix(rand_perm,:);
        Y_perm_g  = Y_bench(rand_perm);
    end

    function [] = perm_f()
        rand_perm = randperm(size(R_matrix,1));
        R_perm_f = R_matrix(rand_perm,:);
    end

    function [] = perm_I()
        perm_Ind = randperm(size(R_matrix,1));
    end
    

    function [] = resample_g(sample_start_pos, ssize)
        if isempty(R_perm_g)
            error('samples are not permutated');
        end
        R_sample_g = R_perm_g(sample_start_pos:sample_start_pos+ssize-1,:);
        Y_sample_g = Y_perm_g(sample_start_pos:sample_start_pos+ssize-1);
    end


    function [] = resample_f(sample_start_pos, ssize)
        if isempty(R_perm_f)
            error('samples are not permutated');
        end
        R_sample_f = R_perm_f(sample_start_pos:sample_start_pos+ssize-1,:);
    end

    function [sample_Ind] = resample_I(sample_start_pos, ssize)
        if isempty(perm_Ind)
            error('samples are not permutated');
        end
        sample_Ind = perm_Ind(sample_start_pos:sample_start_pos+ssize-1);
    end
    
    function [grad] = get_s_grad_g(x,j)
        if isempty(R_sample_g)
            error('sample not initialized');
        end
        ind    = Y_bench(j) - R_sample_g * x > 0;
        cache  = -R_sample_g .* ind;
        grad   = mean(cache,1)';
    end

    function [grad] = get_grad_g(x,j)
        ind    = Y_bench(j) - R_matrix * x > 0;
        cache  = -R_matrix .* ind;
        grad   = mean(cache,1)';
    end    
    
    function [grad] = get_s_grad_f(x)
        if isempty(R_sample_f)
            error('sample not initialized');
        end
        grad   = -mean(R_sample_f,1)';
    end

    function [grad] = get_grad_f(x)
        grad   = -mean(R_matrix,1)';
    end 



    function [L] = get_AL(x,J,mu,rho)
        r = R_matrix * x;
        L = -mean(r);
        for i =1:length(J)
            j = J(i);
            tmp = max(0,Y_bench(j)-r) - max(0,Y_bench(j)-Y_bench);
            delta = mean(tmp);
            tmp0 =max(0, delta+mu(i)/rho);
            L = L + 0.5 * rho * tmp0^2;
        end
    end
 

    
end