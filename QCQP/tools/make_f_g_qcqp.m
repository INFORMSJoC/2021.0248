function [f_g_handle] = make_f_g_qcqp(n,p,R,x_hat,batch)
    f_g_handle.get_obj    = @get_obj;
    f_g_handle.get_cons    = @get_cons;
    f_g_handle.get_f_grad    = @get_f_grad;
    f_g_handle.get_g_grad    = @get_g_grad;    
    f_g_handle.proj    = @get_proj;
    
     f_g_handle.n = n;
     f_g_handle.p = p;
     f_g_handle.R = R;
     f_g_handle.x_hat = x_hat;

     
    
  
    
    
    
    function [ y ] = get_obj( x )
            y = 0.5*norm(x)^2+0.5*norm(x_hat)^2;
    end

    function [ g ] = get_cons( x )
            g =  0.5*(norm(x)^2-norm(x_hat)^2) - [1:p]';
    end
 

    function [z] = get_proj(x)
        z = x;
        if norm(x) > R
            z = R*x/norm(x);
        end
    end


    function [grad] = get_f_grad(x)
        grad = zeros(n,1);
        for j = 1:batch
            Delta = -0.1 + 0.2*rand(n,n);
            Delta = (Delta+Delta')/2;
            b = -1 + 2*rand(n,1);
            grad = grad + x + Delta*x + b;
        end
        grad = grad/batch;
    end

    function [g,G] = get_g_grad(x)
        g = zeros(p,1);
        G = zeros(n,p);
        tmp = norm(x)^2 - norm(x_hat)^2;
        for j = 1:batch
        for i = 1:p
            Delta = -0.1 + 0.2*rand(n,n);
            Delta = (Delta+Delta')/2;
            b = -1 + 2*rand(n,1);
            G(:,i) = G(:,i) + x + Delta*x + b;
            h = 2*i*rand;
            g(i) = g(i) + 0.5*(tmp+x'*Delta*x-x_hat'*Delta*x_hat) + b'*(x-x_hat) -h;
        end
        end
        g = g/batch;
        G = G/batch;
    end
    
end