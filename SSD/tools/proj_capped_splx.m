function [x]=proj_capped_splx(y0,ub)
% function [x]=projection(y0,ub) solves the projection onto
%   the capped simplex:
%     min_x 0.5*|x-y0|^2
%       s.t. 0 <= x_i <= ub,
%            sum_i x_i = 1.
% 
% Inputs
%   y0: nx1 vector.
%   ub: the upper bound.
% 
% Outputs
%   x: Nx1 vector.
%
% When ub = 1, we use the method in the following reference
%
%   Yunmei Chen, Xiaojing Ye: "Projection onto a simplex". 
%     Feb 10, 2011, https://arxiv.org/abs/1101.6081.
%
% When ub < 1, we use the method in the following reference
% 
%   Weiran Wang, Canyi Lu: "Projection onto the capped simplex". 
%     March 3, 2015, https://arxiv.org/abs/1503.01002.

if ub < 1
    z = projection(y0/ub,1/ub);
    x = ub * z;
elseif ub > 1
    error('The upper bound cannot be greater than 1.0!');
else
    x = projsplx(y0);
end

