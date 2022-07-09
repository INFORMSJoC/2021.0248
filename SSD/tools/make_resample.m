function p = make_resample(m,randseed)
    if nargin == 2; rng(randseed); end
    sample_start_pos = 1;
    rand_perm = randperm(m);
    p = @resample;
    function [index, start_pos] = resample(ssize)
        if ssize > m
%             warning('sample size too large');
            index = mod(randperm(ssize), m) + 1;
            return;
        end
        if sample_start_pos + ssize > m + 1
            rand_perm = randperm(m);
            sample_start_pos = 1;
        end     
        index = rand_perm(sample_start_pos:sample_start_pos+ssize-1);
        start_pos = sample_start_pos;
        sample_start_pos = sample_start_pos + ssize;
    end
end