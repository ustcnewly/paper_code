function [out] = calc_perf(gt, dv, measures)
assert( all(ismember(gt, [-1 1])) );
assert(length(gt)==length(dv));
m = length(measures);


out.pformat = [repmat('%.4f\t',1, m-1), '%.4f'];
out.perf = zeros(m,1);
out.perf_names = measures;
for i = 1 : length(measures)
    p = lower(measures{i});
    out.perf(i) = eval(sprintf('calc_%s(gt,dv)', p));
end

out.str = sprintf(out.pformat, out.perf);