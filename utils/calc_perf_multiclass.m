function [out] = calc_perf_multiclass(gt, dvs, yset, measures)
assert(length(gt)==length(dvs));
m = length(measures);

nc = length(yset);

out.perf = zeros(nc,m);
out.perf_names = measures;

for i = 1 : length(measures)
    p = lower(measures{i});
    for c = 1 : length(yset)
        dv = dvs(:,c);
        gtt = 2*double(gt == yset(c))-1;        
        out.perf(c,i) = eval(sprintf('calc_%s(gtt,dv)', p));
    end
end

out.pformat = [repmat('%.4f\t',1, m-1), '%.4f'];
out.mean_perf = mean(out.perf);
out.str = sprintf(out.pformat, out.mean_perf);