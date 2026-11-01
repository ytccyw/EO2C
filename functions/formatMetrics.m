function metricLine = formatMetrics(metrics, elapsedTime)
%FORMATMETRICS Create the single public output line.

metricLine = sprintf('ACC=%.2f | Kappa=%.2f | NMI=%.2f | Purity=%.2f | ARI=%.2f | Time=%.2fs', ...
    metrics(1), metrics(2), metrics(3), metrics(4), metrics(5), elapsedTime);
end
