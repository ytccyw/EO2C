function [q, qinit, objs, times] = EO2C(X, alpha, lambda, k, spLabel, opts)
%EO2C Efficient one-step orthogonal consensus clustering.
%
% Inputs:
%   X       1-by-V cell array. X{v} is N-by-Dv.
%   alpha   reconstruction coupling parameter.
%   lambda  consensus fusion parameter.
%   k       number of clusters.
%   spLabel optional N-by-1 superpixel labels.
%   opts    optional numerical settings:
%           KMeansReplicates - k-means++ starts for initialization.
%           MinIter          - minimum outer iterations.
%           MaxIter          - maximum outer iterations.
%
% Outputs:
%   q       final consensus labels.
%   qinit   initial labels.
%   objs    objective traces: {total, local, consensus}.
%   times   per-iteration timing, recorded only when requested.

if nargin < 6 || isempty(opts)
    opts = struct();
end
if ~isfield(opts, 'KMeansReplicates') || isempty(opts.KMeansReplicates)
    opts.KMeansReplicates = 1;
end
if ~isfield(opts, 'MinIter') || isempty(opts.MinIter)
    opts.MinIter = 20;
end
if ~isfield(opts, 'MaxIter') || isempty(opts.MaxIter)
    opts.MaxIter = 100;
end

V = numel(X);
N = size(X{1}, 1);
recordTimes = (nargout >= 4);
if recordTimes
    times = zeros(1, V + 1);
else
    times = [];
end

useSP = (nargin >= 5) && ~isempty(spLabel);
if useSP
    spLabel = spLabel(:);
    if numel(spLabel) ~= N
        error('spLabel length %d ~= N %d.', numel(spLabel), N);
    end
    numSP = max(spLabel);
end

% Numerical safeguards used by the alternating updates.
smallClusterMode = (k <= 6);
innerIter = 5 - 2 * smallClusterMode;
lambdaWarmup = 3 + 2 * smallClusterMode;
useViewWeights = ~smallClusterMode;
useCautiousSP = ~smallClusterMode;
useHysteresis = smallClusterMode;
usePeriodicPUpdate = true;

Z = cell(1, V);
dims = zeros(1, V);
for v = 1:V
    tLocal = tic;
    [u, ~, ~] = svds(X{v}, k);
    Z{v} = u;
    dims(v) = size(u, 2);
    if recordTimes, times(1, v) = toc(tLocal); end
end

tServer = tic;
Zcat = cell2mat(Z);
rng(24, 'twister');
[qinit, centers] = kmeans(Zcat, k, ...
    'Start', 'plus', 'MaxIter', 200, 'Replicates', opts.KMeansReplicates, ...
    'EmptyAction', 'singleton', 'Options', statset('UseParallel', false));
q = qinit;
P = mat2cell(centers, k, dims);

[u, ~, v] = svds(centers, k);
B = (k / N) * (u * v');
Bblocks = mat2cell(B, k, dims);
A = repmat({q}, 1, V);
if recordTimes, times(1, V + 1) = toc(tServer); end

objTotal = [];
objLocal = [];
objConsensus = [];
t = 1;

while true
    qOld = q;
    AOld = A;
    lambdaEff = lambda * min(1, t / lambdaWarmup);
    rowTime = zeros(1, V + 1);

    if useViewWeights
        viewScore = zeros(1, V);
        for v = 1:V
            viewScore(v) = trace(P{v}(A{v}, :)' * Bblocks{v}(q, :)) / max(N, 1);
            if ~isfinite(viewScore(v))
                viewScore(v) = 0;
            end
        end
        viewScore = viewScore - max(viewScore);
        viewWeight = exp(viewScore);
        viewWeight = viewWeight / max(sum(viewWeight), eps);
    else
        viewWeight = ones(1, V);
    end

    if useCautiousSP && V >= 2
        labelMatrix = zeros(N, V);
        for v = 1:V
            labelMatrix(:, v) = A{v};
        end
        agreement = zeros(N, 1);
        for n = 1:N
            labels = unique(labelMatrix(n, :));
            bestCount = 0;
            for c = 1:numel(labels)
                bestCount = max(bestCount, sum(labelMatrix(n, :) == labels(c)));
            end
            agreement(n) = bestCount / V;
        end
        meanAgreement = mean(agreement);
    else
        meanAgreement = 1;
    end

    for v = 1:V
        tLocal = tic;

        for inner = 1:innerIter
            AvPv = P{v}(A{v}, :);
            G = X{v} * (X{v}' * Z{v}) + alpha * AvPv;
            [u, ~, vv] = svds(G, k);
            Z{v} = u * vv';
        end

        localScores = alpha * (Z{v} * P{v}');
        consensusScores = lambdaEff * viewWeight(v) * (Bblocks{v}(q, :) * P{v}');
        scores = localScores + consensusScores;

        if useHysteresis
            [sortedScores, sortedLabels] = sort(scores, 2, 'descend');
            nextA = sortedLabels(:, 1);
            margin = sortedScores(:, 1) - sortedScores(:, 2);
            sortedMargin = sort(margin(:), 'ascend');
            tau = sortedMargin(max(1, min(numel(sortedMargin), round(0.35 * numel(sortedMargin)))));
            if t > 1
                keepPrevious = margin < tau;
                nextA(keepPrevious) = A{v}(keepPrevious);
            end
            A{v} = nextA;
        else
            [sortedScores, ~] = sort(scores, 2, 'descend');
            margin = sortedScores(:, 1) - sortedScores(:, 2);
            [~, A{v}] = max(scores, [], 2);
        end

        if useSP
            counts = accumarray([spLabel, A{v}], 1, [numSP, k]);
            [maxCount, spMajor] = max(counts, [], 2);

            if useCautiousSP
                spPurity = maxCount ./ max(sum(counts, 2), 1);
                classFreq = histcounts(q, 1:(k + 1));
                classFreq = classFreq(:) / max(sum(classFreq), 1);
                dominant = classFreq >= 0.45;

                sortedMargin = sort(margin(:), 'ascend');
                tau = sortedMargin(max(1, min(numel(sortedMargin), round(0.40 * numel(sortedMargin)))));
                applyMask = margin < tau & spPurity(spLabel) >= 0.60 & ~dominant(spMajor(spLabel));

                if meanAgreement < 0.40 && V >= 2
                    voteCount = zeros(N, 1);
                    labelMatrix = zeros(N, V);
                    for vv = 1:V
                        labelMatrix(:, vv) = A{vv};
                    end
                    for n = 1:N
                        labels = unique(labelMatrix(n, :));
                        bestCount = 0;
                        for c = 1:numel(labels)
                            bestCount = max(bestCount, sum(labelMatrix(n, :) == labels(c)));
                        end
                        voteCount(n) = bestCount;
                    end
                    applyMask = applyMask & voteCount >= 2;
                end

                if any(applyMask)
                    nextA = A{v};
                    nextA(applyMask) = spMajor(spLabel(applyMask));
                    A{v} = nextA;
                end
            else
                A{v} = spMajor(spLabel);
            end
        end

        if usePeriodicPUpdate && t >= 10 && mod(t, 5) == 0
            S = sparse((1:N)', A{v}, 1, N, k);
            left = (1 + lambda) * (S' * S) + 1e-12 * speye(k);
            right = S' * Z{v} + lambda * (S' * Bblocks{v}(q, :));
            P{v} = left \ right;
        end

        rowTime(v) = toc(tLocal);
    end

    tServer = tic;
    Ublocks = cell(1, V);
    for v = 1:V
        Ublocks{v} = viewWeight(v) * P{v}(A{v}, :);
    end
    Ucat = cell2mat(Ublocks);

    Q = sparse((1:N)', q, 1, N, k);
    [u, ~, vv] = svd(Ucat' * Q, 'econ');
    B = (k / N) * (vv(:, 1:k) * u(:, 1:k)');
    Bblocks = mat2cell(B, k, dims);

    qScore = Ucat * B';
    [~, q] = max(qScore, [], 2);

    if usePeriodicPUpdate
        emptyLabels = find(histcounts(q, 1:(k + 1)) == 0);
        for label = emptyLabels(:)'
            [~, idx] = max(qScore(:, label));
            q(idx) = label;
        end
    end

    rowTime(V + 1) = toc(tServer);
    if recordTimes
        times(t + 1, :) = rowTime;
    end

    localObj = 0;
    consensusObj = 0;
    for v = 1:V
        AvPv = P{v}(A{v}, :);
        localObj = localObj + norm(Z{v} - AvPv, 'fro')^2;
        consensusObj = consensusObj + norm(AvPv - Bblocks{v}(q, :), 'fro')^2;
    end
    if smallClusterMode
        objTotal(t) = localObj + lambdaEff * consensusObj;
    else
        objTotal(t) = localObj + lambda * consensusObj;
    end
    objLocal(t) = localObj;
    objConsensus(t) = consensusObj;

    if smallClusterMode && mod(t, 10) == 0
        for v = 1:V
            [Z{v}, ~] = qr(Z{v}, 0);
        end
    end

    objectiveStable = t > 1 && ...
        abs(objTotal(t - 1) - objTotal(t)) / (abs(objTotal(t - 1)) + 1e-9) < 1e-5;
    labelsStable = all(q == qOld);
    for v = 1:V
        labelsStable = labelsStable && all(A{v} == AOld{v});
    end

    if (t >= opts.MinIter && (objectiveStable || labelsStable)) || t > opts.MaxIter
        break;
    end
    t = t + 1;
end

if nargout >= 3
    objs = {objTotal, objLocal, objConsensus};
else
    objs = [];
end
end
