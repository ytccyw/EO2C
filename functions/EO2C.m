function [q, qinit, objs, times] = EO2C(X, alpha, lambda, k, spLabel, opts)
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
    opts.MinIter = 1;
end
if ~isfield(opts, 'MaxIter') || isempty(opts.MaxIter)
    opts.MaxIter = 100;
end

validateattributes(opts.KMeansReplicates, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(opts.MinIter, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(opts.MaxIter, {'numeric'}, {'scalar','integer','positive','finite'});
assert(opts.MinIter <= opts.MaxIter, 'EO2C:Iterations', 'MinIter must not exceed MaxIter.');

V = numel(X);
assert(iscell(X) && isvector(X) && V>=2, 'EO2C:Views', 'X must contain at least two feature matrices.');
N = size(X{1}, 1);
validateattributes(k, {'numeric'}, {'scalar','integer','>=',2,'<=',N,'finite'});
validateattributes(alpha, {'numeric'}, {'scalar','positive','finite'});
validateattributes(lambda, {'numeric'}, {'scalar','nonnegative','finite'});
for v = 1:V
    validateattributes(X{v}, {'double','single'}, {'2d','real','finite','nonempty','nrows',N});
    X{v} = double(X{v});
end
assert(sum(cellfun(@(a) min(k,size(a,2)),X))>=k, 'EO2C:EmbeddingSize', ...
    'The concatenated embedding must have at least k columns.');
recordTimes = (nargout >= 4);
if recordTimes
    times = zeros(opts.MaxIter + 1, V + 1);
else
    times = [];
end

useSP = (nargin >= 5) && ~isempty(spLabel);
if useSP
    spLabel = spLabel(:);
    if numel(spLabel) ~= N
        error('spLabel length %d ~= N %d.', numel(spLabel), N);
    end
    validateattributes(spLabel, {'numeric'}, {'real','finite','integer','positive','<=',N});
    numSP = max(spLabel);
end

innerIter = 5;
lambdaWarmup = 3;

Z = cell(1, V);
dims = zeros(1, V);
for v = 1:V
    if recordTimes, tLocal = tic; end
    [u, ~, ~] = svds(X{v}, k);
    Z{v} = u;
    dims(v) = size(u, 2);
    if recordTimes, times(1, v) = toc(tLocal); end
end

if recordTimes, tServer = tic; end
Zcat = cell2mat(Z);
rng(251, 'twister');
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

objTotal = zeros(1,opts.MaxIter);
objLocal = zeros(1,opts.MaxIter);
objConsensus = zeros(1,opts.MaxIter);
t = 1;

while true
    qOld = q;
    AOld = A;
    lambdaEff = lambda * min(1, t / lambdaWarmup);
    if recordTimes, rowTime = zeros(1, V + 1); end

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

    if useSP && V > 2
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
        if recordTimes, tLocal = tic; end

        AvPv = P{v}(A{v}, :);
        for inner = 1:innerIter
            G = X{v} * (X{v}' * Z{v}) + alpha * AvPv;
            [u, ~, vv] = svds(G, k);
            Z{v} = u * vv';
        end

        localScores = alpha * (Z{v} * P{v}');
        consensusScores = lambdaEff * (viewWeight(v) * (Bblocks{v}(q, :) * P{v}'));
        scores = localScores + consensusScores;

        [~, A{v}] = max(scores, [], 2);

        if useSP
            topScores = maxk(scores, 2, 2);
            margin = topScores(:, 1) - topScores(:, 2);
            counts = accumarray([spLabel, A{v}], 1, [numSP, k]);
            [maxCount, spMajor] = max(counts, [], 2);

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
        end

        if t >= 10 && mod(t, 5) == 0
            S = sparse((1:N)', A{v}, 1, N, k);
            P{v} = (S' * Z{v}) ./ (full(sum(S, 1))' + 1e-12);
        end

        if recordTimes, rowTime(v) = toc(tLocal); end
    end

    if recordTimes, tServer = tic; end
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

    classCounts = accumarray(q, 1, [k, 1]);
    emptyLabels = find(classCounts == 0);
    for label = emptyLabels(:)'
        fillScore = qScore(:, label);
        fillScore(classCounts(q) <= 1) = -Inf;
        [~, idx] = max(fillScore);
        classCounts(q(idx)) = classCounts(q(idx)) - 1;
        q(idx) = label;
        classCounts(label) = 1;
    end

    if recordTimes
        rowTime(V + 1) = toc(tServer);
        times(t + 1, :) = rowTime;
    end

    localObj = 0;
    consensusObj = 0;
    for v = 1:V
        AvPv = P{v}(A{v}, :);
        localObj = localObj + norm(Z{v} - AvPv, 'fro')^2;
        consensusObj = consensusObj + norm(AvPv - Bblocks{v}(q, :), 'fro')^2;
    end
    objTotal(t) = localObj + lambda * consensusObj;
    objLocal(t) = localObj;
    objConsensus(t) = consensusObj;

    objectiveStable = t > 1 && ...
        abs(objTotal(t - 1) - objTotal(t)) / (abs(objTotal(t - 1)) + 1e-9) < 1e-5;
    labelsStable = all(q == qOld);
    for v = 1:V
        labelsStable = labelsStable && all(A{v} == AOld{v});
    end

    if (t >= opts.MinIter && (objectiveStable || labelsStable)) || t >= opts.MaxIter
        break;
    end
    t = t + 1;
end

if nargout >= 3
    objs = {objTotal(1:t), objLocal(1:t), objConsensus(1:t)};
else
    objs = [];
end
if recordTimes
    times = times(1:t + 1, :);
end
end
