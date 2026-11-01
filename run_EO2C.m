function run_EO2C()
%RUN_EO2C Run the MUUFL Gulfport clustering demo.

rootDir = fileparts(mfilename('fullpath'));
originalPath = path;
pathCleanup = onCleanup(@() path(originalPath));
addpath(fullfile(rootDir, 'functions'));

dk = 10;
numPixel = 185;
alpha = 2^6;
lambda = 2^-8;

dataDir = fullfile(rootDir, 'datasets');
gtData = load(fullfile(dataDir, 'gt.mat'), 'gt');
hsiData = load(fullfile(dataDir, 'HSI.mat'), 'HSI');
lidarData = load(fullfile(dataDir, 'LiDAR.mat'), 'LiDAR');
Y = double(gtData.gt(:));
data3D = {hsiData.HSI, lidarData.LiDAR};

[X_denoised, spLabel] = preData(data3D, dk, numPixel);

V = numel(X_denoised);
X = cell(1, V);
for v = 1:V
    X{v} = normData(X_denoised{v}');
end

ind = find(Y > 0);
c = numel(unique(Y(ind)));

tStart = tic;
y_pred = EO2C(X, alpha, lambda, c, spLabel);
tCost = toc(tStart);

rst = evalClustering(Y(ind), y_pred(ind));
metrics = rst.metric * 100;

fprintf('%s\n', formatMetrics(metrics, tCost));
end
