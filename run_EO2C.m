clear; clc; close all;
warning off;

rootDir = fileparts(mfilename('fullpath'));
addpath(fullfile(rootDir, 'functions'));

dk = 28;
numPixel = 180;
alpha = 2^-9;
lambda = 2^-6;

data = load(fullfile(rootDir, 'datasets', 'trento_data.mat'), 'gt', 'HSI', 'LiDAR');
Y = double(data.gt(:));
data3D = {data.HSI, data.LiDAR};

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
