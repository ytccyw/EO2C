function [X, spLabel, nRow, nCol] = preData(data3D, dk, numPixel)
%PREDATA Build ERS superpixels and apply S3-PCA denoising.

viewCount = numel(data3D);
stackedData = data3D{1};
for v = 2:viewCount
    stackedData = cat(3, stackedData, data3D{v});
end

[nRow, nCol, dim] = size(stackedData);
pixels = reshape(stackedData, nRow * nCol, dim);
[pixels, ~] = mapminmax(pixels);

coeff = pca(pixels);
firstComponent = pixels * coeff(:, 1);
superpixelImage = im2uint8(mat2gray(reshape(firstComponent, nRow, nCol)));

spLabel = mex_ers(double(superpixelImage), numPixel) + 1;

X = cell(1, viewCount);
for v = 1:viewCount
    [~, ~, viewDim] = size(data3D{v});
    filteredData = S3PCA(data3D{v}, dk, spLabel);
    X{v} = reshape(filteredData, nRow * nCol, viewDim)';
end
end
