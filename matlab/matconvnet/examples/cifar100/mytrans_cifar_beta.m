function [net, info] = mytrans_cifar_beta(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-100
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'resnet' ;
opts.depth = 20 ;
[opts, varargin] = vl_argparse(opts, varargin) ;

% opts.expDir = fullfile(vl_rootnn, 'data', ...
%   sprintf('cifar-%s', opts.modelType)) ;
opts.mainDir = 'mytransexp';
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(opts.mainDir,...
    sprintf('cifar-%s-%d',opts.modelType,opts.depth));
opts.dataDir = fullfile(vl_rootnn, 'data','cifar') ;
opts.imdbPath = fullfile(opts.mainDir, 'imdb.mat');
opts.featureStdNormalization = true ; %norm by features
opts.contrastNormalization = false ;  %norm by samples
opts.whitenData = false ;
opts.lsuvinit = true;

opts.networkType = 'dagnn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

switch opts.modelType
  case 'lenet'
    net = cnn_cifar_init('networkType', opts.networkType) ;
  case 'nin'
    net = cnn_cifar_init_nin('networkType', opts.networkType) ;
  case 'resnet'
    net = mytrans_cifar_init('depth', opts.depth ,'nClass' ,100) ;          
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCifarImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;

%----------------------lsuv init section---------------------%
if opts.lsuvinit
    lsuv_init(net, imdb, net.meta.trainOpts.batchSize, opts.train.gpus, false) ;
end

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    %fn = @(x,y) getSimpleNNBatch(x,y) ;
    error('simplenn not supported for resnet') ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% % -------------------------------------------------------------------------
% function [images, labels] = getSimpleNNBatch(imdb, batch)
% % -------------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;
% %if rand > 0.5, images=fliplr(images) ; end   #original
% if rand > 0.5, images = flipdim(images,2) ; end   %#modified by lucian
% %pad image with 4 pixels and randomly crop from it
% crop_index = randi(9,1,2);
% images = images(crop_index(1):crop_index(1) + 31 , crop_index(2):crop_index(2) + 31 ,:,:);


% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=flipdim(images,2) ; end
%random crop for each image
%images = cropRand(images);
crop_index = randi(9,1,2);
images = images(crop_index(1):crop_index(1) + 31 , crop_index(2):crop_index(2) + 31 ,:,:);

if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-100-matlab');
files = [{'train.mat'} {'test.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([1, 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-100-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
coarselabels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.fine_labels' + 1; % Index from 1
  coarselabels{fi} = fd.coarse_labels' + 1;
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
%dataMean = mean(data(:,:,:,set == 1), 4);
%data = bsxfun(@minus, data, dataMean);
%commented by lucian

%pad image
data = padarray(data, [4, 4], 128, 'both') ;

%feature wise normalization zero mean and unit variance
%norm train set

for ch = 1:3
   dataMean = mean(reshape(data(:,:,ch,set == 1), [], 1 ));
   dataStd = std(reshape(data(:,:,ch,set == 1), [], 1 ), 0, 1);
   data(:,:,ch,:) = data(:,:,ch,:) -  dataMean;
   data(:,:,ch,:) = data(:,:,ch,:) / dataStd;
end

% %norm test set
% for ch = 1:3
%    dataMean = mean(reshape(data(:,:,ch,set == 3), [], 1 ));
%    dataStd = std(reshape(data(:,:,ch,set == 3), [], 1 ), 0, 1);
%    data(:,:,ch,set==3) = data(:,:,ch,set==3) -  dataMean;
%    data(:,:,ch,set==3) = data(:,:,ch,set==3) / dataStd;
% end


%sample wise normalization

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],60000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],60000) ;
  W = z(:,set == 1)*z(:,set == 1)'/60000 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.coarselabels = single(cat(2, coarselabels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.fine_label_names ;
imdb.meta.coarseclasses = clNames.coarse_label_names ;


% -------------------------------------------------------------------------
function lsuv_init(net, imdb, lsuv_batchSize, layer_wise_shuffle)
% -------------------------------------------------------------------------
% layer sequential unit variance initialization
fprintf('Herr Kaleun, starting LSUV init!\n');

tol_var = 0.1;
t_max = 10; %10 iterations at most

%shuffle

if ~layer_wise_shuffle
    shuffle_order = randperm(sum(imdb.images.set == 1));
end

%only forward
net.mode = 'test';

% move net to gpu
if numel(gpus > 0)
    gpuDevice(gpus(1));
    net.move('gpu');    
end



for l = 1:numel(net.layers)
    
    % set sigma to 1
    if isa(net.layers(l).block,'dagnn.BatchNorm')
        index = net.layers(l).paramIndexes;
        index = index(3);
        net.params(index).value(:,2) = net.params(index).value(:,2) + 1;
    end
    
    
    %conv & fc layers only
    if ~isa(net.layers(l).block,'dagnn.Conv')
        continue;
    end
    
    %
    fprintf('Herr Kaleun, we are now working on Layer %d \n', l);
    
    
    %aoivd small layers where activation variance close to zero, esp. for small batches
    %to be added
    
    
    %set precious to 1
    net.vars(net.layers(l).outputIndexes).precious = 1;
    %orthonormal init
    index = net.layers(l).paramIndexes;
    %we don't care about biases
    index = index(1);
    %change from Xavier to orthonormal init
    net.params(index).value = svd_orthonromal(size(net.params(index).value));
    %layer wise shuffle
    if layer_wise_shuffle
        shuffle_order = randperm(sum(imdb.images.set == 1));
    end
    % counter
    t_i = 1;
    while true
        %get batch
        batchAnfang = 1 + (t_i - 1) * lsuv_batchSize;
        batchEnde = lsuv_batchSize + (t_i - 1) * lsuv_batchSize;
        batch = shuffle_order(batchAnfang:batchEnde);
        inputs = getDagNNBatch(struct('numGpus', numel(gpus)), imdb, batch);
        %forward
        net.eval(inputs);
        activations = net.vars(net.layers(l).outputIndexes).value;
        variance = var(activations(:));
        fprintf('variance is %f \n', variance);
        % avoid zero division
        if abs(sqrt(variance)) < 1e-7
            fprintf('Herr Kaleun, variance too small\n');
            break
        end
        % 
        if abs(variance - 1.0) < tol_var 
            break;
        end
        
        if t_i > t_max 
            fprintf('Herr Kaleun, failed to converge\n');
            break;
        end
        %W_L = W_L / sqrt(variance)
        net.params(index).value = net.params(index).value ./ sqrt(variance); 
        t_i = t_i + 1;
    end
    %set precious to 0
    net.vars(net.layers(l).outputIndexes).precious = 0;
end

%shall we set sigma back to zero?


% set back to normal mode for preceeding training phase
net.mode = 'normal';

% move net back to cpu
if numel(gpus > 0)
    gpuDevice(1);
    net.move('cpu');    
end

fprintf('Herr Kaleun, LSUV init finished!\n')


% -------------------------------------------------------------------------
function q = svd_orthonromal(shape)
% -------------------------------------------------------------------------
% orthonormal init
    if length(shape) < 2
        error('Only shapes of length 2 or more are supported');
    end
    flat_shape = [prod(shape(1:end-1)) shape(end)];
    a = gpuArray(randn(flat_shape,'single'));
    [u,~,v] = svd(a, 'econ');
    % choose u or v
    if isequal(size(u), flat_shape)
        q = u;
    else
        q = v;
    end
    q = reshape(q, shape);


