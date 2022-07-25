function [net, info] = mytrans_mnist_beta(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'resnet' ;
opts.depth = 20 ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.mainDir = 'mytransexp';
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.expDir = fullfile(opts.mainDir,...
    sprintf('mnist-%s-%d',opts.modelType,opts.depth));
opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
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
  case 'resnet'
    net = mytrans_mnist_init('depth', opts.depth) ;          
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts)  ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;

%----------------------lsuv init section---------------------%
if opts.lsuvinit
    lsuv_init(net, imdb, net.meta.trainOpts.batchSize, false) ;
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

if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))];
data = single(reshape(cat(3, x1, x2),28,28,1,[]));

dataMean = mean(reshape(data(:,:,:,set == 1), [], 1 ));
dataStd = std(reshape(data(:,:,:,set == 1), [], 1 ), 0, 1);
data(:,:,:,:) = data(:,:,:,:) -  dataMean;
data(:,:,:,:) = data(:,:,:,:) / dataStd;

imdb.images.data = data ;
imdb.images.data_mean = dataMean;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;


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
    net.params(index).value = single(svd_orthonromal(size(net.params(index).value)));
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
        inputs = getDagNNBatch(struct('numGpus', 0), imdb, batch);
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
fprintf('Herr Kaleun, LSUV init finished!\n')


% -------------------------------------------------------------------------
function q = svd_orthonromal(shape)
% -------------------------------------------------------------------------
% orthonormal init
    if length(shape) < 2
        error('Only shapes of length 2 or more are supported');
    end
    flat_shape = [prod(shape(1:end-1)) shape(end)];
    a = randn(flat_shape);
    [u,~,v] = svd(a, 'econ');
    % choose u or v
    if isequal(size(u), flat_shape)
        q = u;
    else
        q = v;
    end
    q = reshape(q, shape);


