% -------------------------------------------------------------------------
%                                                            initialisation
% -------------------------------------------------------------------------
%run this file in /examples/imagenet
run ../../matlab/vl_setupnn

opts.expDir = './resnet/';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.depth = 152;

opts.numPerClass = 5;
opts.numOfClasses = 1000;
opts.batchSize = 100;
opts.gpus = 2;
opts.numFetchThreads = 12 ;

%imreadjpeg parameters
opts.cropSize = 224 / 256 ;
opts.imageSize = [224, 224] ;
opts.meanImg = [0.485, 0.456, 0.406] ;
opts.std = [0.229, 0.224, 0.225] ;

imdb = load(opts.imdbPath) ;
net = load(['resnet' num2str(opts.depth) '-pt-mcn']);
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';
fprintf('Herr Kaleun, initialisation is finished!\n');

% -------------------------------------------------------------------------
%                                                               Preparation
% -------------------------------------------------------------------------

activation_max = [1 zeros(1, numel(net.vars) - 1, 'single')];
activation_9999 = [1 zeros(1, numel(net.vars) - 1, 'single')];
activation_999 = [1 zeros(1, numel(net.vars) - 1, 'single')];
activation_99 = [1 zeros(1, numel(net.vars) - 1, 'single')];


train = imdb.images.set == 1;
specimen = zeros(1,opts.numPerClass * opts.numOfClasses);

for i = 1:opts.numOfClasses    
    pictures = find(imdb.images.label(train) == i);    %number of pictures in this class    
    shuffle = int32(randperm(numel(pictures)));
    specimen(1 + opts.numPerClass * (i - 1) : opts.numPerClass * i) = sort(pictures(shuffle(1:opts.numPerClass)));  % sort not necessary I knew
end

opts.test = struct(...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  [224 224], ...
  'CropSize', 224/256 ) ;

images = strcat([imdb.imageDir filesep], imdb.images.name(specimen)) ;
data = getImageBatch(images, opts.test) ;

%normalise to [0 1]
data = data / 255;
%subtract mean
data = bsxfun(@minus, data, permute(opts.meanImg, [1 3 2])) ;
%divide standard deviation 
data = bsxfun(@rdivide, data, permute(opts.std, [1 3 2])) ;

fprintf('Herr Kaleun, data preparation is finished!\n');

% -------------------------------------------------------------------------
%                                                                       Run
% -------------------------------------------------------------------------

if numel(opts.gpus) < 1
    error('ERROR! GPUs are required!');
end

net.move('gpu');

for l = numel(net.vars) : -1 : 2
    maps = [];
    num_of_batch = ceil(numel(specimen) / opts.batchSize);
    
    for index = 1 : num_of_batch;
        batchAnfang = 1 + (index - 1) * opts.batchSize; 
        batchEnde = max(index * opts.batchSize, numel(specimen) * (index == num_of_batch));
        %batch = specimen(batchAnfang : batchEnde);
        testdata = gpuArray(data(:,:,:,batchAnfang : batchEnde)) ;
        net.eval({'data',testdata});
        maps = [maps; gather(net.vars(end).value(:))];
    end    

    %fprintf('Herr Kaleun, activations of Layer %d have been recorded. Yet, %d layers remain \n', l - 1, l - 2);
    fprintf('Herr Kaleun, activations of Layer %d have been recorded. Yet, %d layers remain \n', numel(net.layers), numel(net.layers) - 1);   
    maps = sort(maps);
    activation_max(l) = max(maps(:));
    activation_9999(l) = maps(floor(numel(maps)*.9999));
    activation_999(l) = maps(floor(numel(maps)*.999));
    activation_99(l) = maps(floor(numel(maps)*.99));    
    fprintf('Herr Kaleun, statistical calculation is finished!\n');
    maps = [];
    net.removeLayer(net.layers(end).name);
end

save([opts.expDir 'resnet' num2str(opts.depth) '-pt-mcn-activations'],'activation_max' , 'activation_9999', 'activation_999', 'activation_99');