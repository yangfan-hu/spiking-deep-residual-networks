%% initialisation
clear all;
run ../../matlab/vl_setupnn

opts.expDir = './resnet/';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.batchSize = 32;
opts.gpus = 2;
opts.numFetchThreads = 12 ;
opts.Anfang = 40001;
opts.Ende = 50000;


%imreadjpeg parameters
opts.cropSize = 224 / 256 ;
opts.imageSize = [224, 224] ;
opts.meanImg = [0.485, 0.456, 0.406] ;
opts.std = [0.229, 0.224, 0.225] ;


imdb = load(opts.imdbPath) ;
net = load('resnet50-pt-mcn');
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';

opts.dt = 0.001;
opts.duration = 0.350;
opts.max_rate = 400;

global simclock;    %global simulation clock
simclock = 0;

%% data-based normalisation
load([opts.expDir 'resnet50-pt-mcn-activations']); %load sampled activation data
max_activation = activation_9999;  
compensation = 1;

for l = 1:numel(net.layers)
   if isa(net.layers(l).block,'dagnn.Conv')
       if strcmp(net.layers(l).name,'classifier_0')
           %fc layer 
           lambda_previous = max_activation(net.getVarIndex(net.layers(l - 2).outputs));
           lambda = max_activation(net.getVarIndex(net.layers(l).outputs));
           %fprintf('%s lambda-1 %d lambda %d\n',net.layers(l).name, lambda_previous, lambda);
           %normalise weights
           net.params(net.layers(l).paramIndexes(1)).value = net.params(net.layers(l).paramIndexes(1)).value .* lambda_previous ./ lambda * compensation;
           %normalise bias
           net.params(net.layers(l).paramIndexes(2)).value = net.params(net.layers(l).paramIndexes(2)).value ./ lambda;                  
       else
           %conv layer
           if l == 5 || l == 13 %first conv and adapter
               lambda_previous = max_activation(net.getVarIndex(net.layers(l).inputs) - 1); % locate the ReLU before max pool
           else
               lambda_previous = max_activation(net.getVarIndex(net.layers(l).inputs));
           end
           
           if isa(net.layers(l + 2).block, 'dagnn.ReLU')  %regular layer    
               lambda = max_activation(net.getVarIndex(net.layers(l + 2).outputs));
           elseif isa(net.layers(l + 3).block, 'dagnn.ReLU') %3rd conv before sum or adapter before sum
                lambda = max_activation(net.getVarIndex(net.layers(l + 3).outputs));
           elseif isa(net.layers(l + 5).block, 'dagnn.ReLU') %3rd conv before adapter 
               lambda = max_activation(net.getVarIndex(net.layers(l + 5).outputs));
           else
               error('Alarm! Feindliche Zerstoerer!');
           end      
           %fprintf('%s lambda-1 %d lambda %d\n',net.layers(l).name, lambda_previous, lambda);
           %normalise weights
           net.params(net.layers(l).paramIndexes).value = net.params(net.layers(l).paramIndexes).value .* lambda_previous ./ lambda * compensation;
           %normalise batch norm beta
           net.params(net.layers(l + 1).paramIndexes(2)).value = net.params(net.layers(l + 1).paramIndexes(2)).value ./ lambda;
           %mornalise batch norm miu
           net.params(net.layers(l + 1).paramIndexes(3)).value(:,1) = net.params(net.layers(l + 1).paramIndexes(3)).value(:,1) ./ lambda;
           
       end
       
   end   
end

%find and insert scale adapter

i = 1;
for l = 1:numel(net.layers)
    if isa(net.layers(l).block, 'dagnn.Sum')
        inputindex = net.getVarIndex(net.layers(l).inputs{1}) - 1;
        
        if isa(net.layers(inputindex).block, 'dagnn.ReLU') 
            adapter(i).tarname = net.layers(inputindex).name;
            adapter(i).sumname = net.layers(l).name;     
            adapter(i).lambda = max_activation(net.getVarIndex(net.layers(l + 1).outputs));
            adapter(i).lambda_previous = max_activation(net.getVarIndex(net.layers(l - 9).outputs));    %resblock of 3  
            i = i + 1;
        end
    end
    
end


for i = 1:numel(adapter)
    targetname = adapter(i).tarname;
    sumname = adapter(i).sumname;
    name = [adapter(i).sumname 'scale'];
    factor = adapter(i).lambda_previous / adapter(i).lambda * compensation ;
    params = {[adapter(i).sumname 'scale' '_f']};    
    insertAdapter(net, targetname, sumname, name, dagnn.Scale('hasBias', false), factor, params);
end

fprintf('Herr Kaleun, parameters are normalised!\n');

%% convert ANN to SNN
% Replace ReLU with IF neuron and change names
relu2if(net);
%fprintf('Layer number: %d Type: %s \n', l, class(net.layers(l).block));

%Insert IF neuron after Max Pooling layer change to max SPooling
index = net.getLayerIndex('features_3');
net.layers(index).block = dagnn.SPooling( ...
    'poolSize', net.layers(index).block.poolSize, ...
    'stride', net.layers(index).block.stride, ...
    'pad', net.layers(index).block.pad, ...
    'method', net.layers(index).block.method);
net.layers(index).block.attach(net, index);

% Insert IF neuron after Avg Pooling Layer 
insertLayer(net, 'features_8', 'features_8_if', dagnn.If());

%add output neuron 
net.addLayer('classifier_0_if',dagnn.If(),'classifier_0','classifier_0_if');
net.layers(end).block.verbose = 1;
fprintf('Herr Kaleun, conversion is completed!\n');
%% Prepare Data

opts.test = struct(...
  'useGpu', numel(opts.gpus) > 0, ...  
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  [224 224], ...
  'CropSize', 224 / 256 ) ;
%we will subtract the average later

validation = imdb.images.set == 2;

truth = imdb.images.label(validation);
imagesDir = imdb.images.name(validation);

fprintf('Herr Kalen, data preparation is finished \n');

%% simple test

net.move('gpu');

top1count = gpuArray(0);
top5count = gpuArray(0);

maxbatch = ceil(numel(opts.Anfang:opts.Ende) / opts.batchSize);

tic;

for batchnum = 1 : maxbatch
    
    batchAnfang = 1 + (batchnum - 1) * opts.batchSize;
    if batchnum == maxbatch
        batchEnde = numel(opts.Anfang:opts.Ende);
    else
        batchEnde = batchnum * opts.batchSize;
    end
    
    %prepare data for batch
    images = strcat([imdb.imageDir filesep], imagesDir(opts.Anfang - 1 + batchAnfang : opts.Anfang - 1 + batchEnde)) ;
    data = getImageBatch(images, opts.test); 
    %normalise to [0 1]
    data = data / 255;
    %subtract mean
    data = bsxfun(@minus, data, permute(opts.meanImg, [1 3 2])) ;
    %divide standard deviation 
    data = bsxfun(@rdivide, data, permute(opts.std, [1 3 2])) ;   
    
    label = truth(opts.Anfang - 1 + batchAnfang : opts.Anfang - 1 + batchEnde);    

    resetneuron(net);
    for simclock = opts.dt:opts.dt:opts.duration
     net.eval({'data',data});
    end    
    score = net.layers(end).block.sum_spikes + vl_nnsoftmax(net.layers(end).block.membrane) ;
    top1count = top1count + vl_nnloss(score,label,'loss','classerror') ;
    top5count = top5count + vl_nnloss(score,label,'loss','topkerror', 'topK', 5) ; 
 
    fprintf('Herr Kaleun, batch %d finished, %d batches in total\n ', batchnum, maxbatch);
end

toc;

fprintf('top1count %d top5count %d\n',top1count, top5count);
fprintf('Top1 Err: %2.4f%%, Top5 Err: %2.4f%% \n',top1count / numel(opts.Anfang:opts.Ende) * 100, top5count / numel(opts.Anfang:opts.Ende) * 100);
