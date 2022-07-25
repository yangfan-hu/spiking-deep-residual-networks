%% initialisation
run ../../matlab/vl_setupnn
mainDir = 'mytransexp';
depth = 44;

load([mainDir '/cifar-resnet-' num2str(depth) '/net-epoch-113.mat']); 
net = dagnn.DagNN.loadobj(net);
net.mode = 'test'; %load model
load( [mainDir '/imdb.mat'] ); %load data

opts.dt = 0.001;
opts.duration = 0.350;
opts.max_rate = 400;
batchSize = 1000;
numoftest = 10000;
gpus = [];

global simclock;    %global simulation clock
simclock = 0;
%% data-based normalisation
load([mainDir '/res-' num2str(depth) '-centre-activations.mat']); %load sampled activation data
max_activation = activation_999;  


for l = 1:numel(net.layers)
   if isa(net.layers(l).block,'dagnn.Conv')
       if strcmp(net.layers(l).name,'prediction')
           %fc layer 
           lambda_previous = max_activation(net.getVarIndex(net.layers(l - 2).outputs));
           lambda = max_activation(net.getVarIndex(net.layers(l).outputs));
           %fprintf('%s lambda-1 %d lambda %d\n',net.layers(l).name, lambda_previous, lambda);
           %normalise weights
           net.params(net.layers(l).paramIndexes(1)).value = net.params(net.layers(l).paramIndexes(1)).value .* lambda_previous ./ lambda;
           %normalise bias
           net.params(net.layers(l).paramIndexes(2)).value = net.params(net.layers(l).paramIndexes(2)).value ./ lambda;          
       else
           %conv layer
           lambda_previous = max_activation(net.getVarIndex(net.layers(l).inputs));
           if isa(net.layers(l + 2).block, 'dagnn.ReLU')  %regular layer    
               lambda = max_activation(net.getVarIndex(net.layers(l + 2).outputs));
           elseif isa(net.layers(l + 3).block, 'dagnn.ReLU') %conv before sum
               lambda = max_activation(net.getVarIndex(net.layers(l + 3).outputs));
           elseif isa(net.layers(l + 8).block, 'dagnn.ReLU') %adapter layer
               lambda = max_activation(net.getVarIndex(net.layers(l + 8).outputs));
           end        
           %fprintf('%s lambda-1 %d lambda %d\n',net.layers(l).name, lambda_previous, lambda);
           %normalise weights
           net.params(net.layers(l).paramIndexes).value = net.params(net.layers(l).paramIndexes).value .* lambda_previous ./ lambda;
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
            adapter(i).lambda_previous = max_activation(net.getVarIndex(net.layers(l - 6).outputs));
            i = i + 1;
        end
    end
    
end


for i = 1:numel(adapter)
    targetname = adapter(i).tarname;
    sumname = adapter(i).sumname;
    name = [adapter(i).sumname(1:end - 3) 'scale'];
    factor = adapter(i).lambda_previous / adapter(i).lambda;
    params = {[adapter(i).sumname(1:end - 3) 'scale' '_f']};    
    insertAdapter(net, targetname, sumname, name, dagnn.Scale('hasBias', false), factor, params);
end


fprintf('Herr Kaleun, parameters are normalised!\n');
%save('normalised_model.mat','net');


%% convert ANN to SNN
% Replace ReLU with IF neuron and change names
relu2if(net);
%fprintf('Layer number: %d Type: %s \n', l, class(net.layers(l).block));

% Insert IF neuron after Pooling Layer 
insertLayer(net, 'prediction_avg', 'prediction_avg_if', dagnn.If());

%remove loss
net.removeLayer('top5error');
net.removeLayer('top1error');
net.removeLayer('loss');

%add output neuron 
net.addLayer('prediction_if',dagnn.If(),'prediction','prediction_if');
net.layers(end).block.verbose = 1;

fprintf('Herr Kaleun, conversion is completed!\n')

%% simple test
% for i = 1:numel(net.vars)
%    net.vars(i).precious = 1;
% end

net.move('gpu');
net.mode =  'test';

testdata = gpuArray(images.data(5:36,5:36,:,images.set == 3));  %important select centre 
truth = gpuArray(images.labels(images.set==3));
result = gpuArray(zeros(size(truth)));

maxbatch = ceil(numoftest / batchSize);


tic;
for batchnum = 1 : maxbatch
    resetneuron(net);
    batchAnfang = 1 + (batchnum - 1) * batchSize;
    if batchnum == maxbatch
        batchEnde = numoftest;
    else
        batchEnde = batchnum * batchSize;
    end

    for simclock = opts.dt:opts.dt:opts.duration
     net.eval({'input',testdata(:,:,:,batchAnfang:batchEnde)});
    end
    
    [~, class] = max(net.layers(end).block.sum_spikes + vl_nnsoftmax(net.layers(end).block.membrane), [], 3);
    result(batchAnfang:batchEnde) = class;
    fprintf('Herr Kaleun, batch %d finished, %d batches in total\n ', batchnum, maxbatch);
end
toc;

fprintf('Overall Accuracy: %2.2f %% \n', sum(truth(1:numoftest) == result(1:numoftest)) / numel(truth(1:numoftest)) * 100 );


