%% initialisation
clear all;
run ../../matlab/vl_setupnn
load('./exp/imdb.mat');
batchSize = 512;
gpus = 1;

%% find out distribution for models of different depths
%load and prepare model

load('./exp/cifar-resnet-56/net-epoch-250.mat'); 
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';

%remove last 3 layers {loss, top1error, top5error}
net.removeLayer(net.layers(end).name);
net.removeLayer(net.layers(end).name);
net.removeLayer(net.layers(end).name);

numofvars = numel(net.vars);
numoftrain = int32(sum(images.set == 1));

activation_max = [1 zeros(1, numofvars - 1, 'single')];
activation_9999 = [1 zeros(1, numofvars - 1, 'single')];
activation_999 = [1 zeros(1, numofvars - 1, 'single')];
activation_99 = [1 zeros(1, numofvars - 1, 'single')];

net.move('gpu');
traindata = gpuArray(images.data(:,:,:,1:numoftrain));
numofbatches = ceil(numoftrain / batchSize);

final = [];
for j = numofvars : -1 : 2
   
   for i = 1:numofbatches
        batchAnfang = 1 + (i - 1) * batchSize;
        batchEnde = batchAnfang + min(batchSize, numel(i * batchSize + 1:numoftrain)) - 1;
        net.eval({'input',traindata(:,:,:,batchAnfang:batchEnde)});
        final = [final; gather(net.vars(end).value(:))]; 
   end
   
   fprintf('Herr Kaleun, activations of Layer %d have been recorded. Yet, %d layers remain \n', j - 1, j -2);
   final = sort(final);

   activation_max(j) = max(final(:));
   activation_9999(j) = final(floor(numel(final)*.9999));
   activation_999(j) = final(floor(numel(final)*.999));
   activation_99(j) = final(floor(numel(final)*.99));
   final = [];
   net.removeLayer(net.layers(end).name);
end
save('res-56-activations.mat','activation_max' , 'activation_9999', 'activation_999', 'activation_99');
    












