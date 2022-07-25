function net = mytrans_mnist_init(varargin)

opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.networkType = 'dagnn' ;
opts.modelType='resnet' ;
opts.depth = 20 ; %20,32,44,56,110 in paper
opts.nClass = 10;
opts.bottleneck = false;
opts = vl_argparse(opts, varargin) ;

n = (opts.depth - 2) / 6;
if rem(n,1);
    error('wrong depth, depth must 20,32,44,46,110,etc.');
end
net = dagnn.DagNN() ;
blocknumber = 1;
lastAdded.var = 'input' ;
lastAdded.depth = 1 ;

% -------------------------------------------------------------------------
% Add first conv, batchnorm and relu
% -------------------------------------------------------------------------

Conv('conv1', 3, 16, ...
     'relu', true, ...
     'bias', false, ...
     'downsample', false) ;


% -------------------------------------------------------------------------
% Add residual blocks
% -------------------------------------------------------------------------

% 3 resblock
if ~opts.bottleneck
    add_basicblock(16, n, 1);
    add_basicblock(32, n, 2);
    add_basicblock(64, n, 2);
else
    add_bottleneck(16, n, 1);
    add_bottleneck(32, n, 2);
    add_bottleneck(64, n, 2);
end

%global average
net.addLayer('prediction_avg' , ...
             dagnn.Pooling('poolSize', [7 7], 'method', 'avg'), ...
             lastAdded.var, ...
             'prediction_avg') ;
         
%fc basic 64-10  bottleneck 256-10
net.addLayer('prediction' , ...
             dagnn.Conv('size', [1 1 (64 * ~opts.bottleneck + 64 * 4 * opts.bottleneck) opts.nClass]), ...
             'prediction_avg', ...
             'prediction', ...
             {'prediction_f', 'prediction_b'}) ;

net.addLayer('loss', ...
             dagnn.Loss('loss', 'softmaxlog') ,...
             {'prediction', 'label'}, ...
             'objective') ;

net.addLayer('top1error', ...
             dagnn.Loss('loss', 'classerror'), ...
             {'prediction', 'label'}, ...
             'top1error') ;

net.addLayer('top5error', ...
             dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
             {'prediction', 'label'}, ...
             'top5error') ;

% -------------------------------------------------------------------------
%                                                           Meta parameters
% -------------------------------------------------------------------------
net.meta.inputSize = [32 32 3] ;
lr = [0.01*ones(1,3) 0.1*ones(1,80) 0.01*ones(1,10) 0.001*ones(1,20)] ;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.batchSize = 128 ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.classes.name = opts.classNames ;
net.meta.classes.description = opts.classDescriptions ;
% Init parameters randomly
net.initParams() ;


function Conv(name, ksize, depth, varargin)
% Helper function to add a Convolutional + BatchNorm + ReLU
% sequence to the network.
  args.relu = true ;
  args.downsample = false ;
  args.bias = false ;
  args = vl_argparse(args, varargin) ;
  if args.downsample, stride = 2 ; else stride = 1 ; end
  if args.bias, pars = {[name '_f'], [name '_b']} ; else pars = {[name '_f']} ; end
  net.addLayer([name  '_conv'], ...
               dagnn.Conv('size', [ksize ksize lastAdded.depth depth], ...
                          'stride', stride, ....
                          'pad', (ksize - 1) / 2, ...
                          'hasBias', args.bias, ...
                          'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
               lastAdded.var, ...
               [name '_conv'], ...
               pars) ;
  net.addLayer([name '_bn'], ...
               dagnn.BatchNorm('numChannels', depth, 'epsilon', 1e-5), ...
               [name '_conv'], ...
               [name '_bn'], ...
               {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
  lastAdded.depth = depth ;
  lastAdded.var = [name '_bn'] ;
  if args.relu
    net.addLayer([name '_relu'] , ...
                 dagnn.ReLU(), ...
                 lastAdded.var, ...
                 [name '_relu']) ;
    lastAdded.var = [name '_relu'] ;
  end
end

function add_basicblock(channels, n, stride, varargin)
% Helper function to add a Convolutional + BatchNorm + ReLU
% sequence to the network.

  for l = 1:n %2 conv layers per res subblock, n subblocks per resblock
         
      sectioninput = lastAdded;
      name = sprintf('conv%d_%d', blocknumber, l);
      % AB: 3x3, 3x3; downsample if first segment in section from
      %adapter 
      if l == 1
          Conv([name '_adapt_conv'], 1, channels, 'downsample', stride ==2,'relu', false) ;
      end
      sumInput = lastAdded ;
      lastAdded = sectioninput;
      %conv 1
      Conv([name 'a'] ,3, channels,'downsample', stride ==2 & l==1);
      %conv 2
      Conv([name 'b'], 3, channels,'relu', false)
      % Sum layer
      net.addLayer([name '_sum'] , ...
                   dagnn.Sum(), ...
                   {sumInput.var, lastAdded.var}, ...
                   [name '_sum']) ;
      net.addLayer([name '_relu'] , ...
                   dagnn.ReLU(), ...
                   [name '_sum'], ...
                   name) ;
      lastAdded.var = name ;      
      
  end
  blocknumber = blocknumber + 1;
end

function add_bottleneck(channels, n, stride, varargin)
% Helper function to add a Convolutional + BatchNorm + ReLU
% sequence to the network.

  for l = 1:n %2 conv layers per res subblock, n subblocks per resblock
         
      sectioninput = lastAdded;
      name = sprintf('conv%d_%d', blocknumber, l);
       % ABC: 1x1, 3x3, 1x1; downsample if first segment in section from
      %adapter for each resblock 
      if l == 1 
          Conv([name '_adapt_conv'], 1, channels * 4, 'downsample', stride ==2,'relu', false) ;
      end
      sumInput = lastAdded ;
      lastAdded = sectioninput;
      %conv 1
      Conv([name 'a'] ,3, channels);
      %conv 2
      Conv([name 'b'], 3, channels,'downsample', stride ==2 & l==1)
      %conv 3
      Conv([name 'c'], 3, channels * 4,'relu', false)
      
      % Sum layer
      net.addLayer([name '_sum'] , ...
                   dagnn.Sum(), ...
                   {sumInput.var, lastAdded.var}, ...
                   [name '_sum']) ;
      net.addLayer([name '_relu'] , ...
                   dagnn.ReLU(), ...
                   [name '_sum'], ...
                   name) ;
      lastAdded.var = name ;      
      
  end
  blocknumber = blocknumber + 1;
end

end


