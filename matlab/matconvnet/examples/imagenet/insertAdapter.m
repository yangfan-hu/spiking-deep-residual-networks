function insertAdapter(net, targetname, sumname, name, block, factor, params)
% INSERTLAYER insert a layer to dagnn
%   INSERTLAYER(TARGETNAME, NAME, LAYER, INPUTS, OUTPUTS, PARAMS) inserts 
%   the specified layer to the network. TARGETNAME is a string with the 
%   name of the target layer, after which your layer will be inserted.
%   SUMNAME is the name of the sum layer, to which the output of inserted 
%   layer will be fed to. NAME is a string with the layer name, used as a 
%   unique indentifier.   BLOCK is the netect implementing the layer, which 
%   should be a   subclass of the Layer. INPUTS, OUTPUTS are cell arrays of 
%   variable names, and PARAMS of parameter names. NET is a dagnn class
% 
%   Warning: this function is designed only to work within a residual block,
%       adding an adapter between input and sum layer.
      
index = net.getLayerIndex(targetname);

inputs =  net.layers(index).outputs;    
outputs = name; 

%add the layer
if nargin < 7, params = {[name '_f']} ; end
net.addLayer(name, block, inputs, outputs, params);



%shift the layer
templayer = net.layers(end);
for l = numel(net.layers):-1:index + 2
    net.layers(l) = net.layers(l - 1);
    net.layers(l).block.attach(net, l);
end
net.layers(l - 1) = templayer;
net.layers(l - 1).block.attach(net, l - 1);


index = net.getVarIndex(net.layers(index).outputs);
%shift the var
tempvar = net.vars(end);
for l = numel(net.vars):-1:index + 2
    net.vars(l) = net.vars(l - 1);
end
net.vars(l - 1) = tempvar;
net.rebuild;

% adjust inputs
sum_inputs = net.layers(net.getLayerIndex(sumname)).inputs;
sum_inputs(1) = net.layers(net.getLayerIndex(name)).outputs; 
net.setLayerInputs(sumname, sum_inputs);
%setParam = 1
net.params(net.layers(net.getLayerIndex(name)).paramIndexes).value = 1 * factor;

end