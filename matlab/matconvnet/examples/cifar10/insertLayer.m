function insertLayer(net, targetname, name, block, params)
% INSERTLAYER insert a layer to dagnn
%   INSERTLAYER(TARGETNAME, NAME, LAYER, INPUTS, OUTPUTS, PARAMS) inserts 
%   the specified layer to the network. TARGETNAME is a string with the 
%   name of the target layer, after which your layer will be inserted.
%   NAME is a string with the layer name, used as a unique indentifier. 
%   BLOCK is the netect implementing the layer, which should be a 
%   subclass of the Layer. INPUTS, OUTPUTS are cell arrays of variable 
%   names, and PARAMS of parameter names. NET is a dagnn class
% 
%   Warning: this function takes the postulation that target layer
%       outputs only to its following layer and all layers output 
%       to no more than one following layer. Inputs and outputs of
%       inserted intermediate layer are certain. Intermediate layer
%       has no parameters
index = net.getLayerIndex(targetname);

inputs =  net.layers(index).outputs;    
outputs = name; 

%add the layer
if nargin < 5, params = {} ; end
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
index = net.getLayerIndex(targetname);
net.setLayerInputs(net.layers(index + 2).name, net.layers(index + 1).outputs);

end