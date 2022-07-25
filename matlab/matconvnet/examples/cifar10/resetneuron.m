function resetneuron(net)
%RESETNEURON Summary of this function goes here
%   Detailed explanation goes here
for l = 1 : numel(net.layers)    
    if isa(net.layers(l).block, 'dagnn.If')
        net.layers(l).block.resetflag();
    elseif isa(net.layers(l).block, 'dagnn.Pifi')
        net.layers(l).block.resetflag();
    elseif isa(net.layers(l).block, 'dagnn.Pif')
        net.layers(l).block.resetflag();    
    end    
end

end

