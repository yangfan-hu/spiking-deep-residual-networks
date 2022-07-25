function relu2if(net)
%RELU2IF replace relu layer with if neuron
%   RELU2IF(NET) replace relu neuron with if neuron. NET is a 
%   dagnn class
for l = 1 : numel(net.layers)
    
    if isa(net.layers(l).block, 'dagnn.ReLU')
%         if ~isa(net.layers(l - 1).block, 'dagnn.Sum')
%             net.renameVar(net.layers(l).name, [net.layers(l).name(1:end-4) 'if'])
%         end
%        net.renameLayer(net.layers(l).name, [net.layers(l).name(1:end-4) 'if'])
        net.layers(l).block = dagnn.If();
        net.layers(l).block.attach(net, l);
    end    
end

end

