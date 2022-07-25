%Added by Lucian
%A layer of max pooling in SNN
%
classdef SPooling < dagnn.Filter
  properties
    sum_spikes = [];
    sum_spikes_pre = [];
    spikes = [];
    flag = 1;
    method = 'max';
    poolSize = [1 1];
    opts = {'cuDNN'}
  end
  methods
    function outputs = forward(obj, inputs, params)
          
      outputs{1} = vl_nnpool(inputs{1}, obj.poolSize, ...
                             'pad', obj.pad, ...
                             'stride', obj.stride, ...
                             'method', obj.method, ...
                             obj.opts{:}) ; 
      if obj.flag
          obj.flag = 0;
          obj.sum_spikes_pre = zeros(size(inputs{1}),'single');
          obj.sum_spikes = zeros(size(outputs{1}),'single');
          obj.spikes = zeros(size(outputs{1}),'single');
      end      
   
      obj.sum_spikes_pre = obj.sum_spikes_pre + inputs{1};
      
      z = vl_nnpool(obj.sum_spikes_pre, obj.poolSize, ...
                             'pad', obj.pad, ...
                             'stride', obj.stride, ...
                             'method', obj.method, ...
                             obj.opts{:}) ;       
 
     outputs{1}(z <= obj.sum_spikes) = 0;  %supress                     
     obj.sum_spikes = obj.sum_spikes + outputs{1};
     obj.spikes = outputs{1};
      
    end
    
    function resetflag(obj)
        obj.flag = 1;
    end
    
    function obj = SPooling(varargin)
        obj.load(varargin) ;
    end
    
  end
end
