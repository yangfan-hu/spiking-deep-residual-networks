%Added by Lucian
%A layer of IF neuron
%A version for less memory and better speed
classdef If < dagnn.Layer
  properties
    spikes = [];
    membrane = [];
    sum_spikes = [];
    ref_end = [];
    simclock = [];
    flag = 1;
    init_m = 0.5
    verbose = 1; %discard intermediate results by default
    opts = {'ref',0,'threshold',1}
  end
  methods
    function outputs = forward(obj, inputs, params)
      if obj.flag
          obj.flag = 0;
          obj.membrane = obj.init_m + zeros(size(inputs{1}),'single');
          if obj.verbose %
              obj.spikes = zeros(size(inputs{1}),'single');
              obj.sum_spikes = zeros(size(inputs{1}),'single');
          end
          %obj.ref_end = zeros(size(inputs{1}),'single'); %ref = 0, always
          %obj.simclock = zeros(1,1,'single'); %initialise clock
      end
      %obj.simclock = obj.simclock + 0.001; %move the clock forward, 1ms per step
      %inputs{1}(obj.ref_end > obj.simclock) = 0;
      obj.membrane = obj.membrane + inputs{1};
      
      if obj.verbose
          obj.spikes = obj.membrane >= obj.opts{4};
          %obj.membrane(obj.spikes) = 0; %obsoleted, reset by zero
          obj.membrane(obj.spikes) = obj.membrane(obj.spikes) - obj.opts{4}; %reset by subtraction
          %obj.ref_end(obj.spikes) = obj.simclock + obj.opts{2};
          obj.sum_spikes = obj.sum_spikes + obj.spikes;
          obj.spikes = single(obj.spikes); %convert to single 
          outputs{1} = obj.spikes;
      else % verbose = 0
          z = obj.membrane >= obj.opts{4}; %z for spikes
          obj.membrane(z) = obj.membrane(z) - obj.opts{4}; %reset by subtraction
          outputs{1} = single(z);          
      end
    end
    
    function resetflag(obj)
        obj.flag = 1;
    end
    
    function obj = If(varargin)
        obj.load(varargin) ;
    end
    
  end
end
