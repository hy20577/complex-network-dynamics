classdef MatlabString
  % Octave has no string type, so "a" == "b" compares characters one by one.
  % Wrapping a value in MatlabString makes == compare whole strings, as
  % MATLAB does for its string objects (used by Mir_surrogate_data.m).
  properties
    value
  end
  methods
    function obj = MatlabString(v)
      obj.value = char(v);
    end
    function r = eq(a, b)
      if isa(a, 'MatlabString'), a = a.value; end
      if isa(b, 'MatlabString'), b = b.value; end
      r = strcmp(a, b);
    end
  end
end
