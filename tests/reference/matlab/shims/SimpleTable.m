classdef SimpleTable
  % Just enough of MATLAB's table for best_case_TPR_FPR.m:
  % T{rows, cols} returns numbers and T.Name returns a column.
  properties
    data
    names
  end
  methods
    function obj = SimpleTable(data, names)
      obj.data = data;
      obj.names = names;
    end
    function out = subsref(obj, s)
      switch s(1).type
        case '{}'
          out = obj.data(s(1).subs{:});
        case '.'
          col = find(strcmp(obj.names, s(1).subs));
          if isempty(col)
            out = obj.(s(1).subs);  % a property such as data
          else
            out = obj.data(:, col);
          end
        otherwise
          out = builtin('subsref', obj, s(1));
      end
      if numel(s) > 1
        out = subsref(out, s(2:end));
      end
    end
  end
end
