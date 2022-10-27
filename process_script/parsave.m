function [state, str] = parsave(filename, varargin)
    % Save variables in varargin with their own name to the file specified by filename
    % filename: filename (absolute or relative path), can be with or without extend name
    % varargin: variables to save. Pass variables but not variable names directly.
    % return:
    %   state: save state, 0: unsuccessful, 1:successful
    %   str: structure used to save file.
    %       str.msg: if state == 0, str will only have one field '''msg''', to tell which happened
    if nargin == 1
        str.msg = "Expected at least 2 inputs. but got only 1. See the help document of parsave for more details";
        state = 0;
        return
    end

    filename = char(filename);

    if ~strcmp(filename( end - 3:end), '.mat')
    filename = [filename, '.mat'];
end

try

    for ii = 2:nargin
        str.(inputname(ii)) = varargin{ii - 1};
    end

    save(filename, '-struct', 'str', "-mat");
    state = 1;
catch ME
    str.msg = ME.message;
    state = 0;
end
