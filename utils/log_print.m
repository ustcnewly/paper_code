function log_print(log_file, varargin)
%%%
% log_print(log_filename, ....);
%%%
fid = fopen(log_file, 'a+');
fprintf(fid, varargin{:});
fclose(fid);
