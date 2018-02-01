function log_printf(log_file, varargin)
%%%
% log_printf(log_filename, ....);
%%%

fid = fopen(log_file, 'a+');
fprintf(fid, varargin{:});
fclose(fid);

fprintf(1, varargin{:});