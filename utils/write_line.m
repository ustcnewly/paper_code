function write_line(log_file, str)
%%%
% write_line(log_filename, str)
%%%
dlmwrite(log_file, str, 'delimiter', '', '-append');
fprintf('%s\n',str);
