function hostname = get_hostname()
[st, hostname] = system('hostname');
hostname = hostname(1:end-1);