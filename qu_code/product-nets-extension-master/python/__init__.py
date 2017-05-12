import socket
config = {}

host = socket.gethostname()
config['host'] = host.lower()
'''
if config['host'] == 'altria':
    config['data_path'] = '/home/kevin/Documents/Ads-RecSys-Datasets'
elif config['host'] == 'hpc-csyy':
    config['data_path'] = '/lustre/home/acct-csyy/csyy/qyr_test/Ads-RecSys-Datasets'
elif 'quyanru' in config['host']:
    config['data_path'] = '/home/quyanru/Ads-RecSys-Datasets'
elif config['host'] == 'noah':
    config['data_path'] = '/home/tangruiming/qu/Ads-RecSys-Datasets'
'''
config['data_path'] = '/home/jianhua/Desktop/contest/Ads-RecSys-Datasets'

config['dtype'] = 'float32'
config['minval'] = -0.001
config['maxval'] = 0.001
config['mean'] = 0
config['stddev'] = 0.001
