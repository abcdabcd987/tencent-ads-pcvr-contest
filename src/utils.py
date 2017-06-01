import os
import json


def _abs_path(dictionary, key, relative_root):
    if key in dictionary and not os.path.isabs(dictionary[key]):
        dictionary[key] = os.path.join(relative_root, dictionary[key])
    

def json_load(filename):
    l = []
    with open(filename, 'r') as f:
        for line in f:
            strip = line.strip()
            if not strip.startswith('//'):
                l.append(strip)
    return json.loads(''.join(l))


def read_module_config(script_file, config_name):
    dirname = os.path.dirname(os.path.realpath(script_file))
    filename = os.path.join(dirname, config_name)
    config = json_load(filename)
    config['module_name'] = os.path.split(dirname)[1]
    config.update(read_global_config())
    return config


def read_global_config():
    project_dir = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
    filename = os.path.join(project_dir, 'global.json')
    config = json_load(filename)
    keys = config.keys()
    for key in keys:
        if key.endswith('_dir'):
            _abs_path(config, key, project_dir)

    return config
