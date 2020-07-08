import yaml
from subprocess import call

file = open('EXPERIMENTS')
try:
    experiments = yaml.full_load(file)
except:
    raise

for item, doc in experiments.items():
    args_list = []
    args_list.extend(doc['_configs'].values())
    args_list.append(len(doc['_stocks']))
    args_list.extend(doc['_stocks'])
    args_list.extend(doc['_windows'])
    args_list.append(doc['_start_year'])
    args_list.append(doc['_end_year'])
    exec = ["python3", "resam.py"]
    for i in args_list:
        exec.append(str(i))
    print("[PIPELINE] Executing experiment " + item)
    call(exec)