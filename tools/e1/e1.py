import random
import os
log_dir = './e1/' # sasrec p new
gpu_id = 0
i = 0
group = 8 # 4
print("use start i as ", i)
grid = {
    'n_layers': [2],
    'attr_lamdas': [1,2],
    'attr_regi': [1,2],

    # @todo Dataset params should be the last in this map
    # @todo Pls don't change this rule
    'dataset': ['Amazon_Beauty'],
}


models = ['SASRecG']

params = ['']
for k, v in grid.items():
    nparams = []
    for vi in v:
        subp = '--' + k + '=' + str(vi)
        if k == 'dataset':
            # subp += ' --config_files=\"conf/config_d_' + vi + '.yaml conf/config_t_train.yaml '
            subp += ' --config_files=\"conf/' + vi + '.yaml '
        nparams += [p + ' ' + subp for p in params]

    params = nparams

random.shuffle(params)

cmds = []
for m in models:
    for p in params:
        i += 1
        cmd = 'nohup python run_recbole.py'
        cmd += ' --model=' + m
        cmd += ' ' + p
        cmd += 'conf/config_m_' + m + '.yaml\"'
        cmd += ' --gpu_id=$g'
        cmd += ' --log_wandb=True'
        cmd += ' --model_id=' + str(i)
        cmd += ' > ' + log_dir + str(i) + '.log 2>&1'
        cmds.append(cmd)



if __name__ == '__main__':
    idx, g = 0, 1
    if len(cmds) % group == 0:
        g = len(cmds) // group
    else:
        g = len(cmds) // group + 1
    for i, cmd in enumerate(cmds):
        idx = i // g + 1
        f = log_dir + str(idx) + '.sh'
        if not os.path.exists(f):
            with open(f, 'w') as file:
                file.write(cmd)
                file.write("\n")
        else:
            with open(f, 'a') as file:
                file.write(cmd)
                file.write("\n")


    with open(log_dir + 'run.sh', 'w') as file:
        for i in range(idx):
            if i % 4 == 0:
                file.write('export g=' + str(i//4) + '\n')
            file.write('nohup ' + log_dir + str(i + 1) + '.sh &\n')

    print('Scripts for ' + log_dir + ' :', idx, 'Cmds:', len(cmds))
