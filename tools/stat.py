import pandas as pd
import os
import ast
import time
import traceback

exps = ['exp1']
TRAINABLE_PARAMETER = "Trainable parameters"

def parse(exp):
    LOG_DICT = "./" + exp
    data = []
    filename_index = []
    for logfile in os.listdir(LOG_DICT):
        if '.log' not in logfile:
            continue

        filename_index.append(logfile.split('-')[0])
        map = {}
        try:
            with open(os.path.join(LOG_DICT, logfile), 'r') as f:
                line = f.readline()
                while line:
                    if line.__contains__("=") and line[0] != ' ':
                        split_line = line.split()
                        if len(split_line) == 3:
                            map[split_line[0]] = split_line[2]
                        elif len(split_line) == 2:
                            map[split_line[0]] = ''
                    elif line.__contains__(TRAINABLE_PARAMETER):
                        split_line = line.strip().split(":")
                        map['params'] = split_line[1]
                    elif line.__contains__("test result:"):
                        result_string = line.split("test result")[1][2:-1]
                        result_map = ast.literal_eval(result_string)
                        map.update(result_map)
                    elif line.__contains__("Saving current"):
                        map['model'] = line.split('/')[-1].split('-')[0]
                    elif line.__contains__("best eval result in epoch"):
                        map['epoch'] = line.split(' ')[-1]
                    line = f.readline()
                f.close()
        except Exception:
            traceback.print_exc()
            print('Error ', exp, logfile)
            filename_index.pop()
        else:
            data.append(map)

    df = pd.DataFrame(data, filename_index)
    # df.to_csv('./'+ exp + '_' + time.strftime('%Y%m%d_%H-%M') + '.csv')
    cf = './'+ exp + '.csv'
    df.to_csv(cf)
    print('Stat for', exp, 'as', cf)


if __name__ == '__main__':
    for exp in exps:
        parse(exp)