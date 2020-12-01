import subprocess
import re

def get_vram_usage():
    '''
    Gets current GPU Memory usage, in mb
    '''
    smi_output = subprocess.run("nvidia-smi", stdout=subprocess.PIPE).stdout.decode('utf-8')
    important_ix = smi_output.index('MiB')
    good_output = smi_output[important_ix-20:important_ix+30]
    spl = good_output.split('|')
    important = None
    for each_seg in spl:
        if 'MiB' in each_seg:
            important = each_seg
            break
    curr_str, max_str = important.replace("MiB","").split('/')
    curr_i = int(curr_str)
    max_i = int(max_str)
    return curr_i, max_i
