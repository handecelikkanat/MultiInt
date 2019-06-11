#!/usr/bin/env python

# read in a log from multi-encoder training, report perplexities for each 
# language over time in CSV format

import sys
import re
from collections import defaultdict
import ipdb
import numpy as np
import matplotlib.pyplot as plt

def main(field='perplexity'):
    "USAGE: parselog.py <train_logfile>"
    # perplexities grouped by step and language pair
    ppls = defaultdict(lambda: defaultdict(float)) # validation ppls
    accs = defaultdict(lambda: defaultdict(float)) # validation accs
    
    tppls = defaultdict(lambda: defaultdict(float)) # training ppls
    taccs = defaultdict(lambda: defaultdict(float)) # training accs
    
    # which training step we're currently in
    

    # which language pair are we evaluating
    #lang_pair = None
    step = None
    experim=None
    in_path = sys.argv[1]
    with open(in_path, 'r') as in_file:
        for line in in_file:
            experim_match = re.search("Wrote config file to (.*)", line)
            if experim_match:
                experim = experim_match.group(1)
            step_match = re.search("Step (\d+)", line)
            if step_match:
                step_ = int(step_match.group(1))
                if not step or step_ > step:
                    step = step_  # avoid matching validation-run steps
                # get training accs and ppls per step
                tppl_match =  re.search("ppl:(\s*)(\d*.\d*)", line)
                if tppl_match:
                    tppl = float(tppl_match.group(2))
                    tppls[step] = tppl
                tacc_match =  re.search("acc:(\s*)(\d*.\d*)", line)
                if tacc_match:
                    tacc = float(tacc_match.group(2))
                    taccs[step] = tacc
            ppl_match = re.search("Validation perplexity: (.*)", line)
            if ppl_match:
                ppl = float(ppl_match.group(1))
                ppls[step] = ppl
            acc_match = re.search("Validation accuracy: (.*)", line)
            if acc_match:
                acc = float(acc_match.group(1))
                accs[step] = acc
        #   db = re.search("Loading dataset from", line)

    
    # TRAINING VALUES - PLOT
    #lt.ion()
    fig, ax = plt.subplots()
    
    steps=[]; accur=[]; perps=[]; vaccur=[]; vperps=[]; vsteps=[]
    for key in tppls.keys():
        steps.append(key)
        perps.append(tppls[key])
        accur.append(taccs[key])
        for step in sorted(ppls.keys()):
            if step == key:
                vsteps.append([key])
                vperps.append(ppls[key])
                vaccur.append(accs[key])
    
    #ipdb.set_trace()
    # only plot for smaller perplexities! 
    values = [i for i, x in enumerate(np.array(perps) >= 100) if x]
    num=values[-1]+1
    line1 = plt.plot(steps[num:], perps[num:], '-', label="train perplexity")#, color='lightcoral')
    line2 = plt.plot(steps[num:], accur[num:], '-', label="train accuracy")
    #ax.set_yscale('log')
    dots1 = plt.plot(vsteps, vperps, 'o:', label = 'valid. perplexity')
    dots1 = plt.plot(vsteps, vaccur, '^:', label= 'valid. accuracy') 
    plt.xlabel('Step')
    plt.ylabel('Value')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=4)
    plt.title('EXPERIMENT: '+experim[experim.find('en-fi'):experim.find('/train_config.yml')])
    #ax.legend(handles, labels, loc=3)
    
    # VALIDATION VALUES
    print('Experiment:', experim)
    # print CSV header
    print("STEP, val_ppl, val_acc, train_ppl, train_acc")

    # for each step, print line with perplexities for all language pairs
    for step in sorted(ppls.keys()):
        print(str(step)+', '+str(ppls[step])+", "+str(accs[step])+', '+str(tppls[step])+', '+str(taccs[step]))
    

    plt.show()

if __name__ == "__main__":
    main()

