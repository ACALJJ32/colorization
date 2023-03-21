import os
import sys
import time
import torch
import tqdm
import importlib
import numpy as np
sys.path.append(os.path.dirname(sys.path[0]))


def Load_model():
    net = importlib.import_module('VP_code.models.' + "BRT_tlc")
    netG = net.Video_Backbone()
    netG.cuda()
    print("Finish loading model ...")
    return netG


model = Load_model()

device = 'cuda:0'

input_frames = 8
dummy_input = torch.rand(1, input_frames, 3, 960, 540).to(device)
repetitions = 25

with torch.no_grad():
    print("warming ...")
    for _ in range(15):
        _ = model(dummy_input)
    print("warming finished.")


torch.cuda.synchronize()
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
timings = np.zeros((repetitions, 1))

print('testing ...\n')
start_time = time.time()
with torch.no_grad():
    for rep in tqdm.tqdm(range(repetitions)):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize() 
        curr_time = starter.elapsed_time(ender) 
        timings[rep] = curr_time
end_time = time.time()

avg = (end_time - start_time) / (repetitions * input_frames)

print('avg = {}s per frame \n'.format(avg))