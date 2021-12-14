import os
from mir_eval.separation import bss_eval_sources
import soundfile as sf

filename = os.listdir("input_music")[0]
input_wav = "input_music/"+filename
output_wav = "./output_music/instrument_"+filename
x, _ = sf.read(input_wav)
y, _ = sf.read(output_wav)
y = y[:,[0,0]]
print(bss_eval_sources(x, y))