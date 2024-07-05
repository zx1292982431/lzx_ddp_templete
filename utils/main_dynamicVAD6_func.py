import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import soundfile as sf
import time
VERBOSE = False
PLOT = True

IN_PATHS = 'ev_out_01_2001-01-01-12-37-06_ch6_6.wav'
OUT_PATHS = IN_PATHS[:-4] + '_' + str(int(time.time())) + '.wav'
PLOT_PATH = 'output/fig.png'

SAMPLE_RATE = 16000
AGC_FRAME_MS = 20 #ms
AGC_SHIFT_MS = 10
# AGC_SUB_FRAME_MS = 1 #ms
CHUNK_SIZE = int(16000 * AGC_FRAME_MS / 1000)
WShift=int(16000 * AGC_SHIFT_MS / 1000)
VAD_THES = 0.001
VADSmoothFrame = 20
# AGC_WINDOW = AGC_FRAME_MS * 0
AGC_WINDOW = 0
GDTH = 3 # 前后两帧增益的最大差值
Decline = 2 # vad是0时，gain增益降低限制（相比于前一帧，递减2）。
GainAverageFrames = 5 #计算增益时使用的平滑帧数（这里用平均）

Target_level = -2
HEADROOM = Target_level - 5

def voice_activity_detection(peaks, smoothFrame=20):
    return np.mean(peaks[-smoothFrame:]) > VAD_THES #使用前十帧的均值作为判断当前帧是否是语音帧的输入

def wav_segmentation(in_sig, framesamp=320, hopsamp=160, windows = 1):
    sigLength = in_sig.shape[0]
    M = (sigLength - framesamp) // hopsamp + 1
    a = np.zeros((M,framesamp))
    startpoint = 0
    for m in range(M):
        a[m, :] = in_sig[startpoint:startpoint+framesamp] * windows
        startpoint = startpoint + hopsamp
    return a
def auto_gain_control(wav):
    chunks = wav_segmentation(wav, CHUNK_SIZE, WShift)
    Frames_num = chunks.shape[0]
    peaks = []
    vads = []
    frm = 0
    for i in range(Frames_num):
        chunk_in = chunks[i, :]
        frm = frm + 1
        peak = np.max(np.multiply(chunk_in, chunk_in))
        peaks.append(peak)
        vad = voice_activity_detection(peaks, smoothFrame=VADSmoothFrame)
        vads.append(vad)
    return np.array(vads)

#
# print('\nauto_gain_control:', IN_PATHS)
#
# # Input
#
# wav_in, rates = sf.read('/home/imu_heshulin/ddn/downloads/PDOA_LibriSpeech/SPEECH/tt/target/4564-2961-961-0021-6829-68771-0025.wav')
# wav_vads  = auto_gain_control(wav_in)
#
# # 从 -10 到 10 平均分成100份
#
#
# plt.plot(wav_vads)
# plt.show()
#
#
# ##### Graph ####
# if not PLOT:
#     exit()
# print('\nplotting:', PLOT_PATH)
# PLOT_WAV_SAMPLE_NUM = 10000
# PLOT_WAV_MAX = 1.0
# PLOT_DB_SAMPLE_NUM = 100
# PLOT_DB_MIN = -20
# PLOT_DB_MAX = 0
# PLOT_GAIN_MAX = np.max(50)
# PLOT_GAIN_MIN = np.min(-20)
# PLOT_PD_MAX = 2
# PLOT_PD_MIN = 0
# PLOT_VAD_MAX = 2
# PLOT_VAD_MIN = 0
# plots = [
#     wav_sample(wav_in, PLOT_WAV_SAMPLE_NUM),
#     wav_vads,
#     ]
# FIG = 2
# COL = 1
# ROW = 1 * FIG
# fig = plt.figure(figsize=(20 * COL, 3 * ROW))
# for i in range(1):
#     j = 0
#     plot = plots[j + FIG * i]
#     ax = fig.add_subplot(ROW, COL, 1 + j + FIG * i)
#     ax.set_title(IN_PATHS[i])
#     ax.set_autoscale_on(False)
#     ax.axis([0, len(plot), -PLOT_WAV_MAX, PLOT_WAV_MAX])
#     ax.plot(plot)
#
#
#
#
#     j += 1
#     plot = plots[j + FIG * i]
#     ax = fig.add_subplot(ROW, COL, 1 + j + FIG * i)
#     ax.set_title('VAD')
#     ax.set_autoscale_on(False)
#     ax.axis([0, len(plot), PLOT_VAD_MIN, PLOT_VAD_MAX])
#     ax.plot(plot)
#
#
# plt.show()