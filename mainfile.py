
import os
import numpy
from scipy.spatial import distance
import scipy.signal
from utils import normalize_features, vad, mt_feature_extraction, ivad, write_wav
from model.unsupervised_model import kmeans_silhouette
EPS = numpy.finfo(float).eps
import matplotlib.pyplot as plt
import wave


from scipy.io import wavfile

audioname="1brian.wav"
fs, signal = wavfile.read(audioname)
#fs=8000
mt_size=2.0
mt_step=0.2
st_win=0.05

st_step = st_win

#FEATURE EXTRACTION AND NORMALISATION
[mid_term_features, short_term_features] = mt_feature_extraction(signal, fs, mt_size * fs,mt_step * fs,round(fs * st_win))
[mid_term_features_norm, _, _] = normalize_features([mid_term_features.T])
mid_term_features_norm = mid_term_features_norm[0].T
num_of_windows = mid_term_features.shape[1]

# VAD
reserved_time = 1
segment_limits = vad(short_term_features, st_step, smooth_window=0.5, weight=0.3)
i_vad = ivad(segment_limits, mt_step, reserved_time, num_of_windows)
mid_term_features_norm = mid_term_features_norm[:, i_vad]

# remove outliers:
distances_all = numpy.sum(distance.squareform(distance.pdist(mid_term_features_norm.T)), axis=0)
m_distances_all = numpy.mean(distances_all)
i_non_outliers = numpy.nonzero(distances_all < 1.2 * m_distances_all)[0]

mid_term_features_norm = mid_term_features_norm[:, i_non_outliers]
i_features_select = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 41, 42, 43, 44, 45,
                     46, 47, 48, 49, 50, 51, 52, 53]
mid_term_features_norm = mid_term_features_norm[i_features_select, :]

num_range = range(2, 10)
[n_speakers_final, imax, num_speaker_cls] =kmeans_silhouette(mid_term_features_norm, num_range) # to find no of speakers

cls = numpy.zeros((num_of_windows,))-1
valid_pos = i_vad[i_non_outliers]
for i in range(num_of_windows):
    if i in valid_pos:
        j = numpy.argwhere(valid_pos == i)[0][0]
        cls[i] = num_speaker_cls[imax][j]

# median filtering:
cls = scipy.signal.medfilt(cls, 11)
start = 0
end = 0

#fig = plt.figure(figsize=(15,4))
#imageCoordinate = 100 + 10*n_speakers_final + 1
#i = 0
#times = numpy.arange(len(cls))/float(fs)

for i in range(1, len(cls)):
    if cls[i] == cls[i-1]:
        end = i
    else:
        newpath="result_wav/"+audioname
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        audiofile= str(cls[i-1]) + ":" + str(start*mt_step) + "-" +str(end*mt_step) + ".wav"
        write_wav(os.path.join(newpath,audiofile),fs, signal[int(start * mt_step * fs):int(end * mt_step * fs)])
    
      
#        
        
        #next steps
        
        #check whether speaker is known
        #determine gmm for audiofile
        #compare it with previous avaliable gmm models
        #produce result
        #c[i] = speaker number and name matched with gmm model
print ( n_speakers_final, cls)
