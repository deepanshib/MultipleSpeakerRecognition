
import operator
from scipy.io import wavfile
import numpy
from sklearn.mixture import GaussianMixture
EPS = numpy.finfo(float).eps


def read_wav(fname):

    fs, signal = wavfile.read(fname)
    assert len(signal.shape) == 1, "Only Support Mono Wav File!"
    return fs, signal


def write_wav(fname, fs, signal):
    assert len(signal.shape) == 1, "Only Support Mono Wav File!"
    wavfile.write(fname, fs, signal)


def normalize_features(features):

    all_features = numpy.array([])
    count = 0
    for feature in features:
        if feature.shape[0] > 0:
            if count == 0:
                all_features = feature
            else:
                all_features = numpy.vstack((all_features, feature))
            count += 1

    mean_features = numpy.mean(all_features, axis=0) + EPS
    std_features = numpy.std(all_features, axis=0) + EPS

    features_norm = []
    for feature in features:
        feat_tmp = feature.copy()
        for n_samples in range(feature.shape[0]):
            feat_tmp[n_samples, :] = (feat_tmp[n_samples, :] - mean_features) / std_features
        features_norm.append(feat_tmp)
    return (features_norm, mean_features, std_features)


class GMMSet:

    def __init__(self, gmm_order=5):
        self.gmms = []
        self.gmm_order = gmm_order
        self.label_tmp = []

    def fit_new(self, signal_tmp, label):
        
        self.label_tmp.append(label)
        gmm = GaussianMixture(self.gmm_order)
        gmm.fit(signal_tmp)
        self.gmms.append(gmm)

    @staticmethod
    def gmm_score(gmm, signal_tmp):
   
        return numpy.sum(gmm.score(signal_tmp))

    def predict_one(self, signal_tmp):
     
        scores = [self.gmm_score(gmm, signal_tmp) / len(signal_tmp) for gmm in self.gmms]
        predict_value = sorted(enumerate(scores), key=operator.itemgetter(1), reverse=True)
        predict_value = [(str(self.label_tmp[i]), label_tmp, predict_value[0][1] - label_tmp)
                         for i, label_tmp in predict_value]
        result = [(self.label_tmp[index], value) for (index, value) in enumerate(scores)]
        predict_value = max(result, key=operator.itemgetter(1))
        return predict_value[0]
