
import pandas as pd
import numpy as np ## used for methamtical compution and handle multi-dimensional arrays and matrices
#
import joblib



def add_noise(data):
    noise_value = 0.015 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size=data.shape[0])

    return data

from scipy.stats import skew
def extract_process(data):
    output_result = np.array([])
    ft1 = librosa.feature.mfcc(y=data, sr = SAMPLE_RATE, n_mfcc=40)
    ft2 = librosa.feature.zero_crossing_rate(y=data)[0]
    ft3 = librosa.feature.spectral_rolloff(y=data)[0]
    ft4 = librosa.feature.spectral_centroid(y=data)[0]
    ft5 = librosa.feature.spectral_contrast(y=data)[0]
    ft6 = librosa.feature.spectral_bandwidth(y=data)[0]
    stft_out = np.abs(librosa.stft(data))
    ft7 = librosa.feature.chroma_stft(S=stft_out, sr=SAMPLE_RATE)[0]
    ft8 = librosa.feature.rms(y=data)[0]
    ft9 = librosa.feature.melspectrogram(y=data)[0]
    ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1)))
    ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
    ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
    ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
    ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
    ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
    ft7_trunc = np.hstack((np.mean(ft7), np.std(ft7), skew(ft7), np.max(ft7), np.median(ft7), np.max(ft7)))
    ft8_trunc = np.hstack((np.mean(ft8), np.std(ft8), skew(ft8), np.max(ft8), np.median(ft8), np.max(ft8)))
    ft9_trunc = np.hstack((np.mean(ft9), np.std(ft9), skew(ft9), np.max(ft9), np.median(ft9), np.max(ft9)))
    output_result = np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc, ft7_trunc, ft8_trunc, ft9_trunc))
    return output_result


def export_process(path):
    data, sample_rate = librosa.load(path)

    output_1 = extract_process(data)
    result = np.array(output_1)

    noise_out = add_noise(data)
    output_2 = extract_process(noise_out)
    result = np.vstack((result, output_2))

    return result

classes = ['Human Assaults', 'Baby Cry', 'Car Alarm', 'Car Crash', 'Car Horn', 'Explosion', 'Fire Burning', 'Foot Steps', 'Glass Break', 'Gun shot', 'Machine Gun', 'Police Siren', 'Scream']
SAMPLE_RATE=44100


def stacking_classifier_model(filename):
    # preprocessing before model implementation

    new_predict_list = []
    feat_new = export_process(filename)

    for feat in feat_new:
        new_predict_list.append(feat)

    New_Predict_Feat = pd.DataFrame(new_predict_list)
    print('done feature extraction')

    # Model implementation CNN model.
    model = joblib.load(r'C:\Users\ahmed\OneDrive\Documents\CNN\stacking_classifier_model.joblib')
    result = model.predict_proba(New_Predict_Feat)
    # ... (previous code)

    dict_result = {}
    for i in range(len(classes)):
        dict_result[i] = classes[i]

    res = result[0]

    if isinstance(res, np.ndarray):  # Check if res is a numpy array
        res = res.tolist()  # Convert numpy array to list

    prob_class_dict = {class_name: round(prob * 100, 2) for prob, class_name in zip(res, classes)}

    # Sort based on probabilities (in descending order)
    sorted_prob_class = sorted(prob_class_dict.items(), key=lambda x: x[1], reverse=True)

    # Create a dictionary with class names as keys and probabilities as values
    result_dict = {class_name: prob for class_name, prob in sorted_prob_class}

    return result_dict


import librosa

stacking_classifier_model('D:/engf_converted.wav')