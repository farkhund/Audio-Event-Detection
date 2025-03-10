import uuid
import urllib
from flask import Flask , render_template  , request , send_file
from sklearn.preprocessing import StandardScaler
import librosa
import numpy as np
import pandas as pd
import joblib
from pydub import AudioSegment
from scipy.stats import skew
import os



app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# model = load_model("D:/Download here/model.hdf5")
# model = load_model("C:/Users/ahmed/Downloads/model.hdf5")


SAMPLE_RATE = 44100

classes = ['Human Assault',  'Baby Cry', 'Car Crash', 'Explosion', 'Foot Steps', 'Glass Break', 'Gunshot', 'Machine gun', 'Police siren', 'Scream']


ALLOWED_EXT = set(['wav','jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT

# classes = ['airplane' ,'automobile', 'bird' , 'cat' , 'deer' ,'dog' ,'frog', 'horse' ,'ship' ,'truck']

# from tensorflow.keras.models import load_model



# Generate mfcc features with mean and standard deviation
def extract_process(data):
    output_result = np.array([])
    ft1 = librosa.feature.mfcc(data, sr = SAMPLE_RATE, n_mfcc=40)
    ft2 = librosa.feature.zero_crossing_rate(data)[0]
    ft3 = librosa.feature.spectral_rolloff(data)[0]
    ft4 = librosa.feature.spectral_centroid(data)[0]
    ft5 = librosa.feature.spectral_contrast(data)[0]
    ft6 = librosa.feature.spectral_bandwidth(data)[0]
    ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis = 1), np.max(ft1, axis = 1), np.median(ft1, axis = 1), np.min(ft1, axis = 1)))
    ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
    ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
    ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
    ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
    ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
    output_result = np.hstack((ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc))
    return output_result


def pitch_process(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def stretch_process(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)


def add_noise(data):
    noise_value = 0.015 * np.random.uniform() * np.amax(data)
    data = data + noise_value * np.random.normal(size=data.shape[0])

    return data

# def export_process(path):
#     data, sample_rate = librosa.load(path)
#
#     output_1 = extract_process(data)
#     result1 = np.array(output_1)
#
#     noise_out = add_noise(data)
#     output_2 = extract_process(noise_out)
#     result2 = np.vstack((result1, output_2))
#
#     new_out = stretch_process(data)
#     strectch_pitch = pitch_process(new_out, sample_rate)
#     output_3 = extract_process(strectch_pitch)
#     result3 = np.vstack((result2, output_3))
#     return result3

def export_process(path):
    data, sample_rate = librosa.load(path)

    output_1 = extract_process(data)
    result = np.array(output_1)

    noise_out = add_noise(data)
    output_2 = extract_process(noise_out)
    result = np.vstack((result, output_2))

    new_out = stretch_process(data)
    strectch_pitch = pitch_process(new_out, sample_rate)
    output_3 = extract_process(strectch_pitch)
    result = np.vstack((result, output_3))

    return result
    # data,sample_rate = librosa.load(path)

    # output_1 = extract_process(data)
    # result = np.array(output_1)

    # noise_out = add_noise(data)
    # output_2 = extract_process(noise_out)
    # result = np.vstack((result,output_2))

    # new_out = stretch_process(data)
    # strectch_pitch = pitch_process(new_out,sample_rate)
    # output_3 = extract_process(strectch_pitch)
    # result = np.vstack((result,output_3))

scaler_data = StandardScaler()



def get_features(filepath):


    file = os.path.basename(filepath)
    filename, file_extension = os.path.splitext(file)

    def audio2wav(path):
        audio_file = path.replace(file_extension, '.wav')
        print("audio file path of extension:", file_extension)
        aud = AudioSegment.from_file(path)
        aud.export(audio_file, format='wav')
        return audio_file

    if file_extension != '.wav':
        inputFile = audio2wav(filepath)
    elif file_extension == '.wav':
        inputFile = filepath
    else:
        inputFile = None




    data, _ = librosa.load(inputFile, sr=SAMPLE_RATE)
    assert _ == SAMPLE_RATE
    try:
        ft1 = librosa.feature.mfcc(data, sr=SAMPLE_RATE, n_mfcc=40)
        ft2 = librosa.feature.zero_crossing_rate(data)[0]
        ft3 = librosa.feature.spectral_rolloff(data)[0]
        ft4 = librosa.feature.spectral_centroid(data)[0]
        ft5 = librosa.feature.spectral_contrast(data)[0]
        ft6 = librosa.feature.spectral_bandwidth(data)[0]
        stft_out = np.abs(librosa.stft(data))
        ft7 = librosa.feature.chroma_stft(S=stft_out, sr=SAMPLE_RATE)[0]
        ft8 = librosa.feature.rms(data)[0]
        ft9 = librosa.feature.melspectrogram(data)[0]
        ft1_trunc = np.hstack((np.mean(ft1, axis=1), np.std(ft1, axis=1), skew(ft1, axis=1), np.max(ft1, axis=1),
                               np.median(ft1, axis=1), np.min(ft1, axis=1)))
        ft2_trunc = np.hstack((np.mean(ft2), np.std(ft2), skew(ft2), np.max(ft2), np.median(ft2), np.min(ft2)))
        ft3_trunc = np.hstack((np.mean(ft3), np.std(ft3), skew(ft3), np.max(ft3), np.median(ft3), np.min(ft3)))
        ft4_trunc = np.hstack((np.mean(ft4), np.std(ft4), skew(ft4), np.max(ft4), np.median(ft4), np.min(ft4)))
        ft5_trunc = np.hstack((np.mean(ft5), np.std(ft5), skew(ft5), np.max(ft5), np.median(ft5), np.min(ft5)))
        ft6_trunc = np.hstack((np.mean(ft6), np.std(ft6), skew(ft6), np.max(ft6), np.median(ft6), np.max(ft6)))
        ft7_trunc = np.hstack((np.mean(ft7), np.std(ft7), skew(ft7), np.max(ft7), np.median(ft7), np.max(ft7)))
        ft8_trunc = np.hstack((np.mean(ft8), np.std(ft8), skew(ft8), np.max(ft8), np.median(ft8), np.max(ft8)))
        ft9_trunc = np.hstack((np.mean(ft9), np.std(ft9), skew(ft9), np.max(ft9), np.median(ft9), np.max(ft9)))
        return (np.hstack(
            (ft1_trunc, ft2_trunc, ft3_trunc, ft4_trunc, ft5_trunc, ft6_trunc, ft7_trunc, ft8_trunc, ft9_trunc)))
    except:
        print('bad file')
        return pd.Series([0] * 210)





import numpy as np


def predict(filepath):
    print('hi')

    # predicting emotions from pre-trained model
    new_predict_list = []

    feat_new = get_features(filepath)

    feat_new = np.array(feat_new)  # Convert Pandas Series to NumPy array
    feat_new = feat_new.reshape(1, -1)
    print('done feature extraction')

    for feat in feat_new:
        new_predict_list.append(feat)

    New_Predict_Feat = pd.DataFrame(new_predict_list)
    print('done feature extraction')

    filename = r'XGB_ninefeatures.joblib'
    loaded_model = joblib.load(filename)
    result = loaded_model.predict_proba(New_Predict_Feat)

    dict_result = {}

    for i in range(len(classes)):
        print(i)
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:10]

    prob_result = []
    class_result = []

    for i in range(10):
        prob_result.append((prob[i] * 100).round(2))
        class_result.append(dict_result[prob[i]])

    print(prob_result, class_result)
    return class_result, prob_result



@app.route('/')
def home():
    return render_template("index.html")

@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , 'static/images')
    if request.method == 'POST':
        if(request.form):
            link = request.form.get('link')
            try :
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename+".wav"
                img_path = os.path.join(target_img , filename)
                output = open(img_path , "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result , prob_result = predict(img_path)
                print(prob_result)
                print(class_result)
                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "class4": class_result[3],
                    "class5": class_result[4],
                    "class6": class_result[5],
                    "class7": class_result[6],
                    "class8": class_result[7],
                    "class9": class_result[8],
                    "class10": class_result[9],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                    "prob4": prob_result[3],
                    "prob5": prob_result[4],
                    "prob6": prob_result[5],
                    "prob7": prob_result[6],
                    "prob8": prob_result[7],
                    "prob9": prob_result[8],
                    "prob10": prob_result[9],
                }

            except Exception as e :
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)


        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename
                print(img_path)
                class_result, prob_result = predict(img_path)

                predictions = {
                    "class1":class_result[0],
                    "class2":class_result[1],
                    "class3":class_result[2],
                    "class4": class_result[3],
                    "class5": class_result[4],
                    "class6": class_result[5],
                    "class7": class_result[6],
                    "class8": class_result[7],
                    "class9": class_result[8],
                    "class10": class_result[9],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                    "prob4": prob_result[3],
                    "prob5": prob_result[4],
                    "prob6": prob_result[5],
                    "prob7": prob_result[6],
                    "prob8": prob_result[7],
                    "prob9": prob_result[8],
                    "prob10": prob_result[9],
                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)

    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug = False)


