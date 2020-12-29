import pandas as pd
import numpy as np


def prepare_data(data):
    """ Prepare data for modeling 
        input: data frame with labels und pixel data
        output: image and label array """
    
    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data['emotion'])))
    
    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, ' pixels'], dtype=int, sep=' ')
        image = np.reshape(image, (48, 48))
        image_array[i] = image
        
    return image_array, image_label

def loadFer2013(path):
    data = pd.read_csv(path)
    #merge disgust to fear
    data['emotion'][data['emotion']>1]-=1
    
    X_train, y_train = prepare_data(data[data[' Usage']=='Training'])
    X_val, y_val = prepare_data(data[data[' Usage']=='PrivateTest'])
    X_test, y_test = prepare_data(data[data[' Usage']=='PublicTest'])

    return X_train,X_val,X_test,y_train,y_val,y_test