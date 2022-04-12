import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import tensorflow as tf 

from tensorflow.keras.utils import to_categorical
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from mlp import FNC



### Global stuff 🌍
# data path : 
root = os.path.dirname(__file__)
data_path = os.path.join(root,"..","data/")
devices = ['/device:GPU:0','/device:GPU:1','/device:GPU:2','/device:GPU:3',
          '/device:GPU:4','/device:GPU:5','/device:GPU:6','/device:GPU:7']
# End global


def get_data(p=0.2):
    
    df = pd.read_csv(os.path.join(data_path,'df.csv'))
    
    X , y = df['content'].values , df['label'].values
    print('Fitting TFIDF in progress...')
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(X)
    y = to_categorical(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf,y, test_size=p)
    return X_train, X_test, y_train, y_test
    

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    
    strategy = tf.distribute.MirroredStrategy(devices)
    
    with strategy.scope():
        
        print('Building FNC in progress...')
        fnc = FNC(X_train,y_train)
        model = fnc.build_model()
        fnc.model = model 
        print('FNC summary')
        fnc.summary()
        print('Start training FNC on 8 GPU\'s')
        fnc.train()
        print('Start Evaluation of FNC')
        val = fnc.evaluate(X_test,y_test)
        print(f'Evaluation results : for [{fnc}] : {val}')
        
        
        
        
        
        
    
    
    