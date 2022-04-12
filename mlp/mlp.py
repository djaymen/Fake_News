# imports
import os
from unicodedata import name
import tensorflow.nn as nn
from tensorflow.keras import Input,Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers, regularizers,activations
from tensorflow.keras.optimizers import SGD,Adam
# end imports



'''
    Model Configurations :
'''

config =    {   'arch' : {
                            '2L_64x1'    : [64,64],
                            '3L_32x2'    : [32,64,128],
                            '4L_128x0.5' : [128,64,32,16] 
                        },           
                'n_nodes'     : 64,
                'factor'        : 2,
                'n_layers'  : 2,
                'l2_reg'        : 0.0001,
                'dropout' : 0.2,
                'lr'      : 0.0001
            }


'''
    -------------------------------------------------------------------------------------------     
        FNC-MLP (Fake News Challenge Model) :
    -------------------------------------------------------------------------------------------
'''
class FNC(object):
    
    def __init__(self,X_train,y_train,arch='3L_32x2',n_nodes=config['n_nodes'],factor=config['factor'],
                n_layers=config['n_layers'],l2_reg=config['l2_reg'],dropout=config['dropout']) -> None:
        super().__init__()
        self.X_train , self.y_train  = X_train , y_train
        self.arch = arch
        self.n_nodes = n_nodes
        self.factor = factor
        self.n_layers = n_layers
        self.l2_reg = regularizers.l2(l2_reg)
        self.dropout = dropout
        self.model = None
    
    def __str__(self) -> str:
        return f'FNC{self.arch}'

    def summary(self):
        self.model.summary()
    
    def plot_model(self,save_path='models/model_imgs'):
        
        if not self.model:
            print(f' You should build the model first !')
            return
  
        to_file = os.path.join(os.getcwd(),save_path,f'{str(self)}.png')
        plot_model(self.model,to_file=to_file,show_shapes=True)
        
    def build_model(self,n_blocks=config.n_res_blocks):
        # Input Block
        inp = Input(shape=(self.X_train.shape[1], 1), name='Input')

        # Body Block 
        x = self.body_block(x,n_blocks)

        # Output block
        out = self.output_block(x)

        # Build model 
        self.model = Model(inputs=inp, outputs=[out])

        self.model.compile(
                optimizer = Adam(learning_rate=config['lr']),
                loss={'Output': 'categorical_crossentropy'},
                metrics={'Output': 'categorical_accuracy'})

        return self.model
    

    def body_block(self,x):
        n = self.n_nodes
        f = 1
        for i in range(self.n_layers):
            x = layers.Dense(n*f,kernel_regularizer=self.l2_re,name=f"Layer #{i+1}")(x)
            x = self.activation(x)
        return x
        
    def output_block(self,x):
        x = layers.Flatten()(x)
        x = layers.Dense(self.y_train.shape[1],activation ='softmax',name='Output')
        return x

    def activation(self,x):
        return nn.relu(x)