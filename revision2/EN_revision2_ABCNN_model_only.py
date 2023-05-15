from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def model_setup_ABCNN(xs=None, ys=None, lr=1e-4, lnn=7):
    def attention(inputs):
        x = inputs
        e = K.tanh(K.dot(x, W))
        a = K.softmax(e, axis=1)
        output = x * a
        return output
    
    W = K.variable(K.random_normal((512, 1)), name="att_weight")
    
    inputs = Input(shape=xs)
    x = Conv2D(2**lnn, (4, 4), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(2**lnn, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = Conv2D(2**lnn, (2, 2), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    
    lnn2 = lnn-1
    lnn3 = lnn-2
    
    x = Flatten()(x)
    x = Lambda(attention)(x)
    x = Dense(2**lnn2, activation='relu' )(x)
    x = Dense(2**lnn2, activation='relu' )(x)
    x = Dense(2**lnn2, activation='relu' )(x)
    x = Dense(2**lnn3, activation='relu' )(x)
    x = Dense(2**lnn3, activation='relu' )(x)
    x = Dense(2**lnn3, activation='relu' )(x)
    x = Dense(2**lnn3, activation='relu' )(x)
    x = Dense(ys, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer=Adam(learning_rate=lr, decay=1e-3, beta_1=0.9, beta_2=0.999), \
                  loss='categorical_crossentropy') 
        
    return model

model = model_setup_ABCNN(xs=(32,32,3), ys=2, lr=1e-4, lnn=7)
print(model.summary())
