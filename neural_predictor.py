import tensorflow as tf

from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Average, Dropout, Conv1D, MaxPool1D, GlobalAveragePooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import CosineDecay

from spektral.layers import GCNConv, GlobalAvgPool

N_to_D = {
    43: 48 ,
    172: 144,
    86: 72,
    334: 210, 
    129: 96,
    860: 320
}

def classifier(train_data, labels, N=172, D=0, n_gcn=3, n_hidden_fc=[128], init_lr=0.0002, dropout_rate=0.1, weight_decay=0.001, n_epochs=300, batch_size=10):
    tf.keras.backend.clear_session()
    """
    **Input**
    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be computed with
    `spektral.utils.convolution.gcn_filter`.
    """  

    D = D if D > 0 else N_to_D[N] # constant for all layers
    n_nodes, n_dim = train_data[0].shape[1:]

    V0 = Input(shape=[n_nodes, n_dim])
    A = Input(shape=[n_nodes, n_nodes])
    AT = Input(shape=[n_nodes, n_nodes])

    V = V0
    for l in range(n_gcn):
        V1 = GCNConv(D, activation='relu', kernel_regularizer=l2(weight_decay))([V, A])
        V2 = GCNConv(D, activation='relu', kernel_regularizer=l2(weight_decay))([V, AT])
        V = Average()([V1, V2])

    fc_in = GlobalAvgPool()(V)
    for units in n_hidden_fc:        
        fc_in = Dense(units, kernel_regularizer=l2(weight_decay))(fc_in)  
        fc_in = Dropout(dropout_rate)(fc_in)

    y = Dense(1, activation='sigmoid', kernel_regularizer=l2(weight_decay))(fc_in)
    
    model = Model(inputs=[V0, A, AT], outputs=y)

    lr = CosineDecay(init_lr, 10000)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_data, labels, epochs=n_epochs, batch_size=batch_size, verbose=0)
    
    return model


def ss_sigmoid(x):
    x = tf.sigmoid(x) * 100 + 10
    x = tf.where(x >= 100., 100., x)
    return x

def s_sigmoid(x):
    x = tf.sigmoid(x) * 100
    x = tf.where(x >= 100., 100., x)
    return x

def regressor(train_data, labels, N=172, D=0, n_gcn=3, n_hidden_fc=[128], init_lr=0.0001, dropout_rate=0.1, weight_decay=0.001, n_epochs=300, batch_size=10, mode='accuracy', is_shift=True):
    tf.keras.backend.clear_session()
    """
    **Input**
    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be computed with
    `spektral.utils.convolution.gcn_filter`.
    """  

    D = D if D > 0 else N_to_D[N] # constant for all layers
    n_nodes, n_dim = train_data[0].shape[1:]

    V0 = Input(shape=[n_nodes, n_dim])
    A = Input(shape=[n_nodes, n_nodes])
    AT = Input(shape=[n_nodes, n_nodes])

    V = V0
    for l in range(n_gcn):
        V1 = GCNConv(D, activation='relu', kernel_regularizer=l2(weight_decay))([V, A])
        V2 = GCNConv(D, activation='relu', kernel_regularizer=l2(weight_decay))([V, AT])
        V = Average()([V1, V2])

    fc_in = GlobalAvgPool()(V)
    for units in n_hidden_fc:
        fc_in = Dense(units, kernel_regularizer=l2(weight_decay))(fc_in)  
        fc_in = Dropout(dropout_rate)(fc_in)

    if mode == 'accuracy':
        y = Dense(1, activation=ss_sigmoid if is_shift else s_sigmoid, kernel_regularizer=l2(weight_decay))(fc_in)
    elif mode == 'error':
        y = Dense(1, activation='relu', kernel_regularizer=l2(weight_decay))(fc_in)
    
    model = Model(inputs=[V0, A, AT], outputs=y)

    lr = CosineDecay(init_lr, 10000)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    model.fit(x=train_data, y=labels, epochs=n_epochs, batch_size=batch_size, verbose=0)
    
    return model


def regressor_mlp(train_data, labels, N=172, n_gcn=3, n_hidden_fc=[128], init_lr=0.0001, dropout_rate=0.1, weight_decay=0.001, n_epochs=300, batch_size=10):
    tf.keras.backend.clear_session()
    """
    **Input**
    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be computed with
    `spektral.utils.convolution.gcn_filter`.
    """  

    D = N_to_D[N] # constant for all layers
    n_nodes, n_dim = train_data.shape[1:]

    V0 = Input(shape=[n_nodes, n_dim])

    V = Flatten()(V0)
    for l in range(n_gcn):
        V = Dense(D, activation='relu', kernel_regularizer=l2(weight_decay))(V)

    fc_in = V
    for units in n_hidden_fc:
        fc_in = Dense(units, kernel_regularizer=l2(weight_decay))(fc_in)  
        fc_in = Dropout(dropout_rate)(fc_in)

    y = Dense(1, activation=ss_sigmoid, kernel_regularizer=l2(weight_decay))(fc_in)
    
    model = Model(inputs=V0, outputs=y)

    lr = CosineDecay(init_lr, 10000)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    model.fit(train_data, labels, epochs=n_epochs, batch_size=batch_size, verbose=0)
    
    return model


def regressor_cnn(train_data, labels, N=172, n_gcn=3, n_hidden_fc=[128], init_lr=0.0001, dropout_rate=0.1, weight_decay=0.001, n_epochs=300, batch_size=10):
    tf.keras.backend.clear_session()
    """
    **Input**
    - Node features of shape `([batch], n_nodes, n_node_features)`;
    - Modified Laplacian of shape `([batch], n_nodes, n_nodes)`; can be computed with
    `spektral.utils.convolution.gcn_filter`.
    """  

    D = N_to_D[N] # constant for all layers
    n_nodes, n_dim = train_data.shape[1:]

    V0 = Input(shape=[n_nodes, n_dim])

    V = V0
    for l in range(n_gcn):
        V = Conv1D(D, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(weight_decay))(V)
        V = MaxPool1D(2, padding='same')(V)

    fc_in = GlobalAveragePooling1D()(V)
    for units in n_hidden_fc:
        fc_in = Dense(units, kernel_regularizer=l2(weight_decay))(fc_in)  
        fc_in = Dropout(dropout_rate)(fc_in)

    y = Dense(1, activation=ss_sigmoid, kernel_regularizer=l2(weight_decay))(fc_in)
    
    model = Model(inputs=V0, outputs=y)

    lr = CosineDecay(init_lr, 10000)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    model.fit(train_data, labels, epochs=n_epochs, batch_size=batch_size, verbose=0)
    
    return model