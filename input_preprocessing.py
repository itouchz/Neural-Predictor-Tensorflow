import numpy as np
import tensorflow as tf

from spektral.utils.convolution import gcn_filter

def preprocess_nasbench(models, normalize=True):
    X, A, AT, labels, val_acc, test_acc, times = [], [], [], [], [], [], []

    node_ops = {
        'input': [1, 0, 0, 0, 0],
        'conv1x1-bn-relu': [0, 1, 0, 0, 0],
        'conv3x3-bn-relu': [0, 0, 1, 0, 0],
        'maxpool3x3': [0, 0, 0, 1, 0],
        'output': [0, 0, 0, 0, 1],
        'mask': [0, 0, 0, 0, 0]
    }

    for metric in models:
        # Operations: [input, conv1x1, conv3x3, max-pool, output]
        V = [] # V \in R^I x D (# nodes x 5)
        adj = metric['module_adjacency']

        val_acc.append(metric['validation_accuracy'] * 100)
        test_acc.append(metric['test_accuracy'] * 100)
        times.append(metric['training_time'])

        label = 1 if metric['validation_accuracy'] > 0.91 else 0
        labels.append(label)
        for i in range(7):
            if i < len(metric['module_operations']):
                V.append(node_ops[metric['module_operations'][i]])
            else:
                V.append(node_ops['mask'])


        A.append(np.pad(adj, pad_width=(0, 7 - adj.shape[0]), mode='constant', constant_values=0))
        AT.append(np.pad(np.transpose(adj), pad_width=(0, 7 - np.transpose(adj).shape[0]),  mode='constant', constant_values=0))        

        X.append(V)

    norm_A = np.array(gcn_filter(A)) if normalize else np.array(A)
    norm_AT = np.array(gcn_filter(AT)) if normalize else np.array(AT)
    X = np.array(X)
    labels = np.array(labels)
    
    return {'X': X, 'norm_A': norm_A, 'norm_AT': norm_AT, 'labels': labels, 'val_acc': np.array(val_acc), 'test_acc': np.array(test_acc), 'times': np.array(times)}

def preprocess_proxylessnas(models, normalize=True):    
    X = tf.keras.utils.to_categorical(models, num_classes=7, dtype=int)
    A = np.zeros([len(models), 22, 22])
    AT = []
    for m in range(len(models)):
        for i in range(22):
            if i < 21:
                A[m][i][i+1] = 1
        AT.append(np.transpose(A[m]))
    AT = np.array(AT)
    return {'X': X, 'norm_A': gcn_filter(A) if normalize else np.array(A), 'norm_AT': gcn_filter(AT) if normalize else np.array(AT)}