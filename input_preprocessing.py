import numpy as np
import tensorflow as tf
import networkx as nx 

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

def preprocess_nasbench_nlp(models, stats, normalize=True):
    ops = {
        'activation_leaky_relu': 0,
        'activation_sigm': 1,
        'activation_tanh': 2,
        'blend': 3,
        'elementwise_prod': 4,
        'elementwise_sum': 5,
        'linear': 6
    }
    
    X = []
    A = []
    AT = []
    val_loss, test_loss, times = [], [], []
    for model in models:
        Xm = np.zeros([25, 7])
        G = nx.DiGraph()
        for n in model.keys():
            if n not in G.nodes():
                G.add_node(n)
            for k in model[n]['input']:
                if k not in G.nodes():
                    G.add_node(k)
                G.add_edge(n, k, label=model[n]['op'])
                G.add_edge(k, n, label='rev_' + model[n]['op'])
                
        for i, k in enumerate(model.keys()):
            Xm[i][ops[model[k]['op']]] = 1.
            
        adj = np.tril(nx.adjacency_matrix(G).todense())
        adj_t = np.triu(nx.adjacency_matrix(G).todense())
        
        X.append(Xm)
        A.append(np.pad(adj, pad_width=(0, 25 - adj.shape[0]), mode='constant', constant_values=0).astype(float))
        AT.append(np.pad(np.transpose(adj), pad_width=(0, 25 - np.transpose(adj).shape[0]),  mode='constant', constant_values=0).astype(float))
    
    for stat in stats:
        val_loss.append(stat['val_loss'])
        test_loss.append(stat['test_loss'])
        times.append(stat['wall_time'])
        
        
    return {'X': np.array(X), 
            'norm_A': gcn_filter(np.array(A)) if normalize else np.array(A), 
            'norm_AT': gcn_filter(np.array(AT)) if normalize else np.array(AT), 
            'val_loss': np.array(val_loss), 
            'test_loss': np.array(test_loss), 
            'times': np.array(times)
           }