import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
import math
from scipy.optimize import linear_sum_assignment as linear_sum_assignment
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
import numpy as np
from keras import optimizers
from tensorflow import optimizers
import ctype

def k_truss(filename):
    """
       Read core structure information.
    """
    try:
        lib = ctypes.cdll.LoadLibrary("./k-truss.dll")
    except:
        lib = ctypes.cdll.LoadLibrary("./libk-truss.so")
    input_str = ctypes.c_char_p(b'./datasets/' + filename.encode() + b'_adj.csv')
    output_str = ctypes.c_char_p(b'./datasets/' + filename.encode() + b'_out.csv')
    sim = lib.k_truss_cycle(input_str, output_str)
    return sim

def sampling(args):
    """
       Sampling operation in variational autoencoder Model.
    """
    z_mean, z_log_var = args
    epsilon_std = 1.0
    batchl = K.shape(z_mean)[0]
    diml = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batchl, diml), mean=0., stddev=epsilon_std)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def variational_auto_encoder(x, dim, latent_dim, lr, epochs, batch_size):
    """
      Set up the variational autoencoder Model.
    """
    dim_len = len(dim)
    if dim_len == 0:
        return x

    rows = x.shape[0]
    dim_input = rows
    input_matrix = Input(shape=(dim_input,))

    activation = 'softsign'
    h = Dense(dim[0], activation=activation)(input_matrix)
    for i in range(dim_len):
        if i > 0:
            h = Dense(dim[i], activation=activation)(h)

    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    if dim_len == 1:
        decoder_h = Dense(dim[0], activation=activation)(z)
        x_decoded_mean = Dense(dim_input, activation='softplus')(decoder_h)

    if dim_len > 1:
        decoder_h = Dense(dim[dim_len - 2], activation=activation)(z)
        for i in range(dim_len):
            if i > 1:
                decoder_h = Dense(dim[dim_len - 1 - i], activation=activation)(decoder_h)
        x_decoded_mean = Dense(dim_input, activation='softplus')(decoder_h)

    vae = Model(input_matrix, x_decoded_mean)
    encoder = Model(input_matrix, z_mean)
    xent_loss = dim_input * metrics.categorical_crossentropy(input_matrix, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)
    vae.add_loss(vae_loss)
    optimizer = optimizers.Adamax(learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0.00)
    vae.compile(optimizer=optimizer,  metrics=['accuracy'])
    vae.summary()
    x_train = x
    try:
        x_train = x_train.values.reshape((len(x_train), np.prod(x_train.shape[1:])))
    except:
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))

    vae.fit(x_train, x_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True)

    res = encoder.predict(x_train)
    return res

def NMI(com, real_com):
    """
    Compute the Normalized Mutual Information(NMI).
    """
    if len(com) != len(real_com):
        return ValueError('len(A) should be equal to len(B)')

    com = np.array(com)
    real_com = np.array(real_com)
    total = len(com)
    com_ids = set(com)
    real_com_ids = set(real_com)
    #Mutual information
    MI = 0
    eps = 1.4e-45
    for id_com in com_ids:
        for id_real in real_com_ids:
            idAOccur = np.where(com == id_com)
            idBOccur = np.where(real_com == id_real)
            idABOccur = np.intersect1d(idAOccur, idBOccur)
            px = 1.0*len(idAOccur[0])/total
            py = 1.0*len(idBOccur[0])/total
            pxy = 1.0*len(idABOccur)/total
            MI = MI + pxy*math.log(pxy/(px*py) + eps,2)
    # Normalized Mutual information
    Hx = 0
    for idA in com_ids:
        idAOccurCount = 1.0*len(np.where(com == idA)[0])
        Hx = Hx - (idAOccurCount/total)*math.log(idAOccurCount/total + eps, 2)
    Hy = 0
    for idB in real_com_ids:
        idBOccurCount = 1.0*len(np.where(real_com == idB)[0])
        Hy = Hy - (idBOccurCount/total) * math.log(idBOccurCount/total + eps, 2)
    MIhat = 2.0*MI/(Hx + Hy)
    return MIhat

def modularity(G, community):
    """
    Compute modularity of communities of network.
    """
    V = [node for node in G.nodes()]
    m = G.size(weight='weight')  # if weighted
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)

    Q = 0
    for i in range(n):
        node_i = V[i]
        com_i = community[node_i]
        degree_i = G.degree(node_i)
        for j in range(n):
            node_j = V[j]
            com_j = community[node_j]
            if com_i != com_j:
                continue
            degree_j = G.degree(node_j)
            Q += A[i][j] - degree_i * degree_j/(2 * m)
    return Q/(2 * m)

def f_same(cluA, cluB, clusters):
    """
     Calculate Cross Common Fraction(f_same).
    """
    S = np.matrix([[0 for i in range(clusters)] for j in range(clusters)])
    for i in range(len(cluA)):
        S[cluA[i], cluB[i]] += 1
    r = sum(S.max(0).T)
    c = sum(S.max(1))
    fsame = (r+c)/(float(len(cluA))*2)
    return fsame

def ACC(y_true, y_pred):
    """
    Calculate clustering accuracy.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def CSEA(filename, clusters, D, loop_count, lr, epochs, batch_size):
    result = []
    k_truss(filename)
    real_result = pd.read_csv('./datasets/' + str(filename) + '_real.csv', header=None)
    real_com = real_result.values.T[0]
    G = nx.from_numpy_matrix(np.array(pd.read_csv('./datasets/' + str(filename) + '_adj.csv', header=None)))
    #adj = np.array(nx.adjacency_matrix(G).todense())
    for i in range(loop_count):
        sim = pd.read_csv('./datasets/' + str(filename) + '_out.csv', header=None)
        x = variational_auto_encoder(sim, D[0:-1:], D[-1::].pop(), lr, epochs, batch_size)
        km = KMeans(n_clusters=clusters)
        km.fit(x)
        kmlb = km.labels_
        Q = modularity(G, kmlb)
        Q = round(float(Q), 6)
        print('Q=', Q)
        nmi = NMI(kmlb, real_com)
        nmi = round(float(nmi), 6)
        print('NMI=', nmi)
        fsame = float(f_same(kmlb, real_com, clusters))
        fsame = round(float(fsame), 6)
        print('Fsame=', float(fsame))
        acc = ACC(real_com, kmlb)
        acc = round(float(acc), 6)
        print('ACC=', float(acc))
        result.append('Loop' + str(i + 1) + ': ' + 'Q=' + str(Q) + ', NMI=' + str(nmi) + ', Fsame=' + str(
            float(fsame)) + ', ACC=' + str(float(acc)))
    return result
