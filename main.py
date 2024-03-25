import numpy as np
import networkx as nx
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Classe Graph para gerar os dados de entrada
class Graph:
    def __init__(self, vertix=4):
        self.vertix = vertix
        self.array = np.random.randint(2, size=(self.vertix, self.vertix))
    
    def get_matrix(self):
        return self.array

    def get_flatten_matrix(self):
        return self.get_matrix().flatten()
    
    def get_graph(self):
        return nx.from_numpy_array(self.array)
    
    def get_chrome_number(self):
        return len(set(nx.coloring.greedy_color(self.get_graph()).values()))

# Gerando o dataset
N = 8
max_qtd = 782

X, y = [], []
for i in range(max_qtd):
    g = Graph(N)
    X.append(g.get_flatten_matrix())
    y.append(g.get_chrome_number())

# Dividindo o dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Definindo a classe da camada quântica
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, bit_length_response, shots=10000, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.bit_length_response = bit_length_response
        self.shots = shots

    def build(self, input_shape):
        self.N = input_shape[-1]
        self.theta = self.add_weight(name='theta', shape=(self.N,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        outputs = []
        for i in range(inputs.shape[0]):
            output_key, output_value = quantum_nn(inputs[i], self.theta.numpy(), self.bit_length_response, self.shots)
            outputs.append(output_key)
        return tf.convert_to_tensor(outputs, dtype=tf.int32)

    def get_config(self):
        config = super(QuantumLayer, self).get_config()
        config.update({
            'bit_length_response': self.bit_length_response,
            'shots': self.shots,
        })
        return config

# Funções auxiliares para a simulação quântica
def binary2int(binary_str):
    return int(binary_str, 2)

def max_value(dictionary):
    return max(dictionary.items(), key=lambda x: x[1])

# Função para simulação quântica
def quantum_nn(inputs, theta, bit_length_response, shots):
    qc, cReg = feature_map(inputs, bit_length_response)
    qc = variational_circuit(qc, theta, inputs)

    for i in range(bit_length_response):
        qc.measure(i, cReg[i])

    results = AerSimulator().run(qc, shots=shots, memory=True).result()
    counts = results.get_counts(qc)

    stats = {}
    for key, value in counts.items():
        stats[binary2int(key)] = value / shots

    max_key, max_v = max_value(stats)

    return max_key, max_v

# Construindo o modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(N, activation=tf.nn.relu, input_shape=(N*N,), name="Dense_1"),
    QuantumLayer(N, name="QuantumLayer"),
    tf.keras.layers.Dense(N+1, activation=tf.nn.softmax, name="Dense_2")
])

# Compilando o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_data=(np.array(X_test), np.array(y_test)))
