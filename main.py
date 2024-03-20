import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

import tensorflow as tf
# print(tf.__version__)
tf.config.experimental_run_functions_eagerly(True)

# from create_graphs import Graph

print(f"\nCriar classe para os grafos\n")
import networkx as nx

class Graph:
    def __init__(self, vertix=4) -> None:
        self.vertix = vertix
        self.array = np.random.randint(2, size=(self.vertix, self.vertix))
    
    def get_matrix(self):
        return self.array
    
    def get_graph(self):
        return nx.from_numpy_array(self.array)
    
    def get_chrome_number(self):
        return len(set(nx.coloring.greedy_color(self.get_graph()).values()))

    
if __name__ == "__main__":
    
    graph_list = []
    for _ in range(3):
        graph_list.append(Graph(5))

    for i in graph_list:
        print(i.get_graph())

print(f"\nConfigurando N, epocas e quantidade de exemplos para teste e treino\n")
N = 4
max_qtd = 782
epochs = 100


X, y = [], []
for i in range(max_qtd):
    g = Graph(N)
    X.append(g.get_matrix().flatten())
    y.append(g.get_chrome_number())

print(f"\nDividindo dataset em treino e teste \n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(f"\nConfigurando funcoes auxiliares \n")
def binary2int(key):
    if not isinstance(key, str):
        key = str(key)

    return int(key, 2)

def bit_length(number):
    if not isinstance(number, int):
        number = int(number)

    return (number).bit_length()

def max_value(dic):
    max_key = max(dic, key=dic.get)

    max_value = dic[max_key]

    return max_key, max_value

print(f"\nFeature map \n")
def feature_map(inputs, bit_length_response):

    qReg = QuantumRegister(len(inputs))
    print(f"feature maps:{inputs}")
    
    cReg = ClassicalRegister(bit_length_response)
    qc = QuantumCircuit(qReg, cReg)
    
    for i, x in enumerate(inputs):
        qc.rx(x, i)
    
    qc.barrier()
    
    return qc, cReg

print(f"\nVariational Circuit \n")
def variational_circuit(qc, theta, inputs):
    
    for i in range(len(inputs) - 1):
        qc.cx(i+1, i)
    
    qc.cx(0, len(inputs)-1)
    qc.barrier()
    
    for i in range(len(inputs)):
        qc.ry(theta[i], i)

    qc.barrier()

    return qc

print(f"\nMeasure \n")
def measure(qc, bit_length_response):
    for i in range(bit_length_response):
        qc.measure(i, c[i])

    return qc

print(f"\nQuantum Layer \n")
class QuantumLayer(tf.keras.layers.Layer):
    def __init__(self, bit_length_response, shots=1E4, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.bit_length_response = bit_length_response
        self.shots = shots

    def build(self, input_shape):
        self.N = input_shape[-1]
        self.n_response = bit_length(self.bit_length_response) # self.bit_length_response
        self.theta = self.add_weight(name='theta', shape=(self.N,), initializer='random_normal', trainable=True)

    def call(self, inputs):
        if tf.executing_eagerly():
            inputs_np = inputs.numpy()[-1]
            theta_np = self.theta.numpy()

            qc, cReg = feature_map(inputs_np, self.bit_length_response)
            qc = variational_circuit(qc, theta_np, inputs_np)
    
            for i in range(self.bit_length_response):
                qc.measure(i, cReg[i])
    
            results = AerSimulator().run(qc, shots=self.shots, memory=True).result()
            counts = results.get_counts(qc)
    
            stats = {}
            for key, value in counts.items():
                stats[binary2int(key)] = value / self.shots
    
            max_key, max_v = max_value(stats)
    
            print(f"Key: {max_key}")

            outputs = np.zeros(self.N+1)
            if not max_key >= self.N:
                outputs[max_key] = max_v
            
            outputs = tf.constant(outputs, dtype=tf.float32)

            print(f"Outputs: {outputs}")
            return outputs

        return inputs

    def get_config(self):
        config = super(QuantumLayer, self).get_config()
        config.update({
            'bit_length_response': self.bit_length_response,
            'shots': self.shots,
        })
        return config
        

print(f"\nCriando Modelo de Rede Neural Quantica \n")
# Modificação da criação do modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(N, activation='relu'),
    QuantumLayer(N),
    tf.keras.layers.Reshape((N+1, 1)),  # Adiciona uma camada Reshape para tornar a saída bidimensional
    tf.keras.layers.Dense(N, activation='softmax')
])

print(f"\nInicializando o modelo \n")
model.build()

print(f"\nCompilando o Modelo \n")
#### Compilando o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(f"\nNp array de X_train e X_test\n")
X_train = np.array(X_train).astype('float32').reshape(-1, N) 
X_test = np.array(X_test).astype('float32').reshape(-1, N) 
y_train = np.array(y_train).astype('float32').reshape(-1, 1)
y_test = np.array(y_test).astype('float32').reshape(-1, 1)

print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de X_test: {X_test.shape}")
print(f"Forma de y_train: {y_train.shape}")
print(f"Forma de y_test: {y_test.shape}")

print(f"Tamanho X: {len(X_train)} e Y: {len(y_train)}")
# Treinando o modelo e salvando o histórico
print(f"\nTreinamento do modelo...\n")
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))

# Obtendo as métricas de acurácia do histórico
print(f"\nConfigurando Acuracia...\n")
train_accuracy = history.history['accuracy']
test_accuracy = history.history['val_accuracy']

# Construindo o gráfico
print(f"\nConstruindo Grafico...\n")
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Acurácia de Treino')
plt.plot(range(1, len(test_accuracy) + 1), test_accuracy, label='Acurácia de Teste')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.title('Acurácia ao longo das Épocas')
plt.legend()
plt.show()

