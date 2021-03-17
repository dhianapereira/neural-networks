# Rede Neural

Rede neural de um conjunto de dados de câncer de mama que gera uma probabilidade e categoriza novos pacientes.

## Dados para o treinamento

O modelo foi construído com a utilização de um conjunto de dados de 699 pacientes com câncer de mama. O conjunto de dados passou por normalização e limpeza, o que resultou em 500 pacientes no conjunto de dados finais para treinamento e teste.

São 500 pacientes no total, sendo 262 (52,4%) com casos de tumores benignos e 238 (47,6%) com casos de tumores malignos. Para o treinamento foram utilizados 80% dos dados, sendo 40% de casos de tumores benignos e 40% de tumores malignos, e para o teste os 20% restantes. Destes 20%, 12,4% são de tumores benignos e 7,6% são de tumores malignos.

## Conteúdo

- [TensorFlow.js — Making Predictions from 2D Data](https://codelabs.developers.google.com/codelabs/tfjs-training-regression/#0)
- [Tinker With a Neural Network Right Here in Your Browser](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.16809&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)
- [Build a simple Neural Network with TensorFlow.js](https://towardsdatascience.com/build-a-simple-neural-network-with-tensorflow-js-d434a30fcb8)
