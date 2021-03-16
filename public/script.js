async function loadData() {
  $.getJSON("../src/breast-cancer.json", async function (json) {
    var data = json;

    var columns = [
      "clumpThickness",
      "uniformityOfCellSize",
      "uniformityOfCellShape",
      "marginalAdhesion",
      "singleEpithelialCellSize",
      "bareNuclei",
      "blandChromatin",
      "normalNucleoli",
      "mitoses",
    ];

    var result = data.map(function (obj) {
      return columns.map(function (key) {
        return obj[key];
      });
    });

    $.getJSON("../src/labels.json", async function (json) {
      var labels = json;

      const training = result.splice(0, 400);
      const test = result;

      const trainingLabels = labels.splice(0, 400);
      const testLabels = labels;

      const trainingTensor = tf.tensor2d(training, [training.length, 9]);
      const trainingLabelsTensor = tf.tensor2d(trainingLabels, [trainingLabels.length, 1]);
	  
      const testTensor = tf.tensor2d(test, [test.length, 9]);
      const testLabelsTensor = tf.tensor2d(testLabels, [testLabels.length, 1]);

      model = await trainModel(
        trainingTensor,
        trainingLabelsTensor,
        testTensor,
        testLabelsTensor
      );

      const input = tf.tensor(test, [test.length, 9]);
      const prediction = model.predict(input).argMax(-1).dataSync();
      alert(prediction);
    });
  });
}

async function trainModel(training, trainingLabels, test, testLabels) {
  const model = tf.sequential();
  const learningRate = 0.03;
  const epochs = 50;
  const optimizer = tf.train.adam(learningRate);

  console.log(training);
  input_shape = training.shape;

  model.add(
    tf.layers.dense({ units: 10, activation: "sigmoid", inputShape: [9,]})
  );
  model.add(tf.layers.dense({ units: 10, activation: "sigmoid" }));

  model.compile({
    optimizer: optimizer,
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const history = await model.fit(training, trainingLabels, {
    epochs: epochs,
    validationData: [test, testLabels],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log("Epoch: " + epoch + "Logs: " + logs.acc);
        await tf.nextFrame();
      },
    },
  }).then(info => {
   console.log('Precisão final', info.history.acc);
 });

  return model;
}

loadData();
