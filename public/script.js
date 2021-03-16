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
      console.log(training);

      const test = result;
      console.log(test);

      const trainingLabels = labels.splice(0, 400);
      const testLabels = labels;

      const trainingTensor = tf.tensor2d(training, [training.length, 9]);
      const trainingLabelsTensor = tf.tensor2d(trainingLabels, [
        trainingLabels.length,
        1,
      ]);
      const testTensor = tf.tensor2d(test, [test.length, 9]);
      const testLabelsTensor = tf.tensor2d(testLabels, [testLabels.length, 1]);

      model = await trainModel(
        trainingTensor,
        trainingLabelsTensor,
        testTensor,
        testLabelsTensor
      );

      const input = tf.tensor2d(test, [test.length, 9]);
      const prediction = model.predict(input).argMax(-1).dataSync();
      alert(prediction);
    });
  });
}

async function trainModel(training, trainingLabels, test, testLabels) {
  const model = tf.sequential();
  const learningRate = 0.01;
  const epochs = 50;
  const optimizer = tf.train.adam(learningRate);

  model.add(
    tf.layers.dense({ units: 10, activation: "sigmoid", inputShape: [training.length, 9] })
  );
  model.add(tf.layers.dense({ units: 2, activation: "softmax" }));

  model.compile({
    optimizer: optimizer,
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const history = await model.fit(tf.stack(training), trainingLabels, {
    epochs: epochs,
    validationData: [test, testLabels],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log("Epoch: " + epoch + "Logs: " + logs.loss);
        await tf.nextFrame();
      },
    },
  });

  return model;
}

loadData();
