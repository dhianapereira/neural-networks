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
	  
	  tf.util.shuffleCombo(training, trainingLabels);

      const trainingTensor = tf.tensor2d(training, [training.length, 9]);
      const trainingLabelsTensor = tf.tensor2d(trainingLabels, [
        trainingLabels.length,
        1,
      ]);

      const testTensor = tf.tensor2d(test, [test.length, 9]);
      const testLabelsTensor = tf.tensor2d(testLabels, [testLabels.length, 1]);

	//console.log(trainingTensor.shape[0]);
	  //indices = tf.range(start=0, limit=trainingTensor.shape[0], dtype=tf.int32);
	  //console.log(indices);
	  //shuffledIndices = tf.random.shuffle(indices);
	  
	  //shuffledTrainingTensor = tf.gather(trainingTensor, shuffledIndices);
	  //shuffledTrainingLabelsTensor = tf.gather(trainingLabelsTensor, shuffledIndices);

      model = await trainModel(
        trainingTensor,
        trainingLabelsTensor,
        testTensor,
        testLabelsTensor
      );

      const input = tf.tensor(test, [test.length, 9]);
      const prediction = model.predict(input).argMax(-1).dataSync();

      let counter = 0;
      for (let i = 0; i < prediction.length; i++) {
        if (prediction[i] == testLabels[i]) {
          counter++;
        }
      }

      const percentage = (counter * 100) / testLabels.length;
      console.log("Porcentagem de acerto:" + percentage + "%");
    });
  });
}

async function trainModel(training, trainingLabels, test, testLabels) {
  const model = tf.sequential();
  const learningRate = 0.03;
  const epochs = 200;
  const optimizer = tf.train.adam(learningRate);

  model.add(
    tf.layers.dense({ units: 10, activation: "sigmoid", inputShape: [9] })
  );
  
  model.add(tf.layers.dense({ units: 2, activation: "sigmoid" }));

  model.compile({
    optimizer: optimizer,
    loss: "sparseCategoricalCrossentropy",
    metrics: ["accuracy"],
  });

  model.summary();
  const history = await model.fit(training, trainingLabels, {
    epochs: epochs,
    validationData: [test, testLabels],
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log("Epoch: " + epoch + "Logs: " + logs.acc);
        await tf.nextFrame();
      },
    })
    .then((info) => {
      console.log("Precisão final", info.history.acc);
    });

  return model;
}

loadData();
