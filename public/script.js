function loadData() {
    var data = [];
  
    $.getJSON("../src/breast-cancer.json", function (json) {
      data = json;
  
      const training = data.splice(0, 400);
      console.log(training);
  
      const test = data;
  
      console.log(test);
    });
  }