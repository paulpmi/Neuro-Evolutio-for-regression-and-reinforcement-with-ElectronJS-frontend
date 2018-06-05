function getParameters(){
  var population = document.getElementById('NumberPickerPopulation');
  var iteration = document.getElementById('NumberPickerIteration');
  var input = document.getElementById('NumberPickerInput');
  var hidden = document.getElementById('NumberPickerHidden');
  var output = document.getElementById('NumberPickerOutput');
  var layersNr = document.getElementById('NumberPickerLayers');
  var test = document.getElementById("Test");

  return {
    population: population.value,
    iteration: iteration.value,
    input: input.value,
    hidden: hidden.value,
    output: output.value,
    layersNr: layersNr.value,
    test: test.checked
  }
}

function getReinforcementParamters(){
  var population = document.getElementById('NumberPickerPopulation');
  var iteration = document.getElementById('NumberPickerIteration');
  var input = document.getElementById('NumberPickerInput');
  var hidden = document.getElementById('NumberPickerHidden');
  var output = document.getElementById('NumberPickerOutput');
  var layersNr = document.getElementById('NumberPickerLayers');

  var cart = document.getElementById('CartPole-v0').checked;
  var mountain = document.getElementById('MountainCarContinuous-v0').checked;

  var simulation = '';
  if (cart == true)
    simulation = 'CartPole-v0';

  if (mountain == true)
    simulation = 'MountainCarContinuous-v0';

  return{
    population: population.value,
    iteration: iteration.value,
    input: input.value,
    hidden: hidden.value,
    output: output.value,
    layersNr: layersNr.value,
    simulation: simulation
  }
}

const runDeterministic = function() {

  var parameters = getParameters();
  var pop = parameters.population;
  var iter = parameters.iteration;
  var input = parameters.input;
  var hidden = parameters.hidden;
  var output = parameters.output;
  var layersNr = parameters.layersNr;
  var test = parameters.test;

  const remote = require('electron').remote;
  const app = remote.app;

  var childProcess = require('child_process');
  var path = require("path");
  let python = childProcess.spawn('python3', ['-u', path.join(app.getAppPath(), '', './DeterministicEndPoint.py'), 
  pop.toString(), iter.toString(), input.toString(), hidden.toString(), output.toString(), layersNr.toString(), test.toString()]);
  
  const BrowserWindow = remote.BrowserWindow;

  let win = new BrowserWindow({ width: 800, height: 600 });
  win.loadURL('file://' + __dirname + '/console.html');
  //win.webContents.openDevTools();

  python.stdout.on('data', function(data){
    win.webContents.send('newData', data.toString());
  });

  python.stdout.on('end', () => {
    console.log("Learning Ended");
    win.webContents.send('newData', "LearningEnded");
  });
}

const runReinforcement = function(){
  var parameters = getReinforcementParamters();
  var pop = parameters.population;
  var iter = parameters.iteration;
  var input = parameters.input;
  var hidden = parameters.hidden;
  var output = parameters.output;
  var layersNr = parameters.layersNr;
  var simulation = parameters.simulation;

  const remote = require('electron').remote;
  const app = remote.app;

  var childProcess = require('child_process');
  var path = require("path");
  let python = childProcess.spawn('python3', ['-u', path.join(app.getAppPath(), '', './ReinforcementEndPoint.py'), 
  pop.toString(), iter.toString(), input.toString(), hidden.toString(), output.toString(), layersNr.toString(), simulation]);
  
  const BrowserWindow = remote.BrowserWindow;

  let win = new BrowserWindow({ width: 800, height: 600 });
  win.loadURL('file://' + __dirname + '/console.html');
  //win.webContents.openDevTools();

  python.stdout.on('data', function(data){
    win.webContents.send('newData', data.toString());
  });

  python.stdout.on('end', () => {
    console.log("Learning Ended");
    win.webContents.send('newData', "LearningEnded");
  });
}