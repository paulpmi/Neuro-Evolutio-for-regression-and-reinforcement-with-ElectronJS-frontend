document.getElementById('deterministicButton').addEventListener("click", function(){
    var div = document.getElementById('ReinforcementWrapper');
    div.style.display = 'none';
    var removeButton = document.getElementById('DeterministicWrapper');
    removeButton.style.display = 'flex';
});

document.getElementById('reinforcementButton').addEventListener("click", function(){
    var div = document.getElementById('ReinforcementWrapper');
    var removeButton = document.getElementById('DeterministicWrapper');
    removeButton.style.display = 'none';
    div.style.display = 'flex';
});

if(document.getElementById('deterministicButton').checked) {
    var div = document.getElementById('ReinforcementWrapper');
    var removeButton = document.getElementById('DeterministicWrapper');
    removeButton.style.display = 'flex';
    div.style.display = 'none';
}

if(document.getElementById('reinforcementButton').checked) {
    var div = document.getElementById('ReinforcementWrapper');
    var removeButton = document.getElementById('DeterministicWrapper');
    removeButton.style.display = 'none';
    div.style.display = 'flex';
}