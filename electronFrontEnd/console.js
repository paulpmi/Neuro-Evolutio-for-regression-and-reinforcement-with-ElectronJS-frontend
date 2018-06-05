const { ipcRenderer } = require('electron');

ipcRenderer.on('newData', (event, arg) => {

    let data = document.getElementById("Message");
    data.innerHTML += '</br>' + arg;
    
});