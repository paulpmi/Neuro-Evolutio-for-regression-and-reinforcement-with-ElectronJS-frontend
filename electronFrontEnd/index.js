const electron = require('electron');

const app = electron.app;

let BrowserWindow = electron.BrowserWindow;

let mainWindow;

app.on('ready', () => {
    mainWindow = new BrowserWindow({width: 1024, height: 720})

    mainWindow.webContents.openDevTools();

    mainWindow.loadURL('file://' + __dirname + '/index.html');
});
