const { createImage, setCalibrationAutomatic, setCalibrationSemiAutomatic, autoAnalysis} = require('../mcp-vision-detection');
const express = require("express");
const bodyParser = require("body-parser");
var helmet = require('helmet');
const cors = require('cors');
const https = require("https");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(cors());
app.use(helmet());
app.use(bodyParser.urlencoded({extended:false,limit:"5mb"}));
app.use(bodyParser.json({limit:"5mb"}));


app.get('/',function(req,res) {
   console.log("User in index");
   res.status(200).send({ title: 'Welcome...' });
})

app.get('/create-image', function(req, res) {
  createImage();
  const imagePath = path.join(__dirname, 'hello-world.png');
  res.sendFile(imagePath);
});

app.post('/calibration_automatic', function(req, res, next) {
  console.log("Automatic calibration fixed running...")
  let response = setCalibrationAutomatic(req.body.Screenshot);
  let json = JSON.parse(response.replace(/'/g,'"'))
  console.log(json)
  if(json.status == 0){
     res.status(200).json({stateFinal:"OK!",response:json})
  }else{
     res.status(400).send({stateFinal:"Error!"})
  }
});

app.post('/calibration_semiautomatic', function(req, res, next) {
  console.log("Semi Automatic calibration fixed running...")
  let response = setCalibrationSemiAutomatic(req.body.Screenshot, req.body.marks);
  let json = JSON.parse(response.replace(/'/g,'"'));
  console.log(json);
  if(json.status == 0){
      res.status(200).json({stateFinal:"OK!",response:json})
  }else{
      res.status(400).send({stateFinal:"Error!"})
  }
  });

app.post('/autoAnalysis', function(req, res, next) {
  console.log("Automatic Analysis running...")
  let response = autoAnalysis(req.body.contourjson, req.body.videoUrl, req.body.imageUrl, req.body.jsonString);
  let json = JSON.parse(response.replace(/'/g,'"'));
  console.log(json);
  if(json.output){
      res.status(200).json({stateFinal:"OK!",response:json})
  }else{
      res.status(400).send({stateFinal:"Error!"})
  }
  });


app.listen(3001, () => {
  console.log('Server is running at port 3001');
});
