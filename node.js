const express = require('express');
const bodyParser = require('body-parser');
const app = express();
const port = 3000;
const base64Img = require('base64-img');
//Import ML and API calls

app.use(bodyParser.json({ limit: '50mb' }));

app.post('/predict', (req, res) => {
    const imageDataURL = req.body.image;
    const base64Data = imageDataURL.replace(/^data:image\/jpeg;base64,/, "");

    base64Img.img(imageDataURL, '', 'capturedImage', (err, filepath, base64Data)=>{
        if(err){
            console.log(err);
            return res.status(500).send("Error saving image");
        }

        //ML Model prediction and API calls
        const crop = predictCrop(filepath);
        const soil = getSoilData();
        const weather = getWeatherData();

        res.json({ crop: crop, soil: soil, weather: weather });

    });
});

function predictCrop(filePath){
    //ML model inference here
    return "Wheat"; //example
}
function getSoilData(){
    //API call
    return "Loamy"; //example
}
function getWeatherData(){
    //API call
    return "Sunny, 25°C"; //example
}

app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
});