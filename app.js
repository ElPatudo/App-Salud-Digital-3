let model;
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const resultBox = document.getElementById('resultBox');

// Clases típicas de maíz (ajusta los nombres si es necesario)
const classNames = ["Mancha Gris / Cercospora", "Roya Común", "Tizón Norteño", "Sana"];

// 1. Cargar el modelo al iniciar la página
async function loadModel() {
    console.log("Cargando modelo...");
    try {
        // Al usar tf_saved_model, cargamos como GraphModel
        model = await tf.loadGraphModel('./model/model.json');
        console.log("¡Modelo cargado con éxito!");
        document.getElementById('resClass').innerText = "Modelo listo. Sube una foto.";
    } catch (error) {
        console.error("Error cargando el modelo:", error);
        document.getElementById('resClass').innerText = "Error cargando el modelo.";
    }
}
loadModel();

// 2. Mostrar la imagen cuando el usuario la sube
imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            imagePreview.src = event.target.result;
            imagePreview.style.display = 'block';
            predictBtn.disabled = false; // Habilitar el botón
            resultBox.style.display = 'none'; // Ocultar resultados anteriores
        }
        reader.readAsDataURL(file);
    }
});

// 3. Función para extraer los microindicadores de la imagen
function extractIndicators(imgTensor) {
    return tf.tidy(() => {
        // imgTensor viene en rango [0, 255]. Normalizamos a [0, 1]
        const imgNorm = imgTensor.div(255.0);
        
        // Separar canales RGB
        const [R, G, B] = tf.split(imgNorm, 3, 2);
        const epsilon = 1e-6;
        const sumRGB = R.add(G).add(B).add(epsilon);
        
        const r = R.div(sumRGB);
        const g = G.div(sumRGB);
        const b = B.div(sumRGB);
        
        // Fórmulas de índices
        const ExG = g.mul(2).sub(r).sub(b);
        const GLI = G.mul(2).sub(R).sub(B).div(G.mul(2).add(R).add(B).add(epsilon));
        const VARI = G.sub(R).div(G.add(R).sub(B).add(epsilon));
        
        // Máscaras (Aproximaciones lógicas)
        const maskChlorosis = tf.logicalAnd(tf.logicalAnd(R.greater(0.4), G.greater(0.4)), B.less(0.3)).cast('float32');
        const maskNecrosis = tf.logicalAnd(tf.logicalAnd(R.less(0.3), G.less(0.2)), B.less(0.2)).cast('float32');
        
        const areaClorotica = maskChlorosis.mean();
        const areaNecrotica = maskNecrosis.mean();
        
        // Textura (Desviación estándar de grises)
        const gray = R.mul(0.2989).add(G.mul(0.5870)).add(B.mul(0.1140));
        const meanGray = gray.mean();
        const texture = gray.sub(meanGray).square().mean().sqrt();
        
        const exgMean = ExG.mean();
        const gliMean = GLI.mean();
        const variMean = VARI.mean();
        
        // Retornar un tensor 1D con los 6 indicadores (luego se expande a 2D)
        return tf.stack([exgMean, gliMean, variMean, areaClorotica, areaNecrotica, texture]).expandDims(0);
    });
}

// 4. Ejecutar la predicción
predictBtn.addEventListener('click', async () => {
    if (!model) return;
    
    predictBtn.disabled = true;
    predictBtn.innerText = "Analizando...";
    
    // Usamos tf.tidy para limpiar la memoria de la tarjeta gráfica (GPU/CPU) del navegador
    const results = tf.tidy(() => {
        // Convertir la imagen HTML a Tensor y redimensionar a 224x224
        let imgTensor = tf.browser.fromPixels(imagePreview);
        imgTensor = tf.image.resizeBilinear(imgTensor, [224, 224]);
        
        // Extraer indicadores
        const indicatorsTensor = extractIndicators(imgTensor);
        
        // Expandir dimensiones para que el batch sea 1 -> shape [1, 224, 224, 3]
        const inputImage = imgTensor.expandDims(0);
        
        // Ejecutar el modelo (le pasamos las entradas como un arreglo)
        // Nota: El orden [inputImage, indicatorsTensor] debe coincidir con cómo se construyó el modelo.
        return model.predict([inputImage, indicatorsTensor]);
    });

    // Leer los resultados de los tensores (como es multi-salida, results es un arreglo de tensores)
    // El orden de salida suele coincidir con cómo lo definiste: class_out, severity_out, health_out
    const classData = await results[0].data();
    const severityData = await results[1].data();
    const healthData = await results[2].data();
    
    // Obtener la clase con mayor probabilidad
    const maxProbIndex = classData.indexOf(Math.max(...classData));
    
    // Mostrar en pantalla
    document.getElementById('resClass').innerText = classNames[maxProbIndex];
    document.getElementById('resSeverity').innerText = severityData[0].toFixed(2);
    document.getElementById('resHealth').innerText = healthData[0].toFixed(1);
    
    resultBox.style.display = 'block';
    predictBtn.innerText = "Analizar Hoja";
    predictBtn.disabled = false;
    
    // Limpiar tensores resultantes de la memoria
    results.forEach(tensor => tensor.dispose());
});