// Path to the ONNX model file
const MODEL_PATH = './house_price_model.onnx';

async function run() {
    const outputDiv = document.getElementById('out');

    try {
        // 1. Load session only once (Session Caching)
        if (!window.session) {
            outputDiv.innerText = "Loading model for the first time...";
            window.session = await ort.InferenceSession.create(MODEL_PATH);
        }

        outputDiv.innerText = "Running inference...";

        // 2. Collect inputs from f1 to f12
        const vals = [];
        for (let i = 1; i <= 12; i++) {
            const el = document.getElementById("f" + i);
            let val = parseFloat(el.value);
            // Default to 0 if input is empty or invalid
            if (isNaN(val)) val = 0.0;
            vals.push(val);
        }

        // 3. Create Tensor (Shape: [1, 12])
        const tensor = new ort.Tensor("float32", Float32Array.from(vals), [1, 12]);

        // 4. Run Inference
        // We dynamically fetch the input name (usually 'X') to ensure it works 
        // with whatever name the converter assigned.
        const feeds = {};
        feeds[window.session.inputNames[0]] = tensor;

        const results = await window.session.run(feeds);

        // 5. Get Output
        // We dynamically fetch the output name (usually 'variable')
        const outputName = window.session.outputNames[0];
        const prediction = results[outputName].data[0];

        outputDiv.innerText = `Predicted Price: $${prediction.toFixed(2)}`;

    } catch (e) {
        console.error(e);
        outputDiv.innerText = "Error: " + e.message;
    }
}