const form = document.getElementById("upload-form");
const resultDiv = document.getElementById("result");
const previewImg = document.getElementById("preview");

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById("file-input");
    if (!fileInput.files[0]) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    resultDiv.innerHTML = "Predicting...";
    previewImg.src = URL.createObjectURL(fileInput.files[0]);
    previewImg.style.display = "block";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });
        const data = await response.json();
        if (data.predicted_class) {
            resultDiv.innerHTML = `Predicted Class: ${data.predicted_class}`;
        } else if (data.error) {
            resultDiv.innerHTML = `Error: ${data.error}`;
        }
    } catch (err) {
        resultDiv.innerHTML = `Error: ${err}`;
    }
});
