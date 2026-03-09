document.getElementById("analyzeBtn").addEventListener("click", async () => {

    const text = document.getElementById("textInput").value;

    const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    document.getElementById("result").innerHTML =
        "Phishing: " + data.phishing_score + "<br>" +
        "Stress: " + data.stress_score + "<br>" +
        "Toxicity: " + data.toxicity_score + "<br>" +
        "Final Risk: " + data.final_risk;
});