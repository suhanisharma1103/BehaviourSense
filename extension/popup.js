document.getElementById("analyzeBtn").addEventListener("click", async () => {

    const resultDiv = document.getElementById("result");
    resultDiv.innerText = "Extracting text...";

    const tabs = await chrome.tabs.query({
        active: true,
        currentWindow: true
    });

    const tab = tabs[0];

    chrome.tabs.sendMessage(
        tab.id,
        { action: "extractText" },
        async (response) => {

            if (!response || !response.text) {
                resultDiv.innerText = "No text detected on page.";
                return;
            }

            const text = response.text;

            console.log("Text sent to API:", text.substring(0,200));

            try {

                const api = await fetch("http://127.0.0.1:8000/analyze", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({ text: text })
                });

                const data = await api.json();

                console.log("API response:", data);

                resultDiv.innerText =
                    "BehaviorSense Analysis\n\n" +
                    "Phishing Score: " + data.phishing_score +
                    "\nStress Score: " + data.stress_score +
                    "\nToxicity Score: " + data.toxicity_score +
                    "\nFinal Risk: " + data.final_risk +
                    "\n\nGemini Insight:\n" + data.gemini_analysis;

            } catch (err) {

                resultDiv.innerText =
                    "Backend not reachable.\nRun FastAPI server.";

                console.error(err);
            }

        }
    );

});