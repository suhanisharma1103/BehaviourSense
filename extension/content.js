console.log("BehaviorSense content script loaded");

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {

    if (request.action === "extractText") {

        try {

            let text = "";

            // Check selected text first
            const selectedText = window.getSelection().toString().trim();

            if (selectedText.length > 10) {
                text = selectedText;
            } else {
                // fallback → full page text
                text = document.body.innerText || "";
            }

            // limit size
            text = text.substring(0, 2000);

            console.log("Extracted text length:", text.length);

            sendResponse({
                text: text
            });

        } catch (error) {

            console.error("Extraction error:", error);

            sendResponse({
                text: ""
            });

        }

        return true;
    }

});