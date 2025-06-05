async function detectAlphabet() {
    try {
        const response = await fetch('http://localhost:8000/get_alphabet');
        const data = await response.json();
        displayResult(data.alphabet || 'No result found');
    } catch (error) {
        displayResult('Error: Could not detect alphabet');
    }
}

async function detectWord() {
    try {
        const response = await fetch('http://localhost:8000/get_word');
        const data = await response.text();
        displayResult(data || 'No result found');
    } catch (error) {
        displayResult('Error: Could not detect word');
    }
}
// Create a global variable to manage the speech synthesis
let speechInProgress = false;

function speakText(text) {
    if (speechInProgress) return;
    const paddedText =  text;
    const utterance = new SpeechSynthesisUtterance(paddedText);

    utterance.onstart = function () {
        speechInProgress = true;
        document.getElementById('tts-status').style.display = 'inline';
    };

    utterance.onend = function () {
        speechInProgress = false;
        document.getElementById('tts-status').style.display = 'none';
    };

    window.speechSynthesis.speak(utterance);
}



// Modified displayResult function to call speakText when result is displayed
function displayResult(result) {
    const resultBox = document.getElementById('result');
    resultBox.textContent = result;
    resultBox.style.display = 'block';

    // Call the speakText function to say the result aloud
    speakText(result);
}

