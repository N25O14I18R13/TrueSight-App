document.addEventListener('DOMContentLoaded', () => {
    const videoInput = document.getElementById('video-input');
    const uploadLabel = document.getElementById('upload-label');
    const fileNameDisplay = document.getElementById('file-name-display');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    const uploadBox = document.getElementById('upload-box');
    const loadingBox = document.getElementById('loading-box');
    const loadingFileName = document.getElementById('loading-file-name');
    const resultsBox = document.getElementById('results-box');

    let selectedFile = null;

    videoInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            fileNameDisplay.textContent = selectedFile.name;
            analyzeBtn.disabled = false;
        }
    });

    analyzeBtn.addEventListener('click', () => {

        if (!window.TrueSightApp || !window.TrueSightApp.isUserLoggedIn) {
            if (window.TrueSightApp && window.TrueSightApp.openModal) {
                window.TrueSightApp.openModal(false);
            } else {
                alert('Please sign in to analyze videos.');
            }
            return;
        }

        if (!selectedFile) {
            alert('Please select a video file first.');
            return;
        }

        uploadBox.classList.add('hidden');
        loadingFileName.textContent = selectedFile.name;
        loadingBox.classList.remove('hidden');

        const formData = new FormData();
        formData.append('video', selectedFile);

        fetch('/api/analyze-video', {
            method: 'POST',
            body: formData,
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => {
                    throw new Error(err.error || 'Server error');
                });
            }
            return response.json();
        })
        .then(data => {
            loadingBox.classList.add('hidden');
            resultsBox.classList.remove('hidden');
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert(`An error occurred during analysis: ${error.message}`);
            loadingBox.classList.add('hidden');
            uploadBox.classList.remove('hidden');
        });
    });

    function displayResults(data) {
        const resultText = document.getElementById('result-text');
        resultText.textContent = data.prediction;
        document.getElementById('result-confidence').textContent = `Confidence: ${data.confidence}%`;
        
        if (data.prediction.toUpperCase() === 'FAKE') {
            resultText.style.color = '#ef4444';
        } else {
            resultText.style.color = 'var(--accent)';
        }

        const pie = document.querySelector('.pie');
        let confidencePercent = parseFloat(data.confidence);
        pie.style.setProperty('--p', confidencePercent);
        pie.style.setProperty('--c', data.prediction.toUpperCase() === 'FAKE' ? '#ef4444' : 'var(--accent)');
        pie.textContent = `${data.confidence}%`;

        document.getElementById('frames-analyzed').textContent = data.framesAnalyzed;
        document.getElementById('processing-time').textContent = data.processingTime;
        document.getElementById('model-confidence').textContent = `${data.confidence}%`;

        const framesGrid = document.getElementById('frames-grid');
        framesGrid.innerHTML = '';
        data.frames.forEach(frameSrc => {
            const img = document.createElement('img');
            img.src = `${frameSrc}?t=${new Date().getTime()}`;
            img.alt = 'Analyzed Frame';
            framesGrid.appendChild(img);
        });
    }
});