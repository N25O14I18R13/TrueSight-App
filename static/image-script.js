document.addEventListener('DOMContentLoaded', () => {
    const imageInput = document.getElementById('image-input');
    const uploadLabel = document.getElementById('upload-label');
    const fileNameDisplay = document.getElementById('file-name-display');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    const uploadBox = document.getElementById('upload-box');
    const loadingBox = document.getElementById('loading-box');
    const loadingFileName = document.getElementById('loading-file-name');
    const resultsBox = document.getElementById('results-box');
    const imagePreview = document.getElementById('image-preview');

    let selectedFile = null;

    imageInput.addEventListener('change', (e) => {
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
                alert('Please sign in to analyze images.');
            }
            return;
        }

        if (!selectedFile) {
            alert('Please select an image file first.');
            return;
        }

        uploadBox.classList.add('hidden');
        loadingFileName.textContent = selectedFile.name;
        loadingBox.classList.remove('hidden');

        const formData = new FormData();
        formData.append('image', selectedFile);

        fetch('/api/analyze-image', {
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
        
        imagePreview.innerHTML = '';
        const img = document.createElement('img');
        img.src = data.image_url;
        img.alt = 'Uploaded Image';
        imagePreview.appendChild(img);
    }
});