document.getElementById('fileInput').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview').innerHTML = 
                `<img src="${e.target.result}" alt="Preview">`;
        };
        reader.readAsDataURL(file);
    }
});

document.getElementById('uploadForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image file');
        return;
    }
    
    // Show loading
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    
    // Create form data
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + data.error);
        }
    } catch (error) {
        alert('Error processing image: ' + error.message);
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
});

function displayResults(data) {
    // Show results section
    document.getElementById('results').classList.remove('hidden');
    
    // Update weed count
    document.getElementById('weedCount').textContent = data.weed_count;
    
    // Display result image
    document.getElementById('resultImg').src = data.result_image;
    
    // Display coordinates
    const coordinatesList = document.getElementById('coordinatesList');
    coordinatesList.innerHTML = '';
    
    data.laser_coordinates.forEach(coord => {
        const card = document.createElement('div');
        card.className = 'coordinate-card';
        card.innerHTML = `
            <h4>Target #${coord.target_id}</h4>
            <p><strong>X:</strong> ${coord.x} mm</p>
            <p><strong>Y:</strong> ${coord.y} mm</p>
            <p><strong>Class:</strong> ${coord.class_name}</p>
            <p><strong>Confidence:</strong> ${(coord.confidence * 100).toFixed(1)}%</p>
        `;
        coordinatesList.appendChild(card);
    });
    
    // Scroll to results
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}
```

### **13. .gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Flask
instance/
.webassets-cache

# PyCharm
.idea/

# VS Code
.vscode/

# Jupyter Notebook
.ipynb_checkpoints

# Dataset (too large for git)
dataset/
!dataset/.gitkeep

# Model weights (too large for git)
*.pt
*.pth
*.onnx
runs/

# Uploads and results
static/uploads/*
!static/uploads/.gitkeep
static/results/*
!static/results/.gitkeep

# OS
.DS_Store
Thumbs.db