document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const form = e.target;
    const formData = new FormData(form);
    const data = {};
    
    const numericFields = [
        'Patient_Age', 'Claim_Amount', 'Approved_Amount', 
        'Days_Between_Service_and_Claim', 'Number_of_Claims_Per_Provider_Monthly',
        'Length_of_Stay', 'Chronic_Condition_Flag', 'Prior_Visits_12m'
    ];

    formData.forEach((value, key) => {
        if (key === 'Patient_Name') return;
        
        if (numericFields.includes(key)) {
            data[key] = Number(value);
        } else {
            data[key] = value;
        }
    });

    const btn = document.getElementById('analyze-btn');
    const btnText = btn.querySelector('.btn-text');
    const spinner = document.getElementById('btn-spinner');
    
    // UI Loading state
    btnText.style.display = 'none';
    spinner.style.display = 'block';
    btn.disabled = true;
    
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        if (!response.ok) {
            throw new Error('API Request Failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        alert("Error connecting to the intelligence server: " + error.message);
    } finally {
        // Reset UI
        btnText.style.display = 'block';
        spinner.style.display = 'none';
        btn.disabled = false;
    }
});

function displayResults(data) {
    const panel = document.getElementById('results-panel');
    const circle = document.getElementById('progress-circle');
    const confidenceText = document.getElementById('confidence-value');
    const bandBadge = document.getElementById('band-badge');
    const riskDesc = document.getElementById('risk-description');
    
    panel.style.display = 'block';
    
    // Scroll to results smoothly
    panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Animate percentage
    const targetPercent = Math.round(data.confidence * 100);
    let currentPercent = 0;
    
    // Explicitly reset the UI before animating so it doesn't get stuck on old values
    confidenceText.textContent = `0%`;
    circle.style.background = `conic-gradient(var(--risk-negligible) 0deg, rgba(255,255,255,0.05) 0deg)`;
    
    const interval = setInterval(() => {
        if (currentPercent >= targetPercent) {
            clearInterval(interval);
        } else {
            currentPercent++;
            confidenceText.textContent = `${currentPercent}%`;
            
            // Determine color based on band
            let color = 'var(--risk-negligible)';
            if (currentPercent >= 80) color = 'var(--risk-critical)';
            else if (currentPercent >= 60) color = 'var(--risk-elevated)';
            else if (currentPercent >= 40) color = 'var(--risk-moderate)';
            
            circle.style.background = `conic-gradient(${color} ${currentPercent * 3.6}deg, rgba(255,255,255,0.05) 0deg)`;
        }
    }, 20); // 20ms per step
    
    // Update Badge
    bandBadge.textContent = data.band;
    bandBadge.className = `band-badge ${data.band.toLowerCase()}`;
    
    // Set Description
    if (data.band === 'CRITICAL') {
        riskDesc.textContent = "High probability of anomaly detected. Immediate manual review and escalation required.";
    } else if (data.band === 'ELEVATED') {
        riskDesc.textContent = "Significant risk indicators present. Secondary audit recommended before approval.";
    } else if (data.band === 'MODERATE') {
        riskDesc.textContent = "Some irregularities detected. Standard verification procedures should be sufficient.";
    } else {
        riskDesc.textContent = "No significant risk indicators detected. Claim aligns with expected historical patterns.";
    }
    
    // Set Reasons
    const reasonsContainer = document.getElementById('reasons-container');
    const reasonsList = document.getElementById('reasons-list');
    
    if (data.reasons && data.reasons.length > 0 && targetPercent >= 20) {
        reasonsList.innerHTML = '';
        
        // Find the maximum contribution for scaling the bars
        const maxContribution = Math.max(...data.reasons.map(r => r.contribution));
        
        data.reasons.forEach(r => {
            const barWidth = Math.max(5, (r.contribution / maxContribution) * 100);
            
            const li = document.createElement('li');
            li.style.background = 'rgba(255, 255, 255, 0.05)';
            li.style.padding = '0.8rem 1rem';
            li.style.borderRadius = '8px';
            li.style.marginBottom = '0.8rem';
            li.style.borderLeft = '4px solid var(--risk-elevated)';
            
            li.innerHTML = `
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <strong>${r.feature}</strong>
                    <span style="color: var(--text-muted); font-size: 0.85em;">Primary Driver</span>
                </div>
                <div style="width: 100%; height: 6px; background: rgba(0,0,0,0.3); border-radius: 3px; overflow: hidden;">
                    <div style="width: ${barWidth}%; height: 100%; background: linear-gradient(90deg, #f97316, #ef4444); border-radius: 3px; transition: width 1s ease-out;"></div>
                </div>
            `;
            reasonsList.appendChild(li);
        });
        reasonsContainer.style.display = 'block';
    } else {
        reasonsContainer.style.display = 'none';
    }
}
