// Global variables
let attentionChart, trendChart;
let lastData = [];
let historicalData = [];

// Initialize the dashboard when DOM is loaded
// Refresh more frequently (every 3 seconds)
let lastUpdateHash = '';
let isUpdating = false;
const REFRESH_INTERVAL = 3000; // 3 seconds

document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    
    // Initial load
    loadData();
    
    // Set up auto-refresh
    refreshTimer = setInterval(loadData, REFRESH_INTERVAL);
    
    // Add visibility change listener
    document.addEventListener('visibilitychange', handleVisibilityChange);
});

function handleVisibilityChange() {
    if (document.hidden) {
        // Tab is inactive - pause refreshes
        clearInterval(refreshTimer);
    } else {
        // Tab is active - resume refreshes
        refreshTimer = setInterval(loadData, REFRESH_INTERVAL);
        loadData(); // Immediate refresh when returning to tab
    }
}

// Initialize Chart.js charts
function initializeCharts() {
    // Attention Distribution Chart (Bar Chart)
    const attentionCtx = document.getElementById('attentionChart').getContext('2d');
    attentionChart = new Chart(attentionCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Attention Time (s)',
                    backgroundColor: '#2ecc71',
                    data: []
                },
                {
                    label: 'Distraction Time (s)',
                    backgroundColor: '#e74c3c',
                    data: []
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    stacked: true,
                    grid: {
                        display: false
                    }
                },
                y: {
                    stacked: true,
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Time (seconds)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                }
            }
        }
    });

    // Trend Chart (Line Chart)
    const trendCtx = document.getElementById('trendChart').getContext('2d');
    trendChart = new Chart(trendCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Average Attention Score',
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    fill: true,
                    tension: 0.3,
                    data: []
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Score (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top'
                }
            }
        }
    });
}

// Load data from CSV file
function loadData() {
    if (isUpdating) return;
    isUpdating = true;
    
    // Add cache-buster and request fresh data
    fetch(`attention_log.csv?t=${Date.now()}`)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error ${response.status}`);
            return response.text();
        })
        .then(csvText => {
            const currentHash = hashString(csvText);
            if (currentHash !== lastUpdateHash) {
                lastUpdateHash = currentHash;
                processData(csvText);
                updateLastUpdated();
            }
        })
        .catch(error => {
            console.error('Load error:', error);
            showAlert('Data update failed', 'danger');
        })
        .finally(() => {
            isUpdating = false;
        });
}

// Simple hash function to detect changes
function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash |= 0; // Convert to 32bit integer
    }
    return hash;
}

// Initialize auto-refresh
function startAutoRefresh() {
    // Initial load
    loadData();
    
    // Set up regular refresh
    setInterval(() => {
        if (!document.hidden) { // Only refresh when tab is visible
            loadData();
        }
    }, REFRESH_INTERVAL);
    
    // Listen for visibility changes
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
            loadData(); // Immediate refresh when tab becomes visible
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    startAutoRefresh();
});
function parseCSV(csvText) {
    const lines = csvText.trim().split('\n');
    const headers = lines[0].split(',');

    return lines.slice(1).map(line => {
        const values = line.split(',');
        const obj = {};
        headers.forEach((header, i) => {
            obj[header.trim()] = values[i] ? values[i].trim() : '';
        });
        return obj;
    });
}

// Process CSV data
function processData(csvText) {
    const parsedData = parseCSV(csvText);
    
    if (JSON.stringify(parsedData) !== JSON.stringify(lastData)) {
        lastData = parsedData;
        updateDashboard(parsedData);
        updateHistoricalData(parsedData);
        console.log('Data updated at:', new Date().toLocaleTimeString());
    }
}


// Update all dashboard components
function updateDashboard(data) {
    if (!data || data.length === 0) {
        showAlert('No student data available', 'warning');
        return;
    }

    // Update summary cards
    updateSummaryCards(data);
    
    // Update attention chart
    updateAttentionChart(data);
    
    // Update student table
    updateStudentTable(data);
    
    // Update alerts
    updateAlerts(data);
    
    // Update highlights (top performers and most distracted)
    updateHighlights(data);
}

// Update summary cards
function updateSummaryCards(data) {
    document.getElementById('total-students').textContent = data.length;
    
    const presentStudents = data.filter(student => student.Present === 'Present');
    document.getElementById('present-count').textContent = presentStudents.length;
    
    const avgScore = presentStudents.reduce((sum, student) => 
        sum + parseFloat(student.Attention_Score || 0), 0) / presentStudents.length || 0;
    document.getElementById('avg-attention').textContent = avgScore.toFixed(1) + '%';
    
    const distractedStudents = presentStudents.filter(student => 
        parseFloat(student.Attention_Score || 0) < 60);
    document.getElementById('distracted-count').textContent = distractedStudents.length;
}

// Update attention chart
function updateAttentionChart(data) {
    const names = data.map(student => student.Name);
    const attentionTimes = data.map(student => parseFloat(student.T_Attention_Time || 0));
    const distractionTimes = data.map(student => parseFloat(student.T_Distraction_Time || 0));

    attentionChart.data.labels = names;
    attentionChart.data.datasets[0].data = attentionTimes;
    attentionChart.data.datasets[1].data = distractionTimes;
    attentionChart.update();
}

// Update student table
function updateStudentTable(data) {
    const tableBody = document.getElementById('student-table');
    tableBody.innerHTML = '';

    data.forEach(student => {
        const row = document.createElement('tr');
        
        // Name
        const nameCell = document.createElement('td');
        nameCell.textContent = student.Name;
        row.appendChild(nameCell);
        
        // Status
        const statusCell = document.createElement('td');
        const statusIndicator = document.createElement('span');
        statusIndicator.className = 'status-indicator ';
        
        if (student.Present === 'Absent') {
            statusIndicator.className += 'status-absent';
            statusCell.innerHTML = statusIndicator.outerHTML + 'Absent';
        } else {
            const score = parseFloat(student.Attention_Score || 0);
            if (score >= 60) {
                statusIndicator.className += 'status-attentive';
                statusCell.innerHTML = statusIndicator.outerHTML + 'Attentive';
            } else {
                statusIndicator.className += 'status-distracted';
                statusCell.innerHTML = statusIndicator.outerHTML + 'Distracted';
            }
        }
        row.appendChild(statusCell);
        
        // Attention Time
        const attentionCell = document.createElement('td');
        attentionCell.textContent = (student.T_Attention_Time || '0') + 's';
        row.appendChild(attentionCell);
        
        // Distraction Time
        const distractionCell = document.createElement('td');
        distractionCell.textContent = (student.T_Distraction_Time || '0') + 's';
        row.appendChild(distractionCell);
        
        // Score
        const scoreCell = document.createElement('td');
        const score = parseFloat(student.Attention_Score || 0);
        scoreCell.textContent = score.toFixed(1) + '%';
        
        if (student.Present === 'Absent') {
            scoreCell.innerHTML += ' <span class="badge bg-secondary">Absent</span>';
        } else if (score >= 80) {
            scoreCell.innerHTML += ' <span class="badge bg-success">Excellent</span>';
        } else if (score >= 60) {
            scoreCell.innerHTML += ' <span class="badge bg-primary">Good</span>';
        } else if (score >= 40) {
            scoreCell.innerHTML += ' <span class="badge bg-warning">Low</span>';
        } else {
            scoreCell.innerHTML += ' <span class="badge bg-danger">Poor</span>';
        }
        
        row.appendChild(scoreCell);
        
        tableBody.appendChild(row);
    });
}

// Update historical data and trend chart
function updateHistoricalData(data) {
    const now = new Date();
    const timestamp = now.getHours() + ':' + now.getMinutes().toString().padStart(2, '0');
    
    const presentStudents = data.filter(student => student.Present === 'Present');
    const avgScore = presentStudents.reduce((sum, student) => 
        sum + parseFloat(student.Attention_Score || 0), 0) / presentStudents.length || 0;
    
    historicalData.push({
        timestamp: timestamp,
        avgScore: avgScore
    });
    
    // Keep only the last 12 data points
    if (historicalData.length > 12) {
        historicalData.shift();
    }
    
    // Update trend chart
    trendChart.data.labels = historicalData.map(data => data.timestamp);
    trendChart.data.datasets[0].data = historicalData.map(data => data.avgScore);
    trendChart.update();
}

// Update alerts
function updateAlerts(data) {
    const alertsPanel = document.getElementById('alerts-panel');
    
    // Clear existing alerts
    alertsPanel.innerHTML = '';
    
    const presentStudents = data.filter(student => student.Present === 'Present');
    if (presentStudents.length === 0) {
        alertsPanel.innerHTML = '<div class="alert alert-warning">No students present in class</div>';
        return;
    }
    
    // Check for absent students
    const absentStudents = data.filter(student => student.Present === 'Absent');
    if (absentStudents.length > 0) {
        const alert = document.createElement('div');
        alert.className = 'alert alert-secondary alert-dismissible fade show';
        alert.innerHTML = `
            <strong>${absentStudents.length} students absent:</strong> 
            ${absentStudents.map(s => s.Name).join(', ')}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        alertsPanel.appendChild(alert);
    }
    
    // Check for distracted students
    const distractedStudents = presentStudents.filter(student => 
        parseFloat(student.Attention_Score || 0) < 60);
    
    if (distractedStudents.length === 0) {
        alertsPanel.innerHTML += '<div class="alert alert-success">All present students are attentive!</div>';
        return;
    }
    
    // Add alerts for distracted students
    distractedStudents.forEach(student => {
        const score = parseFloat(student.Attention_Score || 0);
        const alert = document.createElement('div');
        
        if (score < 30) {
            alert.className = 'alert alert-danger alert-dismissible fade show';
            alert.innerHTML = `
                <strong>Critical:</strong> ${student.Name} is highly distracted (${score.toFixed(1)}% attention)
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
        } else if (score < 45) {
            alert.className = 'alert alert-warning alert-dismissible fade show';
            alert.innerHTML = `
                <strong>Warning:</strong> ${student.Name} is very distracted (${score.toFixed(1)}% attention)
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
        } else {
            alert.className = 'alert alert-info alert-dismissible fade show';
            alert.innerHTML = `
                <strong>Notice:</strong> ${student.Name} is distracted (${score.toFixed(1)}% attention)
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            `;
        }
        
        alertsPanel.appendChild(alert);
    });
}

// Update highlights (top performers and most distracted)
function updateHighlights(data) {
    const presentStudents = data.filter(student => student.Present === 'Present');
    
    // Top Performers
    const topPerformersContainer = document.getElementById('top-performers');
    topPerformersContainer.innerHTML = '';
    
    if (presentStudents.length === 0) {
        topPerformersContainer.innerHTML = '<p>No students present</p>';
    } else {
        // Sort by attention score (descending)
        const topPerformers = [...presentStudents].sort((a, b) => 
            parseFloat(b.Attention_Score || 0) - parseFloat(a.Attention_Score || 0)).slice(0, 3);
        
        topPerformers.forEach(student => {
            const score = parseFloat(student.Attention_Score || 0);
            const card = document.createElement('div');
            card.className = 'd-flex justify-content-between align-items-center mb-2 p-2 student-card attentive';
            
            card.innerHTML = `
                <div>
                    <strong>${student.Name}</strong>
                    <div class="text-muted small">${student.T_Attention_Time || 0}s focused</div>
                </div>
                <span class="badge bg-success">${score.toFixed(1)}%</span>
            `;
            
            topPerformersContainer.appendChild(card);
        });
    }
    
    // Most Distracted
    const mostDistractedContainer = document.getElementById('most-distracted');
    mostDistractedContainer.innerHTML = '';
    
    if (presentStudents.length === 0) {
        mostDistractedContainer.innerHTML = '<p>No students present</p>';
    } else {
        // Filter and sort distracted students
        const distracted = presentStudents.filter(student => 
            parseFloat(student.Attention_Score || 0) < 60);
        
        if (distracted.length === 0) {
            mostDistractedContainer.innerHTML = '<p>No distracted students</p>';
        } else {
            const mostDistracted = [...distracted].sort((a, b) => 
                parseFloat(a.Attention_Score || 0) - parseFloat(b.Attention_Score || 0)).slice(0, 3);
            
            mostDistracted.forEach(student => {
                const score = parseFloat(student.Attention_Score || 0);
                const card = document.createElement('div');
                card.className = 'd-flex justify-content-between align-items-center mb-2 p-2 student-card distracted';
                
                card.innerHTML = `
                    <div>
                        <strong>${student.Name}</strong>
                        <div class="text-muted small">${student.T_Distraction_Time || 0}s distracted</div>
                    </div>
                    <span class="badge bg-danger">${score.toFixed(1)}%</span>
                `;
                
                mostDistractedContainer.appendChild(card);
            });
        }
    }
}

// Show alert message
function showAlert(message, type) {
    const alertsPanel = document.getElementById('alerts-panel');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.textContent = message;
    alertsPanel.appendChild(alert);
}

// Update last updated time
function updateLastUpdated() {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    document.getElementById('last-updated').textContent = timeString;
}