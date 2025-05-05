let fps = 25;  // Change this if your video has different FPS

function setupVideoPlayer(videoId, tableId, logData) {
    const video = document.getElementById(videoId);
    const tableBody = document.getElementById(tableId).getElementsByTagName('tbody')[0];

    video.addEventListener('timeupdate', function () {
        const currentFrame = Math.floor(video.currentTime * fps);
        const frameData = logData[currentFrame];

        // Clear table
        tableBody.innerHTML = '';

        if (frameData) {
            frameData.forEach(entry => {
                let row = tableBody.insertRow();
                row.insertCell(0).innerText = entry.tracker_id;
                row.insertCell(1).innerText = entry.speed;
                let statusCell = row.insertCell(2);
                let status = entry.speed > 100 ? 'Speeding' : 'Normal';
                statusCell.innerHTML = `<span class="status-${status.toLowerCase()}">${status}</span>`;
            });
        } else {
            let row = tableBody.insertRow();
            let cell = row.insertCell(0);
            cell.colSpan = 3;
            cell.innerText = "No data for this frame.";
        }
    });

    // Trigger initial update
    video.dispatchEvent(new Event('timeupdate'));
}

// Load log data from server and setup video players
Promise.all([
    fetch('/log_data1').then(response => response.json()),
    fetch('/log_data2').then(response => response.json())
]).then(([data1, data2]) => {
    setupVideoPlayer('videoPlayer1', 'vehicleTable1', data1);
    setupVideoPlayer('videoPlayer2', 'vehicleTable2', data2);
}).catch(error => {
    console.error('Error loading log data:', error);
});