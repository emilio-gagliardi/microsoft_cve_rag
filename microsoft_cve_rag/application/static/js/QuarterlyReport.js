// QuarterlyReport.js
document.addEventListener('DOMContentLoaded', function() {
    console.log("Report JS Loaded. Finding chart containers...");
    const chartContainers = document.querySelectorAll('.plotly-chart-container');
    console.log(`Found ${chartContainers.length} chart containers.`);

    if (chartContainers.length === 0) {
        console.warn("No elements with class 'plotly-chart-container' found.");
        return; // Exit if no containers found
    }

    chartContainers.forEach((container, index) => {
        // Ensure the container itself has an ID
        if (!container.id) {
            console.warn(`Chart container at index ${index} is missing an ID. Cannot render Plotly chart.`);
            container.innerHTML = `<div class="text-red-500 text-center p-4">Error: Chart container must have an ID.</div>`;
            return; // Skip this container
        }

        const chartId = container.id;
        const chartJsonPath = `../data/${chartId}.json`; // Adjust path if needed

        console.log(`[${chartId}] Attempting to load chart data from: ${chartJsonPath}`);

        // Display loading state clearly
        container.innerHTML = `<div class="flex items-center justify-center h-full min-h-[400px] text-report-text-muted dark:text-report-text-muted-dark">
                                  <svg class="animate-spin -ml-1 mr-3 h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                  </svg>
                                  <span>Loading ${chartId}...</span>
                               </div>`;

        fetch(chartJsonPath)
            .then(response => {
                console.log(`[${chartId}] Fetch response received for ${chartJsonPath}. Status: ${response.status}`);
                if (!response.ok) {
                    console.error(`[${chartId}] HTTP error! Status: ${response.status}, Path: ${chartJsonPath}`);
                    throw new Error(`HTTP error! status: ${response.status} for ${chartJsonPath}`);
                }
                return response.json();
            })
            .then(figureData => {
                console.log(`[${chartId}] Successfully parsed JSON data.`);
                if (figureData && Array.isArray(figureData.data) && typeof figureData.layout === 'object') {

                    // ---> Explicitly clear the container BEFORE plotting <---
                    container.innerHTML = '';
                    console.log(`[${chartId}] Cleared loading spinner content.`);
                    // --------------------------------------------------------

                    console.log(`[${chartId}] Figure data structure looks valid. Calling Plotly.newPlot...`);
                    try {
                        Plotly.newPlot(chartId, figureData.data, figureData.layout, { responsive: true, displaylogo: false });
                        console.log(`[${chartId}] Plotly.newPlot call completed successfully.`);
                    } catch (plotError) {
                        console.error(`[${chartId}] Error occurred *during* Plotly.newPlot execution:`, plotError);
                        // If plotting fails, the container might be empty now, display error again
                        container.innerHTML = `<div class="text-red-500 text-center p-4">Error rendering chart ${chartId}. Plotly error (See console).</div>`;
                    }
                } else {
                    console.error(`[${chartId}] Invalid Plotly figure data structure received:`, figureData);
                    container.innerHTML = `<div class="text-red-500 text-center p-4">Error: Invalid chart data format for ${chartId}.</div>`;
                }
            })
            .catch(error => {
                console.error(`[${chartId}] Error fetching/parsing chart data for ${chartId}:`, error);
                 container.innerHTML = `<div class="text-red-500 text-center p-4">Error loading chart data for ${chartId}. Check path: ${chartJsonPath} (See console).</div>`;
            });
    });
});
