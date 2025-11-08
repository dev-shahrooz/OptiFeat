(function () {
    const canvas = document.getElementById('candidateChart');
    if (!canvas || !window.optifeatCandidates) {
        return;
    }

    const context = canvas.getContext('2d');
    const candidates = window.optifeatCandidates;
    if (!Array.isArray(candidates) || candidates.length === 0) {
        return;
    }
    const padding = 40;
    const width = canvas.clientWidth || 600;
    const height = 320;

    canvas.width = width;
    canvas.height = height;

    const maxValue = Math.max(...candidates.map(item => item.accuracy), 0.0001);
    const barWidth = (width - padding * 2) / candidates.length;

    context.clearRect(0, 0, width, height);
    context.fillStyle = '#f8fafc';
    context.fillRect(0, 0, width, height);

    candidates.forEach((item, index) => {
        const value = item.accuracy;
        const cost = item.cost;
        const barHeight = (value / maxValue) * (height - padding * 2);
        const x = padding + index * barWidth;
        const y = height - padding - barHeight;

        context.fillStyle = '#4b8df8';
        context.fillRect(x + 10, y, barWidth - 20, barHeight);

        context.fillStyle = '#1f2937';
        context.font = '12px Vazirmatn, sans-serif';
        context.fillText(item.feature, x + 10, height - padding + 16);
        context.fillText(`دقت: ${(value * 100).toFixed(1)}%`, x + 10, y - 6);
        context.fillText(`هزینه: ${cost.toFixed(3)}s`, x + 10, y - 20);
    });
})();
