document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('startBtn');

    // Add a simple entrance animation for the text lines
    const textLines = document.querySelectorAll('.text-line');

    textLines.forEach((line, index) => {
        line.style.opacity = '0';
        line.style.transform = 'translateY(20px)';
        line.style.transition = `all 0.6s ease ${index * 0.1}s`;

        // Trigger reflow
        setTimeout(() => {
            line.style.opacity = '1';
            line.style.transform = 'translateY(0)';
        }, 100);
    });

    // Button Interaction
    startBtn.addEventListener('click', () => {
        console.log('Start Mission clicked');
        // Add a "pressed" effect visually if desired beyond CSS active
        startBtn.style.transform = 'scale(0.95)';
        setTimeout(() => {
            startBtn.style.transform = 'scale(1)';
            alert('Welcome to Kingdom Partners! Let\'s begin setting up your mission profile.');
        }, 150);
    });
});
