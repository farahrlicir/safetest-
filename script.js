
 window.onload = function () {
            // Hide loader and display the content once the page is fully loaded
            document.getElementById('loader').style.display = 'none'; // Hide loader
            document.getElementById('content').style.display = 'block'; // Show content
};

const menuButton = document.getElementById('menu-button');
        const mobileMenu = document.getElementById('mobile-menu');

        menuButton.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });