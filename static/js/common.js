document.addEventListener("DOMContentLoaded", function () {
    const headerElement = document.querySelector('header');

    fetch('header.html')
        .then(response => response.text())
        .then(data => {
            headerElement.innerHTML = data;
            updateActiveLink();
        });

    function updateActiveLink() {
        const currentPath = window.location.pathname;
        const menuItems = document.querySelectorAll('.menu-bar ul li');

        menuItems.forEach(item => {
            if (item.querySelector('a') && item.querySelector('a').getAttribute('href') === currentPath) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
    }
});