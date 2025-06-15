document.addEventListener('DOMContentLoaded', function () {
    const menuToggle = document.querySelector('.menu-toggle');
    const mainNavigation = document.querySelector('.main-navigation');

    if (menuToggle && mainNavigation) {
        menuToggle.addEventListener('click', function () {
            mainNavigation.classList.toggle('toggled'); // 切換 .toggled class
            const isExpanded = mainNavigation.classList.contains('toggled');
            menuToggle.setAttribute('aria-expanded', isExpanded);
        });
    }
});

const backToTopButton = document.getElementById("backToTop");

  // 當滾動超過一定高度時顯示按鈕
  window.onscroll = function () {
    if (document.body.scrollTop > 300 || document.documentElement.scrollTop > 300) {
      backToTopButton.style.display = "block";
    } else {
      backToTopButton.style.display = "none";
    }
  };

   // 點擊按鈕時回到頂部
  backToTopButton.addEventListener("click", function () {
    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  });