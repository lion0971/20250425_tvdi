// base.js 中的範例功能（或你也可以直接貼在模板裡）
document.addEventListener('DOMContentLoaded', function () {
  const backToTopButton = document.getElementById("backToTop");

  // 捲動超過 300px 顯示按鈕
  window.addEventListener("scroll", function () {
    if (window.scrollY > 200) {
      backToTopButton.style.display = "flex";
    } else {
      backToTopButton.style.display = "none";
    }
  });

  // 點擊平滑滾動至頂端
  backToTopButton.addEventListener("click", function () {
    window.scrollTo({
      top: 0,
      behavior: "smooth"
    });
  });
});
