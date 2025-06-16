
  // 自動顯示（可選）
//  window.addEventListener('load', () => {
//     const popup = document.getElementById('popup');
//     const image = document.querySelector('.popup-image img');

//     // 顯示彈出視窗
//     popup.style.display = 'flex';

//     // 加入除錯
//     console.log("圖片元素是：", image);

//     // 點圖片 -> 關閉 popup（加 null 檢查）
//     if (image) {
//       image.addEventListener('click', () => {
//         popup.style.display = 'none';
//       });
//     }

//     // 點背景 -> 關閉 popup（只在點到 .overlay 本身時觸發）
//     popup.addEventListener('click', (event) => {
//       if (event.target === popup) {
//         popup.style.display = 'none';
//       }
//     });
//   });

window.addEventListener('load', () => {
  const popup = document.getElementById('popup');
  const closeBtn = document.getElementById('closePopup');

  // 檢查 sessionStorage 是否有彈過視窗
  if (!JSON.parse(sessionStorage.getItem('popupShown'))) {
    popup.style.display = 'flex';  // 顯示彈窗
    sessionStorage.setItem('popupShown', JSON.stringify(true));  // 標記已經看過
  }

  // 關閉按鈕點擊事件
  closeBtn.addEventListener('click', () => {
    popup.style.display = 'none';
  });

  // 點彈窗以外區域也關閉（點到 overlay 本身）
  popup.addEventListener('click', (e) => {
    if (e.target === popup) {
      popup.style.display = 'none';
    }
  });
});


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