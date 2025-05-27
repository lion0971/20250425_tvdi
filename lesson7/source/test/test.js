document.addEventListener('DOMContentLoaded', function () {
    // 選取需要的 DOM 元素
    const prevArrow = document.querySelector('.prev-arrow');
    const nextArrow = document.querySelector('.next-arrow');
    const cardsWrapper = document.querySelector('.property-cards-wrapper');
    const dotsContainer = document.querySelector('.carousel-dots');
    const cards = document.querySelectorAll('.property-card');

    // 檢查元素是否存在
    if (!prevArrow || !nextArrow || !cardsWrapper || !dotsContainer || cards.length === 0) {
        console.warn('輪播元件未完整找到，JavaScript 功能可能受限。');
        return; // 如果找不到必要元素，則不執行後續程式碼
    }

    let cardWidth = cards[0].offsetWidth; // 取得單張卡片的寬度
    const gap = 30; // 卡片間的間距 (與 CSS 保持一致)
    let scrollAmount = cardWidth + gap; // 每次滾動的距離

    // 更新滾動距離的函數 (考慮 RWD)
    function updateScrollAmount() {
        cardWidth = cards[0].offsetWidth;
        scrollAmount = cardWidth + gap;
    }

    // 視窗大小改變時，重新計算滾動距離
    window.addEventListener('resize', updateScrollAmount);

    // --- 導航按鈕點擊事件 ---
    prevArrow.addEventListener('click', () => {
        // 向左滾動一個卡片的寬度 + 間距
        cardsWrapper.scrollBy({
            left: -scrollAmount,
            behavior: 'smooth' // 平滑滾動
        });
        updateDots(); // 更新分頁點
    });

    nextArrow.addEventListener('click', () => {
        // 向右滾動一個卡片的寬度 + 間距
        cardsWrapper.scrollBy({
            left: scrollAmount,
            behavior: 'smooth' // 平滑滾動
        });
        updateDots(); // 更新分頁點
    });

    // --- 分頁指示器 (基礎版本) ---
    // 計算需要多少個點 (假設每頁顯示 3 張)
    // 註：這是一個簡化的實現，真實世界中會更複雜
    const cardsPerPage = 3; // 假設大螢幕一次顯示 3 張
    const dotCount = Math.ceil(cards.length / cardsPerPage);

    // 清除現有的點並重新生成 (雖然 HTML 已有，但動態生成更靈活)
    dotsContainer.innerHTML = '';
    for (let i = 0; i < dotCount; i++) {
        const dot = document.createElement('span');
        dot.classList.add('dot');
        if (i === 0) {
            dot.classList.add('active'); // 預設第一個為 active
        }
        // 增加點擊跳轉功能 (可選)
        dot.addEventListener('click', () => {
            cardsWrapper.scrollTo({
                left: i * scrollAmount * (window.innerWidth >= 992 ? cardsPerPage : 1), // 根據螢幕寬度跳轉
                behavior: 'smooth'
            });
            updateDots();
        });
        dotsContainer.appendChild(dot);
    }

    const allDots = document.querySelectorAll('.carousel-dots .dot');

    // 更新分頁點的函數
    function updateDots() {
        // 延遲一點時間等待滾動完成再更新，否則可能不準確
        setTimeout(() => {
            updateScrollAmount(); // 確保滾動距離是最新的
            // 計算當前滾動到了第幾頁 (簡化)
            const currentPage = Math.round(cardsWrapper.scrollLeft / scrollAmount / (window.innerWidth >= 992 ? cardsPerPage : 1));

            allDots.forEach((dot, index) => {
                dot.classList.remove('active');
                if (index === currentPage) {
                    dot.classList.add('active');
                }
            });
        }, 350); // 延遲時間應略長於滾動時間
    }

    // 監聽滾動事件來更新分頁點
    cardsWrapper.addEventListener('scroll', updateDots);

    // 初始呼叫一次
    updateScrollAmount();
    updateDots();
});