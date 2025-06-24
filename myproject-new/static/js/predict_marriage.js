button.onclick = function () {
    const error_msg = urlParams.get("error_msg") || "";
    const year = urlParams.get("year") || "";
    const income = urlParams.get("income") || "";
    const region = urlParams.get("region") || "";

    const newUrl = new URL(form.dataset.resetUrl, window.location.origin);
    if (error_msg) newUrl.searchParams.set("error_msg", error_msg);
    if (year) newUrl.searchParams.set("year", year);
    if (income) newUrl.searchParams.set("income", income);
    if (region) newUrl.searchParams.set("region", region);

    // 確保刪除 prediction 參數
    newUrl.searchParams.delete("prediction");

    window.location.href = newUrl.toString();
};
