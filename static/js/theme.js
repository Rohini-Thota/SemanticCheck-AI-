
(() => {
    const STORAGE_KEY = "semantiai-theme";

    function applyTheme(theme) {
        const selectedTheme = theme === "dark" ? "dark" : "light";

        document.documentElement.setAttribute("data-theme", selectedTheme);

        const button = document.getElementById("themeToggle");
        const icon = document.getElementById("themeIcon");

        if (icon) {
            icon.textContent = selectedTheme === "dark" ? "☀️" : "🌙";
        }

        if (button) {
            const label = selectedTheme === "dark"
                ? "Switch to light mode"
                : "Switch to dark mode";

            button.setAttribute("aria-label", label);
            button.title = label;
        }
    }

    function toggleTheme() {
        const currentTheme =
            document.documentElement.getAttribute("data-theme") || "light";

        const nextTheme = currentTheme === "dark" ? "light" : "dark";

        applyTheme(nextTheme);

        try {
            localStorage.setItem(STORAGE_KEY, nextTheme);
        } catch (error) {
            console.warn("Could not save theme preference:", error);
        }
    }

    let savedTheme = "light";

    try {
        savedTheme = localStorage.getItem(STORAGE_KEY) || "light";
    } catch (error) {
        console.warn("Could not read theme preference:", error);
    }

    applyTheme(savedTheme);

    document.addEventListener("DOMContentLoaded", () => {
        const button = document.getElementById("themeToggle");

        if (button && button.dataset.themeReady !== "true") {
            button.dataset.themeReady = "true";
            button.addEventListener("click", toggleTheme);
        }
    });
})();
