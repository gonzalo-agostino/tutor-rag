/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/**/*.html", "./static/**/*.js"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        canvas: "#0F1115",
        surface: "#161920",
        elevated: "#1C2029",
        hairline: "#242936",
        ink: "#E6EAF2",
        muted: "#8B95A7",
        amberSoft: "#F59E0B",
        indigoSoft: "#818CF8",
      },
      fontFamily: {
        sans: ["Inter", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "sans-serif"],
        serif: ["Lora", "Georgia", "Cambria", "serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
