/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./*.html",      // include all HTML files in root folder
    "./js/**/*.js"   // if you have JS files using Tailwind classes
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
