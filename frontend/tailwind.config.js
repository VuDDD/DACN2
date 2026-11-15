/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        medical: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
        lesion: {
          ma: '#ef4444', // Microaneurysms - Red
          he: '#f97316', // Haemorrhages - Orange
          ex: '#eab308', // Hard Exudates - Yellow
          se: '#22c55e', // Soft Exudates - Green
          od: '#3b82f6', // Optic Disc - Blue
          vessel: '#a855f7', // Blood Vessels - Purple
        }
      }
    },
  },
  plugins: [],
}
