/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{js,jsx,ts,tsx}', './index.html'],
  theme: {
    extend: {
      colors: {
        primary: '#3B82F6',
        success: '#10B981',
        danger: '#EF4444',
      }
    }
  },
  plugins: [],
}
