import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        display: ["Avenir Next", "Segoe UI", "Trebuchet MS", "sans-serif"],
        body: ["Avenir Next", "Segoe UI", "Helvetica Neue", "sans-serif"],
      },
      boxShadow: {
        panel: "0 24px 80px rgba(15, 23, 42, 0.18)",
      },
    },
  },
  plugins: [],
} satisfies Config;
