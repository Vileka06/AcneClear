import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

const rootElement = document.getElementById("root");
if (!rootElement) {
  document.body.innerHTML = "<main style='font-family:Arial;padding:40px;color:#243b63'><h1>AcneClear AI</h1><p>App root element is missing. Please check index.html.</p></main>";
} else {
  try {
    createRoot(rootElement).render(<App />);
  } catch (error) {
    console.error("AcneClear render error:", error);
    rootElement.innerHTML = "<main style='font-family:Arial;padding:40px;color:#243b63'><h1>AcneClear AI</h1><p>The page could not render. Open the browser console and share the error.</p></main>";
  }
}
