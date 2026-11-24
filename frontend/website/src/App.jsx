import React, { useEffect, useState } from "react";

const API = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const MAX_BYTES = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp"];

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
  }, [preview]);

  function choosePhoto(event) {
    const selected = event.target.files && event.target.files[0];
    setResult("");
    setError("");
    setNotice("");
    if (!selected) return;
    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setFile(null);
      setPreview("");
      setError("Choose a JPG, PNG or WEBP image.");
      return;
    }
    if (selected.size > MAX_BYTES) {
      setFile(null);
      setPreview("");
      setError("The image must be 10 MB or smaller.");
      return;
    }
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  }

  async function enhancePhoto() {
    if (!file) return;
    setBusy(true);
    setError("");
    setNotice("");
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch(API + "/api/enhance", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Photo enhancement failed.");
      setResult("data:image/png;base64," + data.image_base64);
      setNotice(data.note || "Your enhanced photo is ready.");
    } catch (err) {
      setError(err.message || "Could not connect to the enhancement service. Check that the backend is running.");
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setFile(null);
    setPreview("");
    setResult("");
    setError("");
    setNotice("");
    const input = document.getElementById("photo-input");
    if (input) input.value = "";
  }

  function downloadResult() {
    if (!result) return;
    const anchor = document.createElement("a");
    anchor.href = result;
    anchor.download = "acneclear_enhanced.png";
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  }

  return (
    <div className="page">
      <header className="header">
        <a className="brand" href="/" aria-label="AcneClear AI home">
          <span className="brand-mark">✳</span>
          <span><strong>AcneClear AI</strong><small>PHOTO ENHANCEMENT</small></span>
        </a>
        <span className="privacy-tag">Private photo processing</span>
      </header>
      <main>
        <section className="hero">
          <p className="eyebrow">PHOTO RESTORATION, MADE SIMPLE</p>
          <h1>Enhance your photo.<br /><span>Keep your natural details.</span></h1>
          <p className="hero-copy">Upload a clear face photo to restore facial details and apply gentle texture smoothing. Preview the result and download your enhanced image.</p>
        </section>
        <section className="workspace" aria-labelledby="upload-title">
          <div className="workspace-head">
            <div><p className="eyebrow">ONE SIMPLE STEP</p><h2 id="upload-title">Upload your photo</h2><p className="muted">JPG, PNG or WEBP. Maximum file size: 10 MB.</p></div>
            <span className="step-number">01</span>
          </div>
          {!preview ? (
            <label className="dropzone" htmlFor="photo-input">
              <span className="upload-icon">↑</span><strong>Choose a photo</strong><span>Browse files from your device</span>
              <input id="photo-input" type="file" accept="image/jpeg,image/png,image/webp" onChange={choosePhoto} />
            </label>
          ) : (
            <div className="selected-photo">
              <img className="preview-image" src={preview} alt="Selected original photo" />
              <div className="photo-meta">
                <div><strong>{file?.name}</strong><span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span></div>
                <button className="text-button" onClick={reset} type="button">Remove photo</button>
              </div>
            </div>
          )}
          {error && <p className="feedback error" role="alert">{error}</p>}
          {notice && <p className="feedback success" role="status">{notice}</p>}
          <div className="actions">
            <button className="primary-button" disabled={!file || busy} onClick={enhancePhoto}>
              {busy ? <><span className="spinner" /> Enhancing photo...</> : "Enhance photo"}
            </button>
            {result && <button className="secondary-button" onClick={downloadResult}>Download PNG</button>}
          </div>
        </section>
        {result && (
          <section className="results" aria-labelledby="result-title">
            <div className="workspace-head">
              <div><p className="eyebrow">YOUR RESULT</p><h2 id="result-title">Original and enhanced</h2></div>
              <span className="step-number">02</span>
            </div>
            <div className="comparison">
              <figure><img src={preview} alt="Original photo" /><figcaption>Original</figcaption></figure>
              <figure><img src={result} alt="Enhanced photo" /><figcaption>Enhanced</figcaption></figure>
            </div>
            <button className="primary-button download-wide" onClick={downloadResult}>Download enhanced photo</button>
          </section>
        )}
      </main>
      <footer>
        <strong>AcneClear AI</strong>
        <p>Your photo is processed in memory and is not stored by this application. GFPGAN is a general face-restoration model, not an acne-specific removal or medical diagnosis system. Results vary by image quality and face visibility.</p>
      </footer>
    </div>
  );
}
