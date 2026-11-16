import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function Symbol({ children }) {
  return <span className="symbol" aria-hidden="true">{children}</span>;
}

export default function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [file, setFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then((response) => {
        if (!response.ok) throw new Error("API unavailable");
        return response.json();
      })
      .then(() => setApiStatus("online"))
      .catch(() => setApiStatus("offline"));
  }, []);

  async function validateImage(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setUploadMessage("");
    const body = new FormData();
    body.append("file", file);
    try {
      const response = await fetch(`${API}/api/uploads/validate`, {
        method: "POST",
        body,
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Upload validation failed.");
      setUploadMessage(`Photo accepted: ${result.width} × ${result.height}. AI prediction will be connected in the next milestone.`);
    } catch (error) {
      setUploadMessage(`${error.message} Please check that the backend is running.`);
    } finally {
      setBusy(false);
    }
  }

  const modules = [
    { symbol: "◎", title: "Skin analysis", text: "Upload a facial photo for region-based skin assessment.", tag: "Coming next" },
    { symbol: "✧", title: "Beauty guidance", text: "Personalised morning and evening skincare education.", tag: "Planned" },
    { symbol: "♧", title: "Diet planner", text: "Balanced meal ideas, recipes and food illustrations.", tag: "Planned" },
    { symbol: "✦", title: "Image enhancement", text: "GFPGAN face restoration and an original-versus-restored comparison.", tag: "Model integration" },
  ];

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#"><span className="brand-mark">✧</span><span>AcneClear<span className="brand-ai"> AI</span><small>SKIN WELLNESS STUDIO</small></span></a>
        <nav><a href="#features">Features</a><a href="#how">How it works</a></nav>
        <a className="profile-button" href="#account">Sign in</a>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">✧ YOUR PERSONAL SKIN WELLNESS SPACE</span>
          <h1>Understand your skin.<br/><em>Care with confidence.</em></h1>
          <p>One thoughtful place for skin analysis, personalised skincare education, balanced meal ideas and photo restoration.</p>
          <a className="primary-button" href="#upload">Start with a photo <span>→</span></a>
          <div className="trust-line">✓ Your photo is checked by the API and is not saved in this foundation build.</div>
        </div>
        <div className="hero-art">
          <div className="orb orb-one"></div><div className="orb orb-two"></div>
          <div className="face-card"><div className="face-illustration"><span className="hair"></span><span className="face-shape"><i className="eye eye-left"></i><i className="eye eye-right"></i><i className="nose"></i><i className="mouth"></i></span></div><div className="scan-label">◎ Skin analysis workspace</div></div>
          <div className="floating-note">✧ Personalised care, step by step</div>
        </div>
      </section>

      <section className="upload-section" id="upload">
        <div><span className="eyebrow">GET STARTED</span><h2>Start with your skin photo</h2><p>Choose a clear, front-facing image in JPG, PNG or WEBP format, up to 10 MB.</p></div>
        <form className="upload-box" onSubmit={validateImage}>
          <label className="file-picker"><span>＋</span><span>{file ? file.name : "Choose a photo"}</span><input type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => {setFile(event.target.files?.[0] || null);setUploadMessage("");}}/></label>
          <button className="primary-button" disabled={!file || busy}>{busy ? "Checking..." : "Validate photo"} <span>→</span></button>
          {uploadMessage && <p className="upload-message" role="status">{uploadMessage}</p>}
        </form>
      </section>

      <section id="features" className="features-section">
        <div className="section-heading"><span className="eyebrow">YOUR WELLNESS TOOLKIT</span><h2>Everything in one calm space</h2><p>These modules will be connected one by one as their AI services are implemented and tested.</p></div>
        <div className="feature-grid">{modules.map((item) => <article className="feature-card" key={item.title}><span className="feature-icon">{item.symbol}</span><span className="feature-tag">{item.tag}</span><h3>{item.title}</h3><p>{item.text}</p><span className="feature-arrow">→</span></article>)}</div>
      </section>

      <section className="bottom-grid" id="how">
        <article><Symbol>文</Symbol><h3>Your language, your comfort</h3><p>English, Kannada, Hindi and Telugu support is planned across the website and mobile experience.</p></article>
        <article><Symbol>◷</Symbol><h3>Gentle reminders</h3><p>Optional skincare, hydration, meal and weekly progress reminders will be available with your permission.</p></article>
        <article><Symbol>▤</Symbol><h3>Your progress</h3><p>A private history and progress view will be introduced after secure accounts and storage are ready.</p></article>
      </section>

      <footer><span>AcneClear AI</span><span className={`api-status ${apiStatus}`}><i/> API {apiStatus}</span><p>Educational wellness support only. This application does not diagnose skin disease or replace a dermatologist.</p></footer>
    </main>
  );
}
