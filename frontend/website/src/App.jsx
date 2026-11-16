import { useEffect, useState } from "react";
import { Activity, Bell, Camera, ChevronRight, CircleUserRound, ClipboardList, Heart, Languages, Leaf, ShieldCheck, Sparkles, Utensils, WandSparkles } from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [file, setFile] = useState(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    fetch(`${API}/api/health`)
      .then((r) => { if (!r.ok) throw new Error(); return r.json(); })
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
      const response = await fetch(`${API}/api/uploads/validate`, { method: "POST", body });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Upload validation failed.");
      setUploadMessage(`Image accepted: ${result.width} × ${result.height}. Model prediction will be added in the next milestone.`);
    } catch (error) {
      setUploadMessage(`${error.message} Check that the FastAPI backend is running.`);
    } finally {
      setBusy(false);
    }
  }

  const modules = [
    { icon: Camera, title: "Skin analysis", text: "Upload a facial photo for future region-based skin assessment.", tag: "Next milestone" },
    { icon: Sparkles, title: "Beauty guidance", text: "Personalised morning and evening skincare education.", tag: "Planned" },
    { icon: Utensils, title: "Diet planner", text: "Balanced meal ideas with recipes and food visuals.", tag: "Planned" },
    { icon: WandSparkles, title: "Image enhancement", text: "GFPGAN restoration with an original-versus-restored comparison.", tag: "Existing model integration" },
  ];

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#"><span className="brand-mark"><Sparkles size={20}/></span><span>AcneClear<span className="brand-ai"> AI</span><small>SKIN WELLNESS STUDIO</small></span></a>
        <nav><a href="#features">Features</a><a href="#how">How it works</a></nav>
        <button className="profile-button"><CircleUserRound size={17}/> Sign in</button>
      </header>

      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow"><Heart size={14}/> YOUR PERSONAL SKIN WELLNESS SPACE</span>
          <h1>Understand your skin.<br/><em>Care with confidence.</em></h1>
          <p>One thoughtful place for skin analysis, personalised skincare education, balanced meal ideas and photo restoration.</p>
          <a className="primary-button" href="#upload">Start with a photo <ChevronRight size={17}/></a>
          <div className="trust-line"><ShieldCheck size={16}/> Your photo is validated locally by the API and is not saved in this foundation build.</div>
        </div>
        <div className="hero-art">
          <div className="orb orb-one"></div><div className="orb orb-two"></div>
          <div className="face-card"><div className="face-illustration"><span className="hair"></span><span className="face-shape"><i className="eye eye-left"></i><i className="eye eye-right"></i><i className="nose"></i><i className="mouth"></i></span></div><div className="scan-label"><Activity size={15}/> Skin analysis workspace</div></div>
          <div className="floating-note"><Sparkles size={16}/> Personalised care, step by step</div>
        </div>
      </section>

      <section className="upload-section" id="upload">
        <div><span className="eyebrow">GET STARTED</span><h2>Start with your skin photo</h2><p>Choose a clear, front-facing image in JPG, PNG or WEBP format, up to 10 MB.</p></div>
        <form className="upload-box" onSubmit={validateImage}>
          <label className="file-picker"><Camera size={22}/><span>{file ? file.name : "Choose a photo"}</span><input type="file" accept="image/jpeg,image/png,image/webp" onChange={(e) => {setFile(e.target.files?.[0] || null);setUploadMessage("");}}/></label>
          <button className="primary-button" disabled={!file || busy}>{busy ? "Checking..." : "Validate photo"} <ChevronRight size={17}/></button>
          {uploadMessage && <p className="upload-message" role="status">{uploadMessage}</p>}
        </form>
      </section>

      <section id="features" className="features-section">
        <div className="section-heading"><span className="eyebrow">YOUR WELLNESS TOOLKIT</span><h2>Everything in one calm space</h2><p>These modules will be connected one by one as the AI services are implemented and tested.</p></div>
        <div className="feature-grid">{modules.map(({icon:Icon,title,text,tag})=><article className="feature-card" key={title}><span className="feature-icon"><Icon size={21}/></span><span className="feature-tag">{tag}</span><h3>{title}</h3><p>{text}</p><ChevronRight className="feature-arrow" size={17}/></article>)}</div>
      </section>

      <section className="bottom-grid" id="how">
        <article><Languages/><h3>Your language, your comfort</h3><p>English, Kannada, Hindi and Telugu support is planned across the website and mobile experience.</p></article>
        <article><Bell/><h3>Gentle reminders</h3><p>Optional skincare, hydration, meal and weekly progress reminders will be available with your permission.</p></article>
        <article><ClipboardList/><h3>Your progress</h3><p>A private history and progress view will be introduced after secure accounts and storage are ready.</p></article>
      </section>

      <footer><span>AcneClear AI</span><span className={`api-status ${apiStatus}`}><i/> API {apiStatus}</span><p>Educational wellness support only. This application does not diagnose skin disease or replace a dermatologist.</p></footer>
    </main>
  );
}
