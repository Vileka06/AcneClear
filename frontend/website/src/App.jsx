import React, { useState } from "react";

export default function App() {
  const [photoName, setPhotoName] = useState("");
  const [message, setMessage] = useState("");

  function onChoose(event) {
    const file = event.target.files && event.target.files[0];
    setPhotoName(file ? file.name : "");
    setMessage("");
  }

  return (
    <div className="page">
      <header className="header">
        <a className="brand" href="/"> <span className="brand-icon">✦</span><span>AcneClear <b>AI</b><small>SKIN WELLNESS STUDIO</small></span></a>
        <nav><a href="#features">Features</a><a href="#about">About</a></nav>
        <a className="signin" href="#upload">Get started</a>
      </header>

      <main>
        <section className="hero">
          <div className="hero-text">
            <div className="eyebrow">YOUR PERSONAL SKIN WELLNESS SPACE</div>
            <h1>Understand your skin.<br/><span>Care with confidence.</span></h1>
            <p>Explore skin analysis, personalised skincare education, balanced meal ideas and facial photo restoration in one thoughtful place.</p>
            <a className="button" href="#upload">Start with a photo <span>→</span></a>
            <div className="privacy">♧ Your photo is not stored by this starter website.</div>
          </div>
          <div className="visual">
            <div className="visual-circle"></div>
            <div className="portrait">
              <div className="portrait-hair"></div>
              <div className="portrait-face"><i className="eye left"></i><i className="eye right"></i><i className="nose"></i><i className="lips"></i></div>
              <div className="portrait-caption">✧ YOUR SKIN, YOUR JOURNEY</div>
            </div>
            <div className="floating">Personalised care, step by step</div>
          </div>
        </section>

        <section className="upload" id="upload">
          <div><div className="eyebrow">GET STARTED</div><h2>Begin with a skin photo</h2><p>Choose a clear facial photo in JPG, PNG or WEBP format. Image analysis will be connected in the next development stage.</p></div>
          <div className="upload-control">
            <label className="file-button">＋ <span>{photoName || "Choose a photo"}</span><input type="file" accept="image/jpeg,image/png,image/webp" onChange={onChoose}/></label>
            <button className="button" disabled={!photoName} onClick={() => setMessage("Your photo is selected. AI skin prediction is the next module we will connect.")}>Continue →</button>
            {message && <p className="message" role="status">{message}</p>}
          </div>
        </section>

        <section className="features" id="features">
          <div className="section-title"><div className="eyebrow">YOUR WELLNESS TOOLKIT</div><h2>Everything in one calm space</h2><p>Designed to bring your skin and wellness journey together.</p></div>
          <div className="cards">
            <article><div className="card-icon">◎</div><small>AI ANALYSIS</small><h3>Skin analysis</h3><p>Explore visible skin concerns with planned region-based facial assessment.</p></article>
            <article><div className="card-icon">✧</div><small>DAILY ROUTINE</small><h3>Beauty guidance</h3><p>Learn about morning and evening skincare routines and ingredients.</p></article>
            <article><div className="card-icon">♧</div><small>HEALTHY HABITS</small><h3>Diet planner</h3><p>Discover balanced meal ideas, recipes and food inspiration.</p></article>
            <article><div className="card-icon">✦</div><small>PHOTO RESTORATION</small><h3>Image enhancement</h3><p>Planned GFPGAN facial restoration with an original-versus-restored view.</p></article>
          </div>
        </section>

        <section className="extras" id="about">
          <article><b>文</b><h3>Four languages</h3><p>English, Kannada, Hindi and Telugu support planned.</p></article>
          <article><b>◷</b><h3>Gentle reminders</h3><p>Optional skincare, water, meal and progress reminders.</p></article>
          <article><b>▤</b><h3>Your progress</h3><p>Personal history and progress tracking after secure accounts are ready.</p></article>
        </section>
      </main>
      <footer><b>AcneClear AI</b><span>Educational wellness support. Not a medical diagnosis or a replacement for a dermatologist.</span></footer>
    </div>
  );
}
