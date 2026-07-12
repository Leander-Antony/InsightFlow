import React from 'react';
import './App.css';

function App() {
  const handleDemoClick = () => {
    window.location.href = 'http://localhost:8501';
  };

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="logo">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
            <line x1="12" y1="22.08" x2="12" y2="12"></line>
          </svg>
          InsightFlow
        </div>
      </nav>

      <main>
        <section className="hero">
          <div className="hero-content">
            <h1>
              AutoML <br />Made Simple.
            </h1>
            <p>
              Automatically detect your machine learning problem type and build a complete pipeline from preprocessing to model evaluation in seconds.
            </p>
            <div className="cta-group">
              <button className="btn-primary" onClick={handleDemoClick}>
                Try the Demo 
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="5" y1="12" x2="19" y2="12"></line>
                  <polyline points="12 5 19 12 12 19"></polyline>
                </svg>
              </button>
              <a href="#how-it-works" className="btn-secondary">
                Learn More
              </a>
            </div>
          </div>
          
          <div className="hero-visual">
            <div className="visual-header">
              <div className="dot" style={{ background: '#ff5f56' }}></div>
              <div className="dot" style={{ background: '#ffbd2e' }}></div>
              <div className="dot" style={{ background: '#27c93f' }}></div>
            </div>
            <div className="visual-body">
              <div className="visual-line"></div>
              <div className="visual-line medium"></div>
              <div className="visual-line short"></div>
              <br />
              <div className="visual-line medium"></div>
              <div className="visual-line"></div>
              <div className="visual-line short"></div>
            </div>
          </div>
        </section>

        <section id="what-it-is" className="features-section">
          <h2 className="section-title">Why InsightFlow?</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8"></circle>
                  <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                </svg>
              </div>
              <h3>Auto-Detection</h3>
              <p>Upload any tabular dataset and we'll automatically detect if it's Classification, Regression, Time Series, or Unsupervised.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                  <polyline points="7.5 4.21 12 6.81 16.5 4.21"></polyline>
                  <polyline points="7.5 19.79 7.5 14.6 3 12"></polyline>
                  <polyline points="21 12 16.5 14.6 16.5 19.79"></polyline>
                  <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                  <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
              </div>
              <h3>Intelligent Preprocessing</h3>
              <p>Handles missing values, categorical encoding, and feature scaling automatically based on your specific data types.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <line x1="18" y1="20" x2="18" y2="10"></line>
                  <line x1="12" y1="20" x2="12" y2="4"></line>
                  <line x1="6" y1="20" x2="6" y2="14"></line>
                </svg>
              </div>
              <h3>Interactive EDA</h3>
              <p>Instantly generate correlation matrices, statistical summaries, and insightful visualizations before training.</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2v20"></path>
                  <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path>
                </svg>
              </div>
              <h3>Model Selection</h3>
              <p>Trains and evaluates suitable algorithms for your problem type, displaying key performance metrics clearly.</p>
            </div>
          </div>
        </section>

        <section id="how-it-works" className="how-to-use">
          <h2 className="section-title">How It Works</h2>
          <div className="steps">
            <div className="step">
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Upload Your Data</h3>
                <p>Provide a CSV file. InsightFlow reads the structure and prepares it for analysis.</p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>Select Target Column</h3>
                <p>Choose the column you want to predict, or leave it blank for unsupervised clustering.</p>
              </div>
            </div>
            <div className="step">
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Let AI Do The Rest</h3>
                <p>InsightFlow cleans the data, selects features, trains models, and presents the results interactively.</p>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <p>&copy; {new Date().getFullYear()} InsightFlow. Powered by Streamlit & React.</p>
      </footer>
    </div>
  );
}

export default App;
