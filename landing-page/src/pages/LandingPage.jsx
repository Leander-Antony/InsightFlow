import React from 'react';
import { motion } from 'framer-motion';
import { BrainCircuit, Search, Database, BarChart3, ArrowRight, Play, Server, Layers } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

function LandingPage() {
  const navigate = useNavigate();

  const handleDemoClick = () => {
    navigate('/app');
  };

  const staggerContainer = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.15
      }
    }
  };

  const fadeUp = {
    hidden: { opacity: 0, y: 30 },
    show: { opacity: 1, y: 0, transition: { duration: 0.6, ease: "easeOut" } }
  };

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="logo">
          <BrainCircuit size={28} />
          InsightFlow
        </div>
      </nav>

      <main>
        <section className="hero">
          <motion.div 
            className="hero-content"
            initial="hidden"
            animate="show"
            variants={staggerContainer}
          >
            <motion.h1 variants={fadeUp}>
              AutoML <br />
              <span className="hero-accent-text">Made Simple.</span>
            </motion.h1>
            <motion.p variants={fadeUp}>
              Automatically detect your machine learning problem type and build a complete pipeline from preprocessing to model evaluation in seconds.
            </motion.p>
            <motion.div className="cta-group" variants={fadeUp}>
              <button className="btn-primary" onClick={handleDemoClick}>
                Try the Demo 
                <ArrowRight size={18} />
              </button>
              <a href="#how-it-works" className="btn-secondary">
                Learn More
              </a>
            </motion.div>
          </motion.div>
          
          <motion.div 
            className="hero-visual"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <div className="visual-header">
              <div className="dot" style={{ background: '#ff5f56' }}></div>
              <div className="dot" style={{ background: '#ffbd2e' }}></div>
              <div className="dot" style={{ background: '#27c93f' }}></div>
            </div>
            <div className="visual-body">
              <motion.div 
                className="pipeline-node"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
              >
                <Database size={24} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Data Preprocessing</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Cleaning & Encoding...</div>
                </div>
                <div className="pipeline-connector"></div>
              </motion.div>
              
              <motion.div 
                className="pipeline-node"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.9 }}
              >
                <Layers size={24} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Model Selection</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Training 5 algorithms...</div>
                </div>
                <div className="pipeline-connector"></div>
              </motion.div>

              <motion.div 
                className="pipeline-node"
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 1.2 }}
              >
                <BarChart3 size={24} />
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Evaluation</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Best model: XGBoost (94% acc)</div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </section>

        <section id="what-it-is" className="features-section">
          <motion.h2 
            className="section-title"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
          >
            Why InsightFlow?
          </motion.h2>
          <motion.div 
            className="features-grid"
            variants={staggerContainer}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div className="feature-card" variants={fadeUp}>
              <div className="feature-icon-wrapper">
                <Search size={24} />
              </div>
              <h3>Auto-Detection</h3>
              <p>Upload any tabular dataset and we'll automatically detect if it's Classification, Regression, Time Series, or Unsupervised.</p>
            </motion.div>
            <motion.div className="feature-card" variants={fadeUp}>
              <div className="feature-icon-wrapper">
                <BrainCircuit size={24} />
              </div>
              <h3>Intelligent Preprocessing</h3>
              <p>Handles missing values, categorical encoding, and feature scaling automatically based on your specific data types.</p>
            </motion.div>
            <motion.div className="feature-card" variants={fadeUp}>
              <div className="feature-icon-wrapper">
                <BarChart3 size={24} />
              </div>
              <h3>Interactive EDA</h3>
              <p>Instantly generate correlation matrices, statistical summaries, and insightful visualizations before training.</p>
            </motion.div>
            <motion.div className="feature-card" variants={fadeUp}>
              <div className="feature-icon-wrapper">
                <Server size={24} />
              </div>
              <h3>Model Selection</h3>
              <p>Trains and evaluates suitable algorithms for your problem type, displaying key performance metrics clearly.</p>
            </motion.div>
          </motion.div>
        </section>

        <section id="how-it-works" className="how-to-use">
          <motion.h2 
            className="section-title"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
          >
            How It Works
          </motion.h2>
          <motion.div 
            className="steps"
            variants={staggerContainer}
            initial="hidden"
            whileInView="show"
            viewport={{ once: true, margin: "-100px" }}
          >
            <motion.div className="step" variants={fadeUp}>
              <div className="step-number">1</div>
              <div className="step-content">
                <h3>Upload Your Data</h3>
                <p>Provide a CSV file. InsightFlow reads the structure and prepares it for analysis.</p>
              </div>
            </motion.div>
            <motion.div className="step" variants={fadeUp}>
              <div className="step-number">2</div>
              <div className="step-content">
                <h3>Select Target Column</h3>
                <p>Choose the column you want to predict, or leave it blank for unsupervised clustering.</p>
              </div>
            </motion.div>
            <motion.div className="step" variants={fadeUp}>
              <div className="step-number">3</div>
              <div className="step-content">
                <h3>Let AI Do The Rest</h3>
                <p>InsightFlow cleans the data, selects features, trains models, and presents the results interactively.</p>
              </div>
            </motion.div>
          </motion.div>
        </section>
      </main>

      <footer>
        <p>&copy; {new Date().getFullYear()} InsightFlow. Powered by React & FastAPI.</p>
      </footer>
    </div>
  );
}

export default LandingPage;
