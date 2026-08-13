import React, { useState } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { UploadCloud, CheckCircle, BrainCircuit, Activity, BarChart2 } from 'lucide-react';
import './Dashboard.css';

const API_URL = 'http://localhost:8000';

function Dashboard() {
  const [file, setFile] = useState(null);
  const [sessionData, setSessionData] = useState(null);
  const [targetColumn, setTargetColumn] = useState('');
  const [trainingState, setTrainingState] = useState('idle'); // idle, training, done, error
  const [trainResults, setTrainResults] = useState(null);
  
  // Prediction state
  const [testInput, setTestInput] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);

  const handleFileUpload = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    setFile(selectedFile);
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const response = await axios.post(`${API_URL}/upload`, formData);
      setSessionData(response.data);
      setTargetColumn('');
      setTrainResults(null);
      setTrainingState('idle');
    } catch (err) {
      console.error('Upload failed:', err);
      alert('File upload failed.');
    }
  };

  const handleTrain = async () => {
    if (!sessionData) return;
    setTrainingState('training');
    
    try {
      const response = await axios.post(`${API_URL}/train`, {
        session_id: sessionData.session_id,
        target_column: targetColumn
      });
      setTrainResults(response.data);
      setTrainingState('done');
      
      // Initialize test inputs
      const initialInputs = {};
      response.data.feature_cols.forEach(col => {
        initialInputs[col] = '';
      });
      setTestInput(initialInputs);
      
    } catch (err) {
      console.error('Training failed:', err);
      setTrainingState('error');
      alert('Training failed: ' + (err.response?.data?.detail || err.message));
    }
  };

  const handlePredict = async () => {
    setPredicting(true);
    try {
      // Clean inputs (convert strings to numbers where appropriate)
      const cleanedInput = {};
      Object.keys(testInput).forEach(k => {
        cleanedInput[k] = isNaN(testInput[k]) || testInput[k] === '' ? testInput[k] : Number(testInput[k]);
      });

      const response = await axios.post(`${API_URL}/predict`, {
        session_id: sessionData.session_id,
        features: cleanedInput
      });
      setPrediction(response.data);
    } catch (err) {
      console.error('Prediction failed:', err);
      alert('Prediction failed.');
    } finally {
      setPredicting(false);
    }
  };

  const handleInputChange = (e, col) => {
    setTestInput({ ...testInput, [col]: e.target.value });
  };

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="logo">
          <BrainCircuit size={28} /> InsightFlow AutoML
        </div>
      </header>

      <main className="dashboard-main">
        {/* Step 1: Upload */}
        <section className="dashboard-card">
          <h2>1. Upload Dataset</h2>
          <div className="upload-area">
            <input type="file" id="file-upload" accept=".csv" onChange={handleFileUpload} />
            <label htmlFor="file-upload" className="upload-label">
              <UploadCloud size={48} />
              <span>{file ? file.name : "Drag & Drop or Click to Upload CSV"}</span>
            </label>
          </div>
        </section>

        {/* Step 2: Configure & Train */}
        <AnimatePresence>
          {sessionData && (
            <motion.section 
              className="dashboard-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h2>2. Configure & Train</h2>
              
              <div className="data-preview">
                <h3>Data Preview (First 5 rows)</h3>
                <div className="table-responsive">
                  <table>
                    <thead>
                      <tr>
                        {sessionData.columns.map(col => <th key={col}>{col}</th>)}
                      </tr>
                    </thead>
                    <tbody>
                      {sessionData.preview_data.map((row, i) => (
                        <tr key={i}>
                          {sessionData.columns.map(col => <td key={col}>{row[col]}</td>)}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div className="config-form" style={{ marginTop: '2rem' }}>
                <label>Select Target Column (What do you want to predict?)</label>
                <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)}>
                  <option value="">-- Select Target (Leave empty for Unsupervised) --</option>
                  {sessionData.columns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
                
                <button 
                  className="btn-primary" 
                  onClick={handleTrain} 
                  disabled={trainingState === 'training'}
                >
                  {trainingState === 'training' ? (
                    <><Activity className="spin" size={18} /> Training Models...</>
                  ) : (
                    'Run AutoML Pipeline'
                  )}
                </button>
              </div>
            </motion.section>
          )}
        </AnimatePresence>

        {/* Step 3: Results */}
        <AnimatePresence>
          {trainingState === 'done' && trainResults && (
            <motion.section 
              className="dashboard-card results-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h2>3. Model Results: {trainResults.problem_type}</h2>
              <div className="metrics-grid">
                {Object.entries(trainResults.metrics).map(([key, val]) => {
                  if (typeof val === 'number') {
                    return (
                      <div key={key} className="metric-box">
                        <h4>{key.replace('_', ' ').toUpperCase()}</h4>
                        <p>{val.toFixed(4)}</p>
                      </div>
                    );
                  }
                  return null;
                })}
              </div>

              {trainResults.metrics.leaderboard && (
                <div className="leaderboard">
                  <h3>Model Leaderboard</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        {Object.keys(Object.values(trainResults.metrics.leaderboard)[0]).map(k => (
                          <th key={k}>{k.toUpperCase()}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(trainResults.metrics.leaderboard).map(([modelName, scores]) => (
                        <tr key={modelName} className={modelName === trainResults.metrics.best_model ? 'best-model-row' : ''}>
                          <td>
                            {modelName}
                            {modelName === trainResults.metrics.best_model && <span className="badge">Best</span>}
                          </td>
                          {Object.values(scores).map((score, i) => (
                            <td key={i}>{score.toFixed(4)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </motion.section>
          )}
        </AnimatePresence>

        {/* Step 4: Make Predictions */}
        <AnimatePresence>
          {trainingState === 'done' && trainResults && (
            <motion.section 
              className="dashboard-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <h2>4. Make a Prediction</h2>
              <p>Test the trained <strong>{trainResults.metrics.best_model}</strong> model in real-time.</p>
              
              <div className="prediction-form">
                <div className="inputs-grid">
                  {trainResults.feature_cols.map(col => {
                    const info = sessionData.column_info[col];
                    return (
                      <div key={col} className="input-group">
                        <label>{col}</label>
                        {info && info.type === 'categorical' && info.values && info.values.length > 0 ? (
                          <select 
                            value={testInput[col] !== undefined ? testInput[col] : ''} 
                            onChange={(e) => handleInputChange(e, col)}
                          >
                            <option value="">-- Select --</option>
                            {info.values.map(v => (
                              <option key={v} value={v}>{v}</option>
                            ))}
                          </select>
                        ) : (
                          <input 
                            type={info && info.type === 'numeric' ? 'number' : 'text'} 
                            value={testInput[col] !== undefined ? testInput[col] : ''} 
                            onChange={(e) => handleInputChange(e, col)}
                            placeholder={`Enter ${col}`}
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
                
                <button className="btn-primary" onClick={handlePredict} disabled={predicting}>
                  {predicting ? 'Predicting...' : 'Generate Prediction'}
                </button>
              </div>

              {prediction && (
                <motion.div 
                  className="prediction-result"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                >
                  <div className="pred-header">
                    <h3>Prediction Result: <span>{typeof prediction.prediction === 'number' ? prediction.prediction.toFixed(4) : prediction.prediction}</span></h3>
                    {prediction.confidence && (
                      <p>Confidence: {(prediction.confidence * 100).toFixed(2)}%</p>
                    )}
                  </div>
                  
                  {prediction.shap_values && Object.keys(prediction.shap_values).length > 0 && (
                    <div className="shap-explanation">
                      <h4>Model Explanation (SHAP Values)</h4>
                      <p className="text-muted">How much each feature pushed the prediction higher (green) or lower (red).</p>
                      
                      <div className="shap-bars">
                        {(() => {
                          const maxShapAbs = Math.max(...Object.values(prediction.shap_values).map(Math.abs));
                          return Object.entries(prediction.shap_values)
                            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                            .slice(0, 10) // show top 10
                            .map(([feature, val]) => {
                              const widthPercent = maxShapAbs === 0 ? 0 : (Math.abs(val) / maxShapAbs) * 50;
                              return (
                                <div key={feature} className="shap-bar-row">
                                  <span className="feature-name" title={feature}>{feature}</span>
                                  <div className="bar-container">
                                    <div 
                                      className={`bar ${val > 0 ? 'positive' : 'negative'}`}
                                      style={{ width: `${widthPercent}%` }}
                                    >
                                      <span className="val-text">{val > 0 ? '+' : ''}{val.toFixed(3)}</span>
                                    </div>
                                  </div>
                                </div>
                              );
                            });
                        })()}
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </motion.section>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}

export default Dashboard;
