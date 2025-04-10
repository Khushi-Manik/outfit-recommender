// pages/OutfitResultPage.tsx
import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './OutfitResultPage.css'; // Optional: Add styling here

const OutfitResultPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const outfit = location.state?.outfit;

  if (!outfit) {
    return (
      <div className="result-container">
        <p>No outfit data found. Please go back and try again.</p>
        <button onClick={() => navigate(-1)}>⬅️ Go Back</button>
      </div>
    );
  }

  return (
    <div className="result-container">
      <h2>🎉 Your Recommended Outfit</h2>
      <div className="outfit-card">
        <p><strong>Top:</strong> {outfit.Top}</p>
        <p><strong>Bottom:</strong> {outfit.Bottom}</p>
        <p><strong>Shoes:</strong> {outfit.Shoes}</p>
        <p><strong>Style:</strong> {outfit.StyleCategory}</p>
      </div>
      <button onClick={() => navigate('/recommendations')}>🔁 Try Another</button>
    </div>
  );
};

export default OutfitResultPage;
