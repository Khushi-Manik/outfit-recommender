import React, { useState } from 'react';
import './BodyTypeQuiz.css';
import placeholderShoulder from '@/assets/placeholder_shoulder.jpg';
import placeholderBust from '@/assets/placeholder_bust.jpg';
import placeholderWaist from '@/assets/placeholder_waist.jpg';
import placeholderHips from '@/assets/placeholder_hips.jpg';
import placeholderBelly from '@/assets/placeholder_belly.jpg'; // Add this image
import placeholderHeight from '@/assets/placeholder_height.jpg'; // Add this image

interface Measurements {
  shoulder: number | null;
  bust: number | null;
  waist: number | null;
  hips: number | null;
  belly: number | null;
  height: number | null;
}

interface MeasurementPageProps {
  title: string;
  instruction: string;
  imageSrc: string;
  measurementLabel: string;
  measurementName: keyof Measurements;
  value: number | null;
  onChange: (name: keyof Measurements, value: number | null) => void;
  onNext?: () => void;
  onPrevious?: () => void;
  onSubmit?: () => void;
}

const MeasurementPage: React.FC<MeasurementPageProps> = ({
  title,
  instruction,
  imageSrc,
  measurementLabel,
  measurementName,
  value,
  onChange,
  onNext,
  onPrevious,
  onSubmit,
}) => {
  return (
    <div className="quiz-page">
      <h3>{title}</h3>
      <p className="instruction">{instruction}</p>
      <div className="measurement-section">
        <img src={imageSrc} alt={`How to measure ${measurementName}`} className="measurement-image" />
        <div className="input-group">
          <label htmlFor={`${measurementName}-measurement`}>{measurementLabel}</label>
          <input
            type="number"
            id={`${measurementName}-measurement`}
            placeholder="Enter measurement"
            value={value === null ? '' : value}
            onChange={(e) => onChange(measurementName, e.target.value === '' ? null : parseInt(e.target.value, 10))}
          />
        </div>
      </div>
      {onPrevious && <button className="prev-btn" onClick={onPrevious}>Previous</button>}
      {onNext && <button className="next-btn" onClick={onNext}>Next</button>}
      {onSubmit && <button className="submit-btn" onClick={onSubmit}>Submit</button>}
    </div>
  );
};

interface ResultsPageProps {
  predictedType: string | null;
}

const ResultsPage: React.FC<ResultsPageProps> = ({ predictedType }) => {
  return (
    <div className="results-page">
      <h3>Your Body Type</h3>
      {predictedType ? (
        <p id="body-type-result">{predictedType}</p>
      ) : (
        <p>Processing...</p>
      )}
      {predictedType && <button className="view-info-btn">View Full Info</button>}
    </div>
  );
};

const BodyTypeQuiz: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<
    'shoulder' | 'bust' | 'waist' | 'hips' | 'belly' | 'height' | 'results'
  >('shoulder');

  const [measurements, setMeasurements] = useState<Measurements>({
    shoulder: null,
    bust: null,
    waist: null,
    hips: null,
    belly: null,
    height: null,
  });

  const [predictedBodyType, setPredictedBodyType] = useState<string | null>(null);

  const handleNext = (page: typeof currentPage) => setCurrentPage(page);
  const handlePrevious = (page: typeof currentPage) => setCurrentPage(page);

  const handleMeasurementChange = (name: keyof Measurements, value: number | null) => {
    setMeasurements({ ...measurements, [name]: value });
  };

  const handleSubmit = async () => {
    console.log('Measurements submitted:', measurements);
    setCurrentPage('results');
    setPredictedBodyType(null); // Show "Processing..." while waiting

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(measurements),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setPredictedBodyType(data.body_type || 'Unknown');
    } catch (error) {
      console.error('Error predicting body type:', error);
      setPredictedBodyType('Error predicting body type. Please try again.');
    }
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'shoulder':
        return (
          <MeasurementPage
            title="Measure Your Shoulders"
            instruction="Measure the distance from the tip of one shoulder bone to the tip of the other, across the back."
            imageSrc={placeholderShoulder}
            measurementLabel="Shoulder Measurement (in cm):"
            measurementName="shoulder"
            value={measurements.shoulder}
            onChange={handleMeasurementChange}
            onNext={() => handleNext('bust')}
          />
        );
      case 'bust':
        return (
          <MeasurementPage
            title="Measure Your Bust"
            instruction="Measure around the fullest part of your bust, keeping the tape measure level."
            imageSrc={placeholderBust}
            measurementLabel="Bust Measurement (in cm):"
            measurementName="bust"
            value={measurements.bust}
            onChange={handleMeasurementChange}
            onPrevious={() => handlePrevious('shoulder')}
            onNext={() => handleNext('waist')}
          />
        );
      case 'waist':
        return (
          <MeasurementPage
            title="Measure Your Waist"
            instruction="Measure around the narrowest part of your natural waistline."
            imageSrc={placeholderWaist}
            measurementLabel="Waist Measurement (in cm):"
            measurementName="waist"
            value={measurements.waist}
            onChange={handleMeasurementChange}
            onPrevious={() => handlePrevious('bust')}
            onNext={() => handleNext('hips')}
          />
        );
      case 'hips':
        return (
          <MeasurementPage
            title="Measure Your Hips"
            instruction="Measure around the fullest part of your hips and buttocks, keeping the tape measure level."
            imageSrc={placeholderHips}
            measurementLabel="Hip Measurement (in cm):"
            measurementName="hips"
            value={measurements.hips}
            onChange={handleMeasurementChange}
            onPrevious={() => handlePrevious('waist')}
            onNext={() => handleNext('belly')}
          />
        );
      case 'belly':
        return (
          <MeasurementPage
            title="Measure Your Belly"
            instruction="Measure around your belly at the widest point (typically around the navel)."
            imageSrc={placeholderBelly}
            measurementLabel="Belly Measurement (in cm):"
            measurementName="belly"
            value={measurements.belly}
            onChange={handleMeasurementChange}
            onPrevious={() => handlePrevious('hips')}
            onNext={() => handleNext('height')}
          />
        );
      case 'height':
        return (
          <MeasurementPage
            title="Measure Your Height"
            instruction="Stand straight against a wall and measure from the floor to the top of your head."
            imageSrc={placeholderHeight}
            measurementLabel="Height (in cm):"
            measurementName="height"
            value={measurements.height}
            onChange={handleMeasurementChange}
            onPrevious={() => handlePrevious('belly')}
            onSubmit={handleSubmit}
          />
        );
      case 'results':
        return <ResultsPage predictedType={predictedBodyType} />;
      default:
        return null;
    }
  };

  return (
    <div className="quiz-container">
      <h2 className="quiz-title">Find Your Body Type</h2>
      {renderPage()}
    </div>
  );
};

export default BodyTypeQuiz;
