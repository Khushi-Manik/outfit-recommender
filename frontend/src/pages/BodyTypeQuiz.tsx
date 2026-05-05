import React, { useState } from 'react';
import './BodyTypeQuiz.css';
import placeholderShoulder from '@/assets/placeholder_shoulder.jpg';
import placeholderBust from '@/assets/placeholder_bust.jpg';
import placeholderWaist from '@/assets/placeholder_waist.jpg';
import placeholderHips from '@/assets/placeholder_hips.jpg';
import placeholderBelly from '@/assets/placeholder_belly.jpg';
import placeholderHeight from '@/assets/placeholder_height.jpg';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? 'http://localhost:8000';
const BODY_TYPES = ['Apple', 'Pear', 'Inverted Triangle', 'Hourglass', 'Rectangle'] as const;

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

const determineBodyType = (measurements: {
  shoulder: number;
  bust: number;
  waist: number;
  hips: number;
}): string => {
  const { shoulder, bust, waist, hips } = measurements;

  if (waist > bust && waist > hips) {
    return BODY_TYPES[0];
  }

  if (hips > bust + 1.5 && hips > shoulder + 1) {
    return BODY_TYPES[1];
  }

  if (shoulder > hips + 1.5 && bust > hips + 1) {
    return BODY_TYPES[2];
  }

  if (Math.abs(bust - hips) <= 1.5 && waist < bust - 2 && waist < hips - 2) {
    return BODY_TYPES[3];
  }

  return BODY_TYPES[4];
};

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
  onViewInfo: () => void;
  showInfo: boolean;
}

const BODY_TYPE_DETAILS: Record<string, { summary: string; recommendations: string[] }> = {
  Apple: {
    summary: 'Apple body types usually carry more shape around the midsection, with slimmer hips and legs.',
    recommendations: [
      'Choose V-necks, wrap tops, and open layers to create length.',
      'Go for straight-leg trousers or A-line skirts for balance.',
      'Use structured fabrics that skim instead of cling at the waist.',
    ],
  },
  Pear: {
    summary: 'Pear body types usually have fuller hips with a comparatively narrower upper body.',
    recommendations: [
      'Highlight the shoulders with boat necks, puff sleeves, or statement collars.',
      'Pick darker, streamlined bottoms and softer A-line silhouettes.',
      'Add texture, color, or accessories on top to balance proportions.',
    ],
  },
  'Inverted Triangle': {
    summary: 'Inverted triangle body types tend to have broader shoulders with a narrower lower half.',
    recommendations: [
      'Keep tops clean and structured without too much shoulder detail.',
      'Use wide-leg pants, pleated skirts, or printed bottoms to add volume below.',
      'Wrap dresses and belted outfits help rebalance the frame.',
    ],
  },
  Hourglass: {
    summary: 'Hourglass body types typically have balanced shoulders and hips with a defined waist.',
    recommendations: [
      'Use fitted cuts, wrap dresses, and belted outfits to emphasize the waist.',
      'Choose tailored pieces that follow your natural shape.',
      'Avoid overly boxy silhouettes that hide your proportions.',
    ],
  },
  Rectangle: {
    summary: 'Rectangle body types usually have similar shoulder, waist, and hip measurements with a straighter silhouette.',
    recommendations: [
      'Create shape with peplum tops, belted waists, and layered styling.',
      'Try ruffles, drape, or curved seams to add dimension.',
      'Mix fitted and relaxed pieces to build contrast through the outfit.',
    ],
  },
};

const ResultsPage: React.FC<ResultsPageProps> = ({ predictedType, onViewInfo, showInfo }) => {
  const detail = predictedType ? BODY_TYPE_DETAILS[predictedType] : undefined;

  return (
    <div className="results-page">
      <h3>Your Body Type</h3>
      {predictedType ? (
        <p id="body-type-result">{predictedType}</p>
      ) : (
        <p>Processing...</p>
      )}
      {detail && (
        <>
          <button className="view-info-btn" onClick={onViewInfo}>
            {showInfo ? 'Hide Full Info' : 'View Full Info'}
          </button>
          {showInfo && (
            <div className="body-type-info-card">
              <p className="body-type-summary">{detail.summary}</p>
              <ul className="body-type-tips">
                {detail.recommendations.map((tip) => (
                  <li key={tip}>{tip}</li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
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
  const [showBodyTypeInfo, setShowBodyTypeInfo] = useState(false);

  const hasCompleteMeasurements = (
    values: Measurements
  ): values is { shoulder: number; bust: number; waist: number; hips: number; belly: number; height: number } =>
    Object.values(values).every((value) => value !== null);

  const handleNext = (page: typeof currentPage) => setCurrentPage(page);
  const handlePrevious = (page: typeof currentPage) => setCurrentPage(page);

  const handleMeasurementChange = (name: keyof Measurements, value: number | null) => {
    setMeasurements({ ...measurements, [name]: value });
  };

  const handleSubmit = async () => {
    setCurrentPage('results');
    setPredictedBodyType(null);
    setShowBodyTypeInfo(false);

    if (!hasCompleteMeasurements(measurements)) {
      setPredictedBodyType('Please fill in all measurements before submitting.');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
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
      setPredictedBodyType(
        determineBodyType({
          shoulder: measurements.shoulder,
          bust: measurements.bust,
          waist: measurements.waist,
          hips: measurements.hips,
        })
      );
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
        return (
          <ResultsPage
            predictedType={predictedBodyType}
            showInfo={showBodyTypeInfo}
            onViewInfo={() => setShowBodyTypeInfo((current) => !current)}
          />
        );
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
