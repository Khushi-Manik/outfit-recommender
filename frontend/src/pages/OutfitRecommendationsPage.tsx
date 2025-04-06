import React, { useState } from 'react';
import './OutfitRecommendationsPage.css'; // Create this CSS file

interface OutfitOptions {
  bodyType: string;
  color: string;
  weather: string;
  occasion: string;
  clothingPreferences: string[];
}

const defaultOptions: OutfitOptions = {
  bodyType: '',
  color: '',
  weather: '',
  occasion: '',
  clothingPreferences: [],
};

const OutfitRecommendationsPage: React.FC = () => {
  const [options, setOptions] = useState(defaultOptions);
  const [recommendations, setRecommendations] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  const handleOptionChange = (
    event: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>
  ) => {
    const { name, value, type } = event.target;

    if (type === 'checkbox') {
      const target = event.target as HTMLInputElement; // Type assertion to HTMLInputElement
      setOptions((prevOptions) => ({
        ...prevOptions,
        clothingPreferences: target.checked
          ? [...prevOptions.clothingPreferences, value]
          : prevOptions.clothingPreferences.filter((pref) => pref !== value),
      }));
    } else {
      setOptions((prevOptions) => ({
        ...prevOptions,
        [name]: value,
      }));
    }
  };

  const handleSubmit = async () => {
    setLoading(true);
    setRecommendations([]); // Clear previous recommendations

    // Simulate fetching recommendations based on options
    await new Promise((resolve) => setTimeout(resolve, 1500));

    const generatedRecommendations = generateDummyRecommendations(options);
    setRecommendations(generatedRecommendations);
    setLoading(false);
  };

  const generateDummyRecommendations = (options: OutfitOptions): string[] => {
    const { bodyType, color, weather, occasion, clothingPreferences } = options;
    const recommendations: string[] = [];

    let baseOutfit = '';

    if (occasion === 'Party') {
      baseOutfit = 'Stylish Party Outfit: ';
    } else if (occasion === 'Meeting') {
      baseOutfit = 'Professional Meeting Attire: ';
    } else if (occasion === 'Clg') {
      baseOutfit = 'Casual College Look: ';
    } else if (occasion === 'Outing') {
      baseOutfit = 'Comfortable Outing Ensemble: ';
    }

    baseOutfit += `Featuring a ${color} top `;

    if (clothingPreferences.includes('Dress')) {
      recommendations.push(baseOutfit + 'and a flattering dress suitable for the ' + occasion + ' in ' + weather + ' weather.');
    }
    if (clothingPreferences.includes('Top')) {
      recommendations.push(baseOutfit + 'paired with stylish bottoms for the ' + occasion + ' in ' + weather + ' weather.');
    }
    if (clothingPreferences.includes('Bottom')) {
      recommendations.push(baseOutfit + 'with comfortable bottoms perfect for the ' + occasion + ' in ' + weather + ' weather.');
    }
    if (clothingPreferences.includes('Jacket')) {
      recommendations.push(baseOutfit + 'layered with a trendy jacket for the ' + occasion + ' in ' + weather + ' weather.');
    }
    if (clothingPreferences.length === 0) {
      recommendations.push(baseOutfit + 'a versatile outfit suitable for the ' + occasion + ' in ' + weather + ' weather.');
    }

    recommendations.push(`(Considering your body type: ${bodyType})`);
    return recommendations;
  };

  return (
    <div className="recommendations-container">
      <h2 className="recommendations-title">Outfit Recommendations</h2>
      <div className="options-panel">
        <div className="option">
          <label htmlFor="bodyType">Body Type:</label>
          <select
            id="bodyType"
            name="bodyType"
            value={options.bodyType}
            onChange={handleOptionChange}
          >
            <option value="">Select</option>
            <option value="Hourglass">Hourglass</option>
            <option value="Rectangle">Rectangle</option>
            <option value="Pear">Pear</option>
            <option value="Apple">Apple</option>
            <option value="Inverted Triangle">Inverted Triangle</option>
          </select>
        </div>

        <div className="option">
          <label htmlFor="color">Color:</label>
          <input
            type="text"
            id="color"
            name="color"
            value={options.color}
            onChange={handleOptionChange}
            placeholder="e.g., Red, Blue, Floral"
          />
        </div>

        <div className="option">
          <label htmlFor="weather">Weather:</label>
          <select
            id="weather"
            name="weather"
            value={options.weather}
            onChange={handleOptionChange}
          >
            <option value="">Select</option>
            <option value="Sunny">Sunny</option>
            <option value="Rainy">Rainy</option>
            <option value="Cloudy">Cloudy</option>
            <option value="Winter">Winter</option>
            <option value="Summer">Summer</option>
            <option value="Spring">Spring</option>
            <option value="Autumn">Autumn</option>
          </select>
        </div>

        <div className="option">
          <label htmlFor="occasion">Occasion:</label>
          <select
            id="occasion"
            name="occasion"
            value={options.occasion}
            onChange={handleOptionChange}
          >
            <option value="">Select</option>
            <option value="Party">Party</option>
            <option value="Meeting">Meeting</option>
            <option value="Clg">College</option>
            <option value="Outing">Outing</option>
            <option value="Date">Date</option>
            <option value="Wedding">Wedding</option>
            <option value="Casual">Casual</option>
            <option value="Formal">Formal</option>
            {/* Add more occasions as needed */}
          </select>
        </div>

        <div className="option">
          <label>Clothing Preferences:</label>
          <div className="checkbox-group">
            <label>
              <input
                type="checkbox"
                name="clothingPreferences"
                value="Dress"
                checked={options.clothingPreferences.includes('Dress')}
                onChange={handleOptionChange}
              />
              Dress
            </label>
            <label>
              <input
                type="checkbox"
                name="clothingPreferences"
                value="Top"
                checked={options.clothingPreferences.includes('Top')}
                onChange={handleOptionChange}
              />
              Top
            </label>
            <label>
              <input
                type="checkbox"
                name="clothingPreferences"
                value="Bottom"
                checked={options.clothingPreferences.includes('Bottom')}
                onChange={handleOptionChange}
              />
              Bottom
            </label>
            <label>
              <input
                type="checkbox"
                name="clothingPreferences"
                value="Jacket"
                checked={options.clothingPreferences.includes('Jacket')}
                onChange={handleOptionChange}
              />
              Jacket
            </label>
            {/* Add more clothing preferences */}
          </div>
        </div>

        <button className="get-recommendations-btn" onClick={handleSubmit} disabled={loading}>
          {loading ? 'Getting Recommendations...' : 'Get Recommendations'}
        </button>
      </div>

      {recommendations.length > 0 && (
        <div className="recommendations-results">
          <h3>Outfit Suggestions:</h3>
          <ul className="recommendation-list">
            {recommendations.map((rec, index) => (
              <li key={index}>{rec}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default OutfitRecommendationsPage;