import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from typing import Dict, Union
from pydantic import BaseModel
from fastapi import FastAPI


# Constants
BODY_TYPES = ['Apple', 'Pear', 'Inverted Triangle', 'Hourglass', 'Rectangle']
MODEL_DIR = "models"

# Make sure directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Path constants
SCALER_PATH = os.path.join(MODEL_DIR, 'body_type_scaler.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
GB_MODEL_PATH = os.path.join(MODEL_DIR, 'gradient_boosting_model.pkl')
NN_MODEL_PATH = os.path.join(MODEL_DIR, 'neural_network_model.keras')

class BodyTypeModel:
    def __init__(self, train_data_path=None):
        """
        Initialize the BodyTypeModel.
        
        Args:
            train_data_path: Path to training data CSV file. If None, no training is performed.
        """
        self.scaler = None
        self.rf_model = None
        self.gb_model = None
        self.nn_model = None
        self.features = [
            'ShoulderWidth', 'ChestWidth', 'Waist', 'Hips', 'Belly', 'TotalHeight',
            'Bust_Waist_Ratio', 'Waist_Hip_Ratio', 'Shoulder_Hip_Ratio', 'Shoulder_Bust_Ratio',
            'Bust_Hip_Diff', 'Waist_Hip_Diff', 'Shoulder_Hip_Diff', 'Bust_Waist_Diff', 'BMI_Proxy'
        ]
        
        # Always load models first
        self.load_models()
        
        # Only train if explicitly requested and models don't exist
        if train_data_path and (not os.path.exists(RF_MODEL_PATH) or 
                               not os.path.exists(GB_MODEL_PATH) or 
                               not os.path.exists(NN_MODEL_PATH)):
            print("Models not found. Training new models...")
            self.train(train_data_path)
    
    def determine_body_type(self, measurements: Dict[str, float]) -> int:
        """
        Rule-based body type determination.
        
        Args:
            measurements: Dictionary of body measurements
            
        Returns:
            int: Body type index (0-4)
        """
        bust = measurements['ChestWidth']
        waist = measurements['Waist']
        hips = measurements['Hips']
        shoulder = measurements['ShoulderWidth']

        # Apple: Waist > Bust AND Waist > Hips (full middle)
        if waist > bust and waist > hips:
            return 0
        # Pear: Hips > Bust by significant amount AND Hips > Shoulder (bottom heavy)
        elif hips > bust + 1.5 and hips > shoulder + 1:
            return 1
        # Inverted Triangle: Shoulder > Hips significantly AND Bust > Hips (top heavy)
        elif shoulder > hips + 1.5 and bust > hips + 1:
            return 2
        # Hourglass: Bust ~ Hips AND Waist significantly smaller
        elif abs(bust - hips) <= 1.5 and waist < bust - 2 and waist < hips - 2:
            return 3
        # Rectangle: All measurements relatively similar
        else:
            return 4
    
    def preprocess_input(self, measurements: Dict[str, float]) -> pd.DataFrame:
        """
        Preprocess input measurements to create features.
        
        Args:
            measurements: Dictionary of body measurements
            
        Returns:
            pd.DataFrame: Processed features
        """
        input_df = pd.DataFrame([measurements])
        
        # Calculate derived features
        input_df['Bust_Waist_Ratio'] = input_df['ChestWidth'] / input_df['Waist']
        input_df['Waist_Hip_Ratio'] = input_df['Waist'] / input_df['Hips']
        input_df['Shoulder_Hip_Ratio'] = input_df['ShoulderWidth'] / input_df['Hips']
        input_df['Shoulder_Bust_Ratio'] = input_df['ShoulderWidth'] / input_df['ChestWidth']
        input_df['Bust_Hip_Diff'] = input_df['ChestWidth'] - input_df['Hips']
        input_df['Waist_Hip_Diff'] = input_df['Waist'] - input_df['Hips']
        input_df['Shoulder_Hip_Diff'] = input_df['ShoulderWidth'] - input_df['Hips']
        input_df['Bust_Waist_Diff'] = input_df['ChestWidth'] - input_df['Waist']
        input_df['BMI_Proxy'] = input_df['Belly'] / (input_df['TotalHeight'] ** 2) * 100
        
        # Ensure columns are in the right order
        return input_df[self.features]
    
    def train(self, data_path: str) -> None:
        """
        Train the body type prediction models.
        
        Args:
            data_path: Path to the CSV file with training data
        """
        print("Loading dataset...")
        df = pd.read_csv(data_path)
        
        # Clean column names - remove any trailing spaces
        df.columns = df.columns.str.strip()
        
        # Add body type labels
        df['BodyType'] = df.apply(lambda row: self.determine_body_type(row), axis=1)
        
        # Drop missing values
        df = df.dropna()
        
        # Enhanced feature engineering
        df = df.copy()
        df['Bust_Waist_Ratio'] = df['ChestWidth'] / df['Waist']
        df['Waist_Hip_Ratio'] = df['Waist'] / df['Hips']
        df['Shoulder_Hip_Ratio'] = df['ShoulderWidth'] / df['Hips']
        df['Shoulder_Bust_Ratio'] = df['ShoulderWidth'] / df['ChestWidth']
        df['Bust_Hip_Diff'] = df['ChestWidth'] - df['Hips']
        df['Waist_Hip_Diff'] = df['Waist'] - df['Hips']
        df['Shoulder_Hip_Diff'] = df['ShoulderWidth'] - df['Hips']
        df['Bust_Waist_Diff'] = df['ChestWidth'] - df['Waist']
        df['BMI_Proxy'] = df['Belly'] / (df['TotalHeight'] ** 2) * 100
        
        X = df[self.features]
        y = df['BodyType']
        
        # Split data for final testing
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply SMOTE to training data only
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_full, y_train_full)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_resampled)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save the scaler
        joblib.dump(self.scaler, SCALER_PATH)
        print(f"Saved scaler to {SCALER_PATH}")
        
        # Train Random Forest
        print("Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self.rf_model.fit(X_train_scaled, y_train_resampled)
        joblib.dump(self.rf_model, RF_MODEL_PATH)
        print(f"Saved Random Forest model to {RF_MODEL_PATH}")
        
        # Train Gradient Boosting
        print("Training Gradient Boosting...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.gb_model.fit(X_train_scaled, y_train_resampled)
        joblib.dump(self.gb_model, GB_MODEL_PATH)
        print(f"Saved Gradient Boosting model to {GB_MODEL_PATH}")
        
        # Train Neural Network
        print("Training Neural Network...")
        self.nn_model = self._create_nn_model(len(self.features))
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
        
        class_weights = {
            i: len(y_train_resampled) / (len(np.unique(y_train_resampled)) * np.sum(y_train_resampled == i))
            for i in np.unique(y_train_resampled)
        }
        
        self.nn_model.fit(
            X_train_scaled, y_train_resampled,
            epochs=100,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )
        
        self.nn_model.save(NN_MODEL_PATH)
        print(f"Saved Neural Network model to {NN_MODEL_PATH}")
        
        # Evaluate models
        print("\nEvaluating models on test data...")
        
        rf_pred = self.rf_model.predict(X_test_scaled)
        rf_accuracy = np.mean(rf_pred == y_test)
        print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
        
        gb_pred = self.gb_model.predict(X_test_scaled)
        gb_accuracy = np.mean(gb_pred == y_test)
        print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f}")
        
        nn_loss, nn_accuracy = self.nn_model.evaluate(X_test_scaled, y_test)
        print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
        
        # Show detailed classification report for ensemble
        print("\nDetailed classification report for ensemble:")
        ensemble_pred = self._ensemble_predict(X_test_scaled)
        print(classification_report(y_test, ensemble_pred, target_names=BODY_TYPES))
    
    def _create_nn_model(self, input_shape: int) -> keras.Sequential:
        """
        Create a neural network model for body type prediction.
        
        Args:
            input_shape: Number of input features
            
        Returns:
            keras.Sequential: Neural network model
        """
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.BatchNormalization(),
            
            layers.Dense(256, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            layers.Dense(128, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            layers.Dense(64, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            
            layers.Dense(32, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.1),
            
            layers.Dense(16, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Dropout(0.1),
            
            layers.Dense(5, activation='softmax')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_models(self) -> None:
        """Load pre-trained models and scaler"""
        try:
            print("Loading pre-trained models...")
            self.scaler = joblib.load(SCALER_PATH)
            self.rf_model = joblib.load(RF_MODEL_PATH)
            self.gb_model = joblib.load(GB_MODEL_PATH)
            
            # Check if neural network model exists
            if os.path.exists(NN_MODEL_PATH):
                self.nn_model = keras.models.load_model(NN_MODEL_PATH)
            else:
                print(f"Neural network model not found at {NN_MODEL_PATH}")
                
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            print("Models need to be trained first")
    
    def _ensemble_predict(self, X_scaled):
        """
        Internal method to combine predictions from all models.
        
        Args:
            X_scaled: Scaled input features
            
        Returns:
            numpy.ndarray: Final predictions
        """
        # Get predictions from each model
        rf_pred = self.rf_model.predict(X_scaled)
        gb_pred = self.gb_model.predict(X_scaled)
        nn_pred = np.argmax(self.nn_model.predict(X_scaled), axis=1)
        
        # Get confidence scores
        rf_conf = np.max(self.rf_model.predict_proba(X_scaled), axis=1)
        gb_conf = np.max(self.gb_model.predict_proba(X_scaled), axis=1)
        nn_conf = np.max(self.nn_model.predict(X_scaled), axis=1)
        
        # Ensemble predictions with weighted voting
        final_pred = np.zeros(len(X_scaled), dtype=int)
        
        for i in range(len(X_scaled)):
            # Calculate rule-based prediction for this sample
            # We need to convert the scaled input back to original values
            original_values = self.scaler.inverse_transform([X_scaled[i]])[0]
            
            # Create a dictionary mapping feature names to their values
            measurements = {
                feature: value for feature, value in zip(self.features, original_values)
            }
            
            # Keep only the base measurements
            base_measurements = {
                k: v for k, v in measurements.items() 
                if k in ['ShoulderWidth', 'ChestWidth', 'Waist', 'Hips', 'Belly', 'TotalHeight']
            }
            
            # Get rule-based prediction
            rule_based = self.determine_body_type(base_measurements)
            
            # Weighted voting
            votes = np.zeros(5)
            votes[rf_pred[i]] += rf_conf[i]
            votes[gb_pred[i]] += gb_conf[i]
            votes[nn_pred[i]] += nn_conf[i]
            votes[rule_based] += 1.2  # Rule-based gets a higher weight
            
            final_pred[i] = np.argmax(votes)
        
        return final_pred
    
    def predict(self, measurements: Dict[str, float]) -> Dict[str, Union[str, float]]:
        """
        Predict body type using ensemble of models.
        
        Args:
            measurements: Dictionary with keys 'shoulder', 'bust', 'waist', 
                         'hips', 'belly', 'height'
            
        Returns:
            Dict: Dictionary with body type and confidence score
        """
        # Convert input measurements to format expected by model
        formatted_measurements = {
            'ShoulderWidth': measurements['shoulder'],
            'ChestWidth': measurements['bust'],
            'Waist': measurements['waist'],
            'Hips': measurements['hips'],
            'Belly': measurements['belly'],
            'TotalHeight': measurements['height']
        }
        
        # Preprocess input
        input_df = self.preprocess_input(formatted_measurements)
        
        # Scale input
        scaled_input = self.scaler.transform(input_df)
        
        # Get rule-based prediction
        rule_based_class = self.determine_body_type(formatted_measurements)
        
        # Get model predictions
        predictions = {}
        confidences = {}
        
        # Random Forest prediction
        if self.rf_model:
            pred_class = self.rf_model.predict(scaled_input)[0]
            confidence = self.rf_model.predict_proba(scaled_input)[0, pred_class]
            predictions["Random Forest"] = pred_class
            confidences["Random Forest"] = confidence
        
        # Gradient Boosting prediction
        if self.gb_model:
            pred_class = self.gb_model.predict(scaled_input)[0]
            confidence = self.gb_model.predict_proba(scaled_input)[0, pred_class]
            predictions["Gradient Boosting"] = pred_class
            confidences["Gradient Boosting"] = confidence
        
        # Neural Network prediction
        if self.nn_model:
            pred_probs = self.nn_model.predict(scaled_input)[0]
            pred_class = np.argmax(pred_probs)
            confidence = pred_probs[pred_class]
            predictions["Neural Network"] = pred_class
            confidences["Neural Network"] = confidence
        
        # Weighted voting system
        vote_counts = {i: 0 for i in range(5)}
        
        # Add votes from each model, weighted by confidence
        for name, pred_class in predictions.items():
            vote_counts[pred_class] += confidences[name]
        
        # Add vote for rule-based method with high weight
        vote_counts[rule_based_class] += 1.2
        
        # Find the class with the most votes
        final_class = max(vote_counts.items(), key=lambda x: x[1])[0]
        final_confidence = vote_counts[final_class] / (len(predictions) + 1.2)
        
        return {
            "body_type": BODY_TYPES[final_class],
            "confidence": float(final_confidence)
        }


# Example usage
if __name__ == "__main__":
    # Just load pre-trained models without training
    model = BodyTypeModel()
    
    # Test prediction
    test_data = {
        'shoulder': 18.0,
        'bust': 20.0,
        'waist': 14.0,
        'hips': 22.0,
        'belly': 18.0,
        'height': 52.0
    }
    
    result = model.predict(test_data)
    print(f"Predicted body type: {result['body_type']}")
    print(f"Confidence: {result['confidence']:.4f}")

# Pydantic classes for FastAPI
class BodyMeasurements(BaseModel):
    shoulder: float
    bust: float
    waist: float
    hips: float
    belly: float
    height: float

class PredictionResult(BaseModel):
    body_type: str
    confidence: float

# Optional if main.py expects this function
def create_fastapi_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="Body Type Prediction API")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Modify this in production to only allow your frontend
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize model on startup - only load models, no training
    model = BodyTypeModel()
    
    @app.post("/predict", response_model=PredictionResult)
    async def predict_body_type(measurements: BodyMeasurements):
        try:
            result = model.predict(measurements.dict())
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app