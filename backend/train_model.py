import pickle
from sklearn.preprocessing import LabelEncoder

# Assuming you have already fitted these encoders
color_encoder = LabelEncoder()
weather_encoder = LabelEncoder()
occasion_encoder = LabelEncoder()
article_encoder = LabelEncoder()
body_type_encoder = LabelEncoder()

# Fit them with training data before this step...

# Save as a dictionary
encoders = {
    'color': color_encoder,
    'weather': weather_encoder,
    'occasion': occasion_encoder,
    'article': article_encoder,
    'body_type': body_type_encoder
}

with open("models/article_encoder.pkl", "wb") as f:
    pickle.dump(encoders, f)
