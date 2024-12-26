import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

class SimplePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.location_columns = None
        self.avg_prices = {}

    def prepare_data(self, df):
        """Data with one-hot encoded locations"""
        # Find location columns
        self.location_columns = [col for col in df.columns if col.startswith('location_')]
        if not self.location_columns:
            raise ValueError("No location columns found. Ensure location columns start with 'location_'")

        # Calculate average price per sqft for each location
        for loc_col in self.location_columns:
            loc_data = df[df[loc_col] == 1]
            if len(loc_data) > 0:
                self.avg_prices[loc_col] = loc_data['price_per_sqft'].mean()

        # Prepare features
        feature_columns = ['size', 'total_sqft', 'bath', 'balcony'] + self.location_columns
        X = df[feature_columns].copy()

        # Target variable
        y = df['price_per_sqft']

        return X, y

    def train(self, file_path):
        """Training the model on the specified features"""
        # Load data
        df = pd.read_csv(file_path)

        # Calculate price_per_sqft if not present
        if 'price_per_sqft' not in df.columns:
            df['price_per_sqft'] = df['price'] / df['total_sqft']

        # Remove outliers
        df = self.remove_outliers(df)

        # Prepare data
        X, y = self.prepare_data(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale non-location features
        self.scaler = StandardScaler()
        non_loc_cols = ['size', 'total_sqft', 'bath', 'balcony']
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()

        X_train_scaled[non_loc_cols] = self.scaler.fit_transform(X_train[non_loc_cols])
        X_test_scaled[non_loc_cols] = self.scaler.transform(X_test[non_loc_cols])

        # Training model with optimized parameters
        self.model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )

        # Fit model
        self.model.fit(X_train_scaled, y_train)

        # Make predictions
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        # Calculate metrics
        metrics = {
            'Train MAE': mean_absolute_error(y_train, train_pred),
            'Test MAE': mean_absolute_error(y_test, test_pred),
            'Train R2': r2_score(y_train, train_pred),
            'Test R2': r2_score(y_test, test_pred)
        }

        return metrics

    def remove_outliers(self, df):
        """Remove outliers using IQR method for each location"""
        df_clean = df.copy()

        # Finding location columns
        location_columns = [col for col in df.columns if col.startswith('location_')]

        # Remove price_per_sqft outliers by location
        for loc_col in location_columns:
            location_data = df[df[loc_col] == 1]
            if len(location_data) > 10:  # Only process locations with enough data
                Q1 = location_data['price_per_sqft'].quantile(0.05)
                Q3 = location_data['price_per_sqft'].quantile(0.95)
                IQR = Q3 - Q1

                # Filter data
                mask = df_clean[loc_col] == 1
                df_clean = df_clean[
                    ~mask | (
                        (df_clean['price_per_sqft'] >= Q1 - 1.5 * IQR) &
                        (df_clean['price_per_sqft'] <= Q3 + 1.5 * IQR)
                    )
                ]

        return df_clean

    def predict_price(self, size, total_sqft, bath, balcony, location_name):
        """Prediction of price per sqft and total price"""
        try:
            # Create location column name
            location_col = f"location_{location_name}"
            if location_col not in self.location_columns:
                raise ValueError(f"Location '{location_name}' not found in training data")

            # Prepare input features
            input_data = pd.DataFrame({
                'size': [size],
                'total_sqft': [total_sqft],
                'bath': [bath],
                'balcony': [balcony]
            })

            # Add location columns (all 0 except the selected location)
            for loc_col in self.location_columns:
                input_data[loc_col] = 1 if loc_col == location_col else 0

            # Scale non-location features
            non_loc_cols = ['size', 'total_sqft', 'bath', 'balcony']
            input_data[non_loc_cols] = self.scaler.transform(input_data[non_loc_cols])

            # Predict price per sqft
            predicted_price_per_sqft = self.model.predict(input_data)[0]

            # Calculate total price
            total_price = predicted_price_per_sqft * total_sqft

            return {
                'location': location_name,
                'area_sqft': total_sqft,
                'predicted_price_per_sqft': predicted_price_per_sqft,
                'predicted_total_price': total_price,
                'location_avg_price_per_sqft': self.avg_prices.get(location_col, 0)
            }

        except Exception as e:
            raise RuntimeError(f"Prediction error: {str(e)}")

# Example usage to test our model
if __name__ == "__main__":
    try:
        # Initialize and train model
        predictor = SimplePricePredictor()
        print("Training model...")
        metrics = predictor.train('/content/Bengaluru_House_Data_preprocessed.csv')

        # Print metrics
        print("\nModel Performance Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.2f}")

        # Make a prediction
        result = predictor.predict_price(
            size=4,
            total_sqft=2615,
            bath=5,
            balcony=3,
            location_name='1st Phase JP Nagar'  # Use location name without 'location_' prefix
        )

        print("\nPrediction Results:")
        print(f"Location: {result['location']}")
        print(f"Area: {result['area_sqft']} sq.ft")
        print(f"Average Price per sq.ft in this location: ₹{result['location_avg_price_per_sqft']:.2f}")
        print(f"Predicted Price per sq.ft: ₹{result['predicted_price_per_sqft']:.2f}")
        print(f"Predicted Total Price: ₹{result['predicted_total_price']:.2f}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")