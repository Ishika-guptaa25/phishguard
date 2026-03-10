import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from feature_extractor import PhishingFeatureExtractor
import warnings

warnings.filterwarnings('ignore')


class PhishingDetectionModel:
    """
    Train and manage phishing detection models.
    Supports multiple algorithms with performance metrics.
    """

    def __init__(self):
        self.extractor = PhishingFeatureExtractor()
        self.model = None
        self.scaler = None
        self.feature_names = None

    def train_from_csv(self, csv_path, label_column='class', url_column='url', test_size=0.2,
                       model_type='random_forest'):
        """
        Train model from CSV dataset with a raw URL column.
        Args:
            csv_path: Path to CSV file
            label_column: Name of target column (0=legitimate, 1=phishing)
            url_column: Name of URL column
            test_size: Train/test split ratio
            model_type: 'random_forest' or 'gradient_boosting'
        """
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Extract features
        print("\nExtracting features from URLs...")
        X_list = []
        y_list = []
        feature_names = None

        for idx, row in df.iterrows():
            try:
                url = str(row[url_column])
                label = int(row[label_column])

                features, feature_names = self.extractor.extract_features(url)
                X_list.append(features)
                y_list.append(label)

                if (idx + 1) % 1000 == 0:
                    print(f"  Processed {idx + 1} URLs...")

            except Exception as e:
                print(f"  Error processing row {idx}: {e}")
                continue

        X = np.array(X_list)
        y = np.array(y_list)

        if len(X_list) == 0:
            raise ValueError("No URLs were successfully processed. Check your dataset and URL column.")

        self.feature_names = feature_names
        return self._train_and_evaluate(X, y, test_size, model_type)

    def train_from_features_csv(self, csv_path, label_column=None, test_size=0.2, model_type='random_forest'):
        """
        Train directly from a pre-extracted feature CSV (e.g. Kaggle phishing dataset).
        Skips URL parsing entirely — uses the numeric columns as features directly.

        Args:
            csv_path: Path to CSV file with pre-extracted numeric features
            label_column: Name of label column. If None, auto-detects.
            test_size: Train/test split ratio
            model_type: 'random_forest' or 'gradient_boosting'
        """
        print(f"Loading feature dataset from {csv_path}...")
        df = pd.read_csv(csv_path)

        print(f"Dataset shape: {df.shape}")

        # Auto-detect label column
        if label_column is None:
            for candidate in ['phishing', 'label', 'class', 'target', 'status']:
                if candidate in df.columns:
                    label_column = candidate
                    break
            if label_column is None:
                label_column = df.columns[-1]
                print(f"Could not find standard label column. Using last column: '{label_column}'")
            else:
                print(f"Auto-detected label column: '{label_column}'")

        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found. Available: {df.columns.tolist()}")

        # Use only numeric feature columns, drop URL/text/label columns
        non_feature_cols = {label_column, 'url', 'URL', 'domain', 'Domain'}
        feature_cols = [
            c for c in df.columns
            if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        print(f"Feature columns found: {len(feature_cols)}")
        print(f"Label distribution:\n{df[label_column].value_counts().to_string()}")

        X = df[feature_cols].fillna(0).values
        y = df[label_column].values

        self.feature_names = feature_cols
        return self._train_and_evaluate(X, y, test_size, model_type)

    def _train_and_evaluate(self, X, y, test_size=0.2, model_type='random_forest'):
        """
        Shared training + evaluation logic used by both train methods.
        """
        print(f"\nSamples: {X.shape[0]}, Features: {X.shape[1]}")
        print(f"Class distribution: Legitimate={sum(y == 0)}, Phishing={sum(y == 1)}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Standardize features
        print("\nStandardizing features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        print(f"\nTraining {model_type} model...")
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                subsample=0.8
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\nEvaluating model...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        accuracy = self.model.score(X_test_scaled, y_test)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nModel Performance:")
        print(f"  Accuracy:      {accuracy:.4f}")
        print(f"  ROC-AUC Score: {roc_auc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

        # Top feature importances
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importances = self.model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1][:10]
            print(f"\nTop 10 Important Features:")
            for rank, idx in enumerate(sorted_idx, 1):
                print(f"  {rank}. {self.feature_names[idx]}: {importances[idx]:.4f}")

        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

    def predict(self, url):
        """
        Predict if a raw URL is phishing.
        Handles mismatch between URL-extracted features (35) and model features (111).
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Load or train model first.")

        # Extract what we can from the URL
        raw_features, raw_feature_names = self.extractor.extract_features(url)

        # Build a feature vector matching exactly what the model was trained on
        if self.feature_names:
            feature_vector = np.zeros(len(self.feature_names))
            name_to_val = dict(zip(raw_feature_names, raw_features))
            for i, fname in enumerate(self.feature_names):
                if fname in name_to_val:
                    feature_vector[i] = name_to_val[fname]
        else:
            feature_vector = raw_features

        features_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        return {
            'is_phishing': bool(prediction),
            'confidence': float(probability[prediction]),
            'phishing_probability': float(probability[1]),
            'legitimate_probability': float(probability[0])
        }

    def save_model(self, model_path='phishing_model.pkl', scaler_path='scaler.pkl', features_path='features.pkl'):
        """Save trained model, scaler, and feature names"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.feature_names, features_path)
        print(f"Model saved  → {model_path}")
        print(f"Scaler saved → {scaler_path}")
        print(f"Features saved → {features_path}")

    def load_model(self, model_path='phishing_model.pkl', scaler_path='scaler.pkl', features_path='features.pkl'):
        """Load trained model, scaler, and feature names"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = joblib.load(features_path)
        print("Model loaded successfully")


if __name__ == "__main__":
    model = PhishingDetectionModel()

    csv_path = "dataset_full.csv"

    # dataset_full.csv has 111 pre-extracted numeric features + 'phishing' label column.
    # Use train_from_features_csv — no URL parsing needed.
    metrics = model.train_from_features_csv(csv_path, label_column='phishing')

    # Save model
    model.save_model()

    print(f"\nFinal Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Final ROC-AUC:   {metrics['roc_auc']:.4f}")