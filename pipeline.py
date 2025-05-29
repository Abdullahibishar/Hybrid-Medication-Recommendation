# pipeline.py
import pandas as pd
import numpy as np
import re
import math
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GridSearchCV, train_test_split

# For handling class imbalance using SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class HybridRecommenderPipeline:
    def __init__(self, csv_path):
        # Load the dataset from CSV
        self.df = pd.read_csv(csv_path)
        self._explore_data()
        self._preprocess_data()
        self._setup_supervised_model()
        self._setup_cf_model()
        self._setup_cb_model()
    
    def _explore_data(self):
        # Display basic information about the dataset.
        print("Dataset Shape:", self.df.shape)
        print("Columns:", self.df.columns.tolist())
        print("Missing values per column:\n", self.df.isnull().sum())
    
    def _preprocess_data(self):
        df = self.df
        # --- Gender Handling ---
        # Replace "Other" with "Female" for simplicity, then convert to numeric: 0 = Male, 1 = Female.
        df['Gender'] = df['Gender'].apply(lambda x: 'Female' if x == 'Other' else x)
        df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
        
        # --- Fill Missing Medication ---
        # Use the most common medication to fill missing Recommended_Medication values.
        most_common_medication = df['Recommended_Medication'].mode()[0]
        df['Recommended_Medication'] = df['Recommended_Medication'].fillna(most_common_medication)
        
        # --- Combined Text Feature ---
        # Concatenate Diagnosis and Symptoms into one text field.
        df["combined_text"] = df["Diagnosis"].fillna('') + " " + df["Symptoms"].fillna('')
        
        # --- Dosage Cleaning ---
        # Remove non-numeric characters (e.g., "mg") from Dosage.
        def clean_dosage(val):
            if pd.isna(val):
                return np.nan
            try:
                return float(val)
            except:
                m = re.search(r'\d+(\.\d+)?', str(val))
                return float(m.group()) if m else np.nan
        df['Dosage_clean'] = df['Dosage'].apply(clean_dosage)
        median_value = df['Dosage_clean'].median()
        df['Dosage'] = df['Dosage_clean'].fillna(median_value)
        df['Dosage'] = df['Dosage'].astype(float)
        
        # --- Duration Conversion ---
        # Convert duration text (e.g., "3 days") into numeric hours.
        def convert_to_hours(duration):
            if pd.isna(duration) or duration == 'None':
                return np.nan
            if 'days' in duration:
                days = int(duration.split()[0])
                return days * 24
            return np.nan
        df['Duration'] = df['Duration'].apply(convert_to_hours)
        df = df.rename(columns={'Duration': 'Duration(hrs)'})
        
        # --- Duration Imputation ---
        # Impute missing Duration(hrs) using regression if necessary.
        features = ['Age', 'Gender', 'BMI', 'Recovery_Time_Days']
        df_train = df.dropna(subset=['Duration(hrs)']).copy()
        df_test = df[df['Duration(hrs)'].isna()].copy()
        if not df_test.empty:
            X_train = df_train[features]
            y_train = df_train['Duration(hrs)']
            X_test = df_test[features]
            X_train_split, X_valid, y_train_split, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_reg.fit(X_train_split, y_train_split)
            y_pred_valid = rf_reg.predict(X_valid)
            from sklearn.metrics import mean_absolute_error
            mae = mean_absolute_error(y_valid, y_pred_valid)
            print(f"Validation MAE for Duration(hrs) imputation: {mae:.2f}")
            df.loc[df['Duration(hrs)'].isna(), 'Duration(hrs)'] = rf_reg.predict(X_test)
        
        df['Duration(hrs)'] = df['Duration(hrs)'].astype(int)
        # Winsorize Duration(hrs) to reduce outliers' impact
        Q1 = df["Duration(hrs)"].quantile(0.25)
        Q3 = df["Duration(hrs)"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df["Duration(hrs)"] = df["Duration(hrs)"].clip(lower=lower_bound, upper=upper_bound)
        
        self.df = df
        print("Preprocessing completed. Missing values after processing:\n", self.df.isnull().sum())
    
    def _setup_supervised_model(self):
        df = self.df
        # Encode target: Convert medication names to numerical labels.
        self.le_med = LabelEncoder()
        df['Medication_Label'] = self.le_med.fit_transform(df['Recommended_Medication'])
        numeric_features = ['Age', 'Gender', 'BMI', 'Recovery_Time_Days', 'Duration(hrs)']
        text_feature = 'combined_text'
        X = df[numeric_features + [text_feature]]
        y = df['Medication_Label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Build preprocessing pipeline for numeric and text features.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('text', Pipeline([
                    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_features=3000)),
                    ('svd', TruncatedSVD(n_components=20, random_state=42))
                ]), text_feature)
            ]
        )
        
        # Build a pipeline using RandomForestClassifier with class_weight='balanced' and apply SMOTE.
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        imb_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ])
        
        param_grid_rf = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
        
        grid_search_rf = GridSearchCV(imb_pipeline, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)
        
        print("Best parameters with SMOTE for RandomForestClassifier:")
        print(grid_search_rf.best_params_)
        
        y_pred_rf = grid_search_rf.best_estimator_.predict(X_test)
        from sklearn.metrics import accuracy_score, classification_report
        rf_accuracy = accuracy_score(y_test, y_pred_rf)
        print("RandomForest Classifier Accuracy:", rf_accuracy)
        print(classification_report(y_test, y_pred_rf, target_names=[str(x) for x in self.le_med.classes_]))
        
        # Store the best supervised model.
        self.supervised_model = grid_search_rf.best_estimator_
        self.numeric_features = numeric_features
        self.text_feature = text_feature
    
    def get_supervised_recommendation(self, patient_index):
        patient_data = self.df.iloc[[patient_index]][self.numeric_features + [self.text_feature]]
        pred_label = self.supervised_model.predict(patient_data)[0]
        return self.le_med.inverse_transform([pred_label])[0]
    
    def _setup_cf_model(self):
        cf_features = ['Age', 'BMI', 'Recovery_Time_Days', 'Duration(hrs)']
        self.cf_data = self.df[cf_features].copy()
        self.nn_model = NearestNeighbors(n_neighbors=6, metric='euclidean')
        self.nn_model.fit(self.cf_data)
    
    def get_cf_recommendations(self, patient_index, top_n=5):
        distances, indices = self.nn_model.kneighbors(self.cf_data.iloc[patient_index:patient_index+1])
        neighbor_indices = indices.flatten()[1:top_n+1]
        cf_rec = self.df.iloc[neighbor_indices][["Patient_ID", "Recommended_Medication"]]
        return cf_rec
    
    def _setup_cb_model(self):
        self.df["combined_text"] = self.df["Diagnosis"].fillna('') + " " + self.df["Symptoms"].fillna('')
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["combined_text"])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
    
    def get_cb_recommendations(self, patient_index, top_n=5):
        sim_scores = list(enumerate(self.cosine_sim[patient_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        neighbor_indices = [i[0] for i in sim_scores]
        cb_rec = self.df.iloc[neighbor_indices][["Patient_ID", "Diagnosis", "Recommended_Medication"]]
        return cb_rec
    
    # New method: Generate recommendations from free-text symptoms input
    def get_recommendations_from_symptoms(self, symptoms_text, top_n=5):
        # Transform the input symptoms using the same vectorizer from the CB model.
        query_vec = self.vectorizer.transform([symptoms_text])
        # Compute cosine similarity between the query and the dataset
        query_sim = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        # Get indices of the top similar records
        top_indices = np.argsort(query_sim)[::-1][:top_n]
        # Retrieve diagnoses and medications from these records.
        diagnoses = self.df.iloc[top_indices]["Diagnosis"]
        medications = self.df.iloc[top_indices]["Recommended_Medication"]
        # Get the most common diagnosis as a prediction
        predicted_disease = diagnoses.mode()[0] if not diagnoses.mode().empty else "Unknown"
        # Aggregate medication recommendations
        med_counts = medications.value_counts().reset_index()
        med_counts.columns = ['Medication', 'Count']
        return predicted_disease, med_counts
    
    def get_hybrid_recommendations(self, patient_index, top_n=5, cf_weight=0.1, cb_weight=0.1, sup_weight=0.7):
        cf_rec = self.get_cf_recommendations(patient_index, top_n=top_n)
        cb_rec = self.get_cb_recommendations(patient_index, top_n=top_n)
        sup_rec = self.get_supervised_recommendation(patient_index)
        combined_rec = pd.concat([
            cf_rec["Recommended_Medication"],
            cb_rec["Recommended_Medication"],
            pd.Series([sup_rec])
        ], ignore_index=True)
        rec_counts = combined_rec.value_counts().reset_index()
        rec_counts.columns = ['Medication', 'Count']
        top_recommendations = rec_counts.head(top_n)
        return top_recommendations

# End of pipeline.py
