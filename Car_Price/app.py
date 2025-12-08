"""
Advanced Car Price Prediction Web Application
Complete ML system with training, prediction, and visualization
"""
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Cleaned_Car_data.csv"
MODEL_PATH = BASE_DIR / "best_car_price_model.pkl"


# Page configuration
st.set_page_config(
    page_title="Car Price Prediction System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f2f6 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


class CarPricePredictor:
    """Car Price Prediction Model Class"""
    def __init__(self):
        self.models = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load and preprocess the car data"""
        df = pd.read_csv(filepath)
        
        # Remove unnamed index column if exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        if len(df.columns) > 0 and df.columns[0] == '':
            df = df.drop(df.columns[0], axis=1)
        
        return df
    
    def feature_engineering(self, df):
        """Advanced feature engineering"""
        df = df.copy()
        
        # Calculate car age
        current_year = 2024
        df['car_age'] = current_year - df['year']
        
        # Extract brand from name
        df['brand'] = df['name'].apply(lambda x: str(x).split()[0])
        
        # Kilometers per year (usage intensity)
        df['kms_per_year'] = df['kms_driven'] / (df['car_age'] + 1)
        
        # Encode categorical variables
        le_company = LabelEncoder()
        le_fuel = LabelEncoder()
        le_brand = LabelEncoder()
        
        df['company_encoded'] = le_company.fit_transform(df['company'])
        df['fuel_encoded'] = le_fuel.fit_transform(df['fuel_type'])
        df['brand_encoded'] = le_brand.fit_transform(df['brand'])
        
        # Store encoders
        self.label_encoders['company'] = le_company
        self.label_encoders['fuel'] = le_fuel
        self.label_encoders['brand'] = le_brand
        
        # Feature columns for training
        self.feature_columns = ['year', 'kms_driven', 'company_encoded', 'fuel_encoded', 
                               'car_age', 'kms_per_year', 'brand_encoded']
        
        return df
    
    def train_models(self, X_train, y_train):
        """Train multiple ML models with hyperparameter tuning"""
        progress_placeholder = st.empty()
        
        # Random Forest
        progress_placeholder.info("🔄 Training Random Forest...")
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10]
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='neg_mean_squared_error', 
                               n_jobs=-1, verbose=0)
        rf_grid.fit(X_train, y_train)
        self.models['Random Forest'] = rf_grid.best_estimator_
        
        # Gradient Boosting
        progress_placeholder.info("🔄 Training Gradient Boosting...")
        gb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        }
        gb = GradientBoostingRegressor(random_state=42)
        gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='neg_mean_squared_error', 
                              n_jobs=-1, verbose=0)
        gb_grid.fit(X_train, y_train)
        self.models['Gradient Boosting'] = gb_grid.best_estimator_
        
        # XGBoost
        progress_placeholder.info("🔄 Training XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        }
        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        xgb_grid = GridSearchCV(xgb_model, xgb_params, cv=5, 
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        xgb_grid.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_grid.best_estimator_
        
        # LightGBM
        progress_placeholder.info("🔄 Training LightGBM...")
        lgb_params = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7]
        }
        lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        lgb_grid = GridSearchCV(lgb_model, lgb_params, cv=5, 
                               scoring='neg_mean_squared_error', n_jobs=-1, verbose=0)
        lgb_grid.fit(X_train, y_train)
        self.models['LightGBM'] = lgb_grid.best_estimator_
        
        progress_placeholder.success("✅ All models trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and select the best one"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'RMSE': rmse,
                'MAE': mae,
                'R2 Score': r2,
                'MSE': mse
            }
        
        # Select best model based on R2 score
        best_model_name = max(results, key=lambda x: results[x]['R2 Score'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        return results
    
    def save_model(self, filepath):
        """Save the best model and encoders"""
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load a saved model"""
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.scaler = model_data.get('scaler', StandardScaler())
        except AttributeError as e:
            # Re-raise with more context for scikit-learn version issues
            raise AttributeError(f"Model compatibility error - scikit-learn version mismatch: {str(e)}. Please retrain the model.")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    def predict(self, company, year, kms_driven, fuel_type, name=""):
        """Predict car price for given features"""
        if self.best_model is None:
            raise ValueError("Model not trained or loaded yet!")
        
        # Feature engineering
        current_year = 2024
        car_age = current_year - year
        kms_per_year = kms_driven / (car_age + 1)
        
        # Extract brand from name
        if name:
            brand = str(name).split()[0]
        else:
            brand = company
        
        # Encode categorical variables
        try:
            company_encoded = self.label_encoders['company'].transform([company])[0]
            fuel_encoded = self.label_encoders['fuel'].transform([fuel_type])[0]
            brand_encoded = self.label_encoders['brand'].transform([brand])[0]
        except (KeyError, ValueError) as e:
            company_encoded = 0
            fuel_encoded = 0
            brand_encoded = 0
        
        # Create feature array
        features = np.array([[year, kms_driven, company_encoded, fuel_encoded,
                             car_age, kms_per_year, brand_encoded]])
        
        # Predict
        price = self.best_model.predict(features)[0]
        return max(0, price)


@st.cache_data
def load_data():
    """Load car data"""
    try:
        df = pd.read_csv(DATA_PATH)
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        if len(df.columns) > 0 and df.columns[0] == '':
            df = df.drop(df.columns[0], axis=1)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def load_saved_model():
    """Load saved model from file"""
    try:
        if MODEL_PATH.exists():
            predictor = CarPricePredictor()
            predictor.load_model(MODEL_PATH)
            return predictor
        return None
    except AttributeError as e:
        # Handle scikit-learn version compatibility issues
        st.warning(f"⚠️ Model compatibility issue detected: {str(e)}")
        st.info("💡 **Solution:** Please retrain the model using the '🤖 Train Model' page. The existing model was trained with a different version of scikit-learn.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Error loading model: {str(e)}")
        st.info("💡 **Solution:** Please retrain the model using the '🤖 Train Model' page.")
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Price Prediction System</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Unable to load data. Please check the data file.")
        return
    
    # Initialize predictor in session state
    if 'predictor' not in st.session_state:
        saved_model = load_saved_model()
        if saved_model:
            st.session_state.predictor = saved_model
        else:
            st.session_state.predictor = CarPricePredictor()
    
    # Show info if model failed to load but file exists
    if MODEL_PATH.exists() and st.session_state.predictor.best_model is None:
        with st.expander("⚠️ Model Compatibility Issue - Click to see details", expanded=True):
            st.warning("""
            **Model file exists but couldn't be loaded due to version compatibility issues.**
            
            **Common causes:**
            - The model was trained with a different version of scikit-learn
            - Scikit-learn internal API changes between versions
            
            **Solution:** 
            1. Go to the **🤖 Train Model** page
            2. Click **🚀 Start Training** to create a new compatible model
            
            The old model file will be automatically replaced with a new one.
            """)
            
            # Option to delete old model file
            if st.button("🗑️ Delete Incompatible Model File", help="This will delete the old model file. You'll need to retrain."):
                try:
                    MODEL_PATH.unlink()
                    st.success("✅ Old model file deleted! Please go to '🤖 Train Model' to create a new one.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete model file: {e}")
    
    # Sidebar
    st.sidebar.header("📑 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Home", "🤖 Train Model", "💰 Price Predictor", "📊 Data Analysis", "📈 Model Info"]
    )
    
    if page == "🏠 Home":
        show_home(df, st.session_state.predictor)
    elif page == "🤖 Train Model":
        show_training(df, st.session_state.predictor)
    elif page == "💰 Price Predictor":
        show_predictor(df, st.session_state.predictor)
    elif page == "📊 Data Analysis":
        show_analysis(df)
    elif page == "📈 Model Info":
        show_model_info(st.session_state.predictor)


def show_home(df, predictor):
    """Home page"""
    st.markdown("### Welcome to Advanced Car Price Prediction System! 👋")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cars", f"{len(df):,}")
    
    with col2:
        st.metric("Companies", df['company'].nunique())
    
    with col3:
        st.metric("Avg Price", f"₹{df['Price'].mean():,.0f}")
    
    with col4:
        model_status = "✅ Trained" if predictor.best_model is not None else "❌ Not Trained"
        st.metric("Model Status", model_status)
    
    st.markdown("---")
    
    st.markdown("""
    ### ✨ Key Features:
    - 🤖 **Multi-Model ML System**: Train Random Forest, XGBoost, Gradient Boosting, and LightGBM
    - 💰 **Price Prediction**: Get instant and accurate car price estimates
    - 📊 **Interactive Data Analysis**: Explore car prices, brands, and trends with beautiful visualizations
    - 📈 **Model Comparison**: Evaluate and compare different ML models
    - 💡 **Advanced Feature Engineering**: Age calculation, usage intensity, and brand analysis
    """)
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📈 Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Price Statistics:**")
        st.dataframe(df['Price'].describe(), use_container_width=True)
    
    with col2:
        st.write("**Top 10 Companies:**")
        top_companies = df['company'].value_counts().head(10)
        fig = px.bar(x=top_companies.index, y=top_companies.values,
                    labels={'x': 'Company', 'y': 'Count'},
                    title="Top 10 Car Companies",
                    color=top_companies.values,
                    color_continuous_scale='viridis')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


def show_training(df, predictor):
    """Model training page"""
    st.header("🤖 Train Machine Learning Models")
    
    st.markdown("""
    Train multiple machine learning models to predict car prices. 
    The system will automatically select the best performing model.
    """)
    
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if predictor.best_model is not None:
            if not st.session_state.get('retrain_confirmed', False):
                st.warning("⚠️ A model is already trained. Training again will replace the current model.")
                if st.button("Continue Anyway", key="continue_retrain"):
                    st.session_state.retrain_confirmed = True
                return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Feature engineering
            status_text.text("Step 1/5: Feature Engineering...")
            progress_bar.progress(20)
            df_processed = predictor.feature_engineering(df)
            
            # Step 2: Prepare data
            status_text.text("Step 2/5: Preparing Data...")
            progress_bar.progress(40)
            X = df_processed[predictor.feature_columns].values
            y = df_processed['Price'].values
            
            # Step 3: Split data
            status_text.text("Step 3/5: Splitting Data (80% train, 20% test)...")
            progress_bar.progress(50)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Step 4: Train models
            status_text.text("Step 4/5: Training Models (this may take a few minutes)...")
            progress_bar.progress(60)
            predictor.train_models(X_train, y_train)
            
            # Step 5: Evaluate models
            status_text.text("Step 5/5: Evaluating Models...")
            progress_bar.progress(90)
            results = predictor.evaluate_models(X_test, y_test)
            
            # Save model
            predictor.save_model(MODEL_PATH)
            
            progress_bar.progress(100)
            status_text.text("✅ Training Completed Successfully!")
            st.balloons()
            
            # Display results
            st.success(f"✅ Model training completed! Best model: **{predictor.best_model_name}**")
            
            # Model comparison table
            st.subheader("📊 Model Performance Comparison")
            results_df = pd.DataFrame(results).T
            results_df = results_df.sort_values('R2 Score', ascending=False)
            results_df['R2 Score'] = results_df['R2 Score'].apply(lambda x: f"{x:.4f}")
            results_df['RMSE'] = results_df['RMSE'].apply(lambda x: f"{x:.2f}")
            results_df['MAE'] = results_df['MAE'].apply(lambda x: f"{x:.2f}")
            results_df['MSE'] = results_df['MSE'].apply(lambda x: f"{x:.2f}")
            st.dataframe(results_df, use_container_width=True)
            
            # Best model highlight
            st.subheader("🏆 Best Model Details")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", predictor.best_model_name)
            with col2:
                best_r2 = results[predictor.best_model_name]['R2 Score']
                st.metric("R2 Score", f"{best_r2:.4f}")
            with col3:
                best_rmse = results[predictor.best_model_name]['RMSE']
                st.metric("RMSE", f"{best_rmse:.2f}")
            
            st.session_state.retrain_confirmed = False
            
        except Exception as e:
            st.error(f"Error during training: {e}")
            st.exception(e)
    
    # Training information
    with st.expander("ℹ️ About Model Training"):
        st.markdown("""
        **Training Process:**
        1. **Feature Engineering**: Creates new features (car age, km/year, brand encoding)
        2. **Data Preparation**: Encodes categorical variables and prepares features
        3. **Data Splitting**: Splits data into 80% training and 20% testing sets
        4. **Model Training**: Trains 4 models with hyperparameter tuning:
           - Random Forest (ensemble method)
           - Gradient Boosting (boosting method)
           - XGBoost (gradient boosting framework)
           - LightGBM (lightweight gradient boosting)
        5. **Model Evaluation**: Evaluates all models and selects best based on R2 Score
        
        **Evaluation Metrics:**
        - **R2 Score**: Coefficient of determination (higher is better, max 1.0)
        - **RMSE**: Root Mean Squared Error (lower is better)
        - **MAE**: Mean Absolute Error (lower is better)
        """)


def show_predictor(df, predictor):
    """Price prediction page"""
    st.header("💰 Car Price Predictor")
    
    if predictor.best_model is None:
        st.warning("⚠️ No trained model found. Please train a model first from the 'Train Model' page.")
        return
    
    st.markdown("Enter the details of your car to get an accurate price prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        companies = sorted(df['company'].unique().tolist())
        selected_company = st.selectbox("Select Company", companies)
        
        company_cars = df[df['company'] == selected_company]['name'].unique()
        selected_car = st.selectbox("Select Car Model", sorted(company_cars))
        
        fuel_types = sorted(df['fuel_type'].unique().tolist())
        selected_fuel = st.selectbox("Fuel Type", fuel_types)
    
    with col2:
        current_year = 2024
        selected_year = st.slider("Year", min_value=1995, max_value=current_year, 
                                  value=2015, step=1)
        
        max_kms = int(df['kms_driven'].max())
        selected_kms = st.number_input("Kilometers Driven", min_value=0, 
                                       max_value=max_kms, value=50000, step=1000)
        
        car_age = current_year - selected_year
        if car_age > 0:
            kms_per_year = selected_kms / car_age
            st.info(f"📊 Kilometers per year: {kms_per_year:,.0f} km")
    
    st.markdown("---")
    
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        with st.spinner("Calculating price..."):
            try:
                predicted_price = predictor.predict(
                    selected_company, selected_year, 
                    selected_kms, selected_fuel, selected_car
                )
                
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>Predicted Price</h2>
                    <h1>₹ {predicted_price:,.0f}</h1>
                    <p>Based on {predictor.best_model_name}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show similar cars
                st.markdown("### 📋 Similar Cars in Dataset")
                similar = df[
                    (df['company'] == selected_company) & 
                    (df['year'] >= selected_year - 2) & 
                    (df['year'] <= selected_year + 2) &
                    (df['fuel_type'] == selected_fuel)
                ].head(10)
                
                if len(similar) > 0:
                    similar_display = similar[['name', 'year', 'kms_driven', 'Price']].copy()
                    similar_display['Price'] = similar_display['Price'].apply(
                        lambda x: f"₹ {x:,.0f}"
                    )
                    similar_display.columns = ['Car Name', 'Year', 'Kilometers', 'Price']
                    st.dataframe(similar_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No similar cars found in the dataset.")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")


def show_analysis(df):
    """Data analysis page"""
    st.header("📊 Data Analysis & Visualizations")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_companies = st.multiselect(
            "Filter by Company",
            sorted(df['company'].unique()),
            default=sorted(df['company'].unique())[:5]
        )
    
    with col2:
        min_year, max_year = int(df['year'].min()), int(df['year'].max())
        year_range = st.slider("Year Range", min_year, max_year, (min_year, max_year))
    
    with col3:
        min_price, max_price = int(df['Price'].min()), int(df['Price'].max())
        price_range = st.slider("Price Range (₹)", min_price, max_price, 
                               (min_price, max_price), step=10000)
    
    # Filter data
    filtered_df = df[
        (df['company'].isin(selected_companies)) &
        (df['year'] >= year_range[0]) &
        (df['year'] <= year_range[1]) &
        (df['Price'] >= price_range[0]) &
        (df['Price'] <= price_range[1])
    ]
    
    st.info(f"Showing {len(filtered_df)} cars matching your filters")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Trends", "🏢 By Company", "⛽ By Fuel Type", "📊 Distributions"])
    
    with tab1:
        st.subheader("Price Trends Over Years")
        yearly_stats = filtered_df.groupby('year')['Price'].agg(['mean', 'median', 'count']).reset_index()
        yearly_stats.columns = ['Year', 'Mean Price', 'Median Price', 'Count']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Mean Price'],
                      name="Mean Price", line=dict(color='blue', width=3)),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Median Price'],
                      name="Median Price", line=dict(color='red', width=3)),
            secondary_y=False
        )
        fig.add_trace(
            go.Bar(x=yearly_stats['Year'], y=yearly_stats['Count'],
                  name="Number of Cars", opacity=0.3),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Price (₹)", secondary_y=False)
        fig.update_yaxes(title_text="Count", secondary_y=True)
        fig.update_layout(title="Price Trends and Car Count Over Years", height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Price Analysis by Company")
        company_stats = filtered_df.groupby('company')['Price'].agg(['mean', 'median', 'count']).reset_index()
        company_stats = company_stats.sort_values('mean', ascending=False).head(15)
        
        fig = px.bar(company_stats, x='company', y='mean',
                    labels={'company': 'Company', 'mean': 'Mean Price (₹)'},
                    title="Mean Price by Company (Top 15)",
                    color='mean', color_continuous_scale='viridis')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        company_stats['mean'] = company_stats['mean'].apply(lambda x: f"₹ {x:,.0f}")
        company_stats['median'] = company_stats['median'].apply(lambda x: f"₹ {x:,.0f}")
        company_stats.columns = ['Company', 'Mean Price', 'Median Price', 'Count']
        st.dataframe(company_stats, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Price Analysis by Fuel Type")
        fuel_stats = filtered_df.groupby('fuel_type')['Price'].agg(['mean', 'median', 'count']).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(fuel_stats, values='count', names='fuel_type',
                        title="Distribution by Fuel Type")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(fuel_stats, x='fuel_type', y='mean',
                        labels={'fuel_type': 'Fuel Type', 'mean': 'Mean Price (₹)'},
                        title="Mean Price by Fuel Type",
                        color='fuel_type')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Price Distributions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(filtered_df, x='Price', nbins=50,
                              title="Price Distribution",
                              labels={'Price': 'Price (₹)', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(filtered_df, y='Price', x='fuel_type',
                        title="Price Distribution by Fuel Type",
                        labels={'Price': 'Price (₹)', 'fuel_type': 'Fuel Type'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        numeric_cols = ['year', 'Price', 'kms_driven']
        corr_matrix = filtered_df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Correlation Matrix",
                       labels=dict(color="Correlation"))
        st.plotly_chart(fig, use_container_width=True)


def show_model_info(predictor):
    """Model information page"""
    st.header("📈 Model Information")
    
    if predictor.best_model is None:
        st.warning("⚠️ No trained model found. Please train a model first.")
        return
    
    model_name = predictor.best_model_name
    st.success(f"**Current Model:** {model_name}")
    
    st.markdown("---")
    
    st.subheader("Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Feature Columns")
        feature_cols = predictor.feature_columns
        for i, col in enumerate(feature_cols, 1):
            st.write(f"{i}. {col}")
    
    with col2:
        st.markdown("### Categorical Encoders")
        encoders = predictor.label_encoders
        for encoder_name in encoders.keys():
            num_classes = len(encoders[encoder_name].classes_)
            st.write(f"**{encoder_name.title()}**: {num_classes} categories")
    
    st.markdown("---")
    
    st.subheader("Model Location")
    if MODEL_PATH.exists():
        st.info(f"✅ Model saved at: `{MODEL_PATH}`")
    else:
        st.warning("⚠️ Model file not found on disk")


if __name__ == "__main__":
    main()
