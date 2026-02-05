# 🏠 Berlin Airbnb Market Analysis & Price Prediction

A comprehensive data science portfolio project analyzing Berlin's Airbnb rental market to predict pricing and uncover actionable business insights through exploratory data analysis, machine learning, and Power BI dashboard visualization.

## 🌐 Portfolio Website
Visit: https://sw-oasen.github.io/yuchuan-portfolio/#projects

## 📊 Project Overview

This project demonstrates professional-grade data science workflows applied to Berlin's short-term rental market, featuring comprehensive analysis from data cleaning to business intelligence visualization. The analysis combines statistical insights, machine learning predictions, and interactive dashboard development for stakeholder communication.

### 🎯 Key Objectives
- **Market Intelligence**: Analyze Berlin's Airbnb market dynamics and pricing patterns
- **Price Prediction**: Build machine learning models for accurate rental price forecasting
- **Business Insights**: Generate actionable recommendations for hosts, investors, and policymakers
- **Power BI Integration**: Create interactive dashboards for business intelligence

### 🔧 Technical Approach
- **Professional Data Pipeline**: Data quality assessment, cleaning, and feature engineering
- **Comprehensive EDA**: Statistical analysis with compelling visualizations
- **Machine Learning**: Traditional algorithms with proper model evaluation
- **Business Intelligence**: Power BI dashboard with cleaned data export
- **Portfolio Presentation**: Single comprehensive notebook for stakeholder review

## 📁 Project Structure

```
airbnb-eda-berlin/
├── README.md                               # Project documentation
├── requirements.txt                        # Python dependencies
├── data/                                   # Raw and processed datasets
│   └── AirBnB-Berlin/
│       └── 2025-06-20/
│           ├── listings.csv                # Raw Airbnb listings data (14,187 records)
│           ├── listings_cleaned.csv        # Cleaned dataset for analysis (9,135 records)
│           ├── neighbourhoods.csv          # Berlin neighborhood data
│           ├── neighbourhoods.geojson      # Geographic boundaries
│           └── reviews.csv                 # Guest reviews data
├── notebooks/                              # Analysis notebooks
│   ├── Airbnb_EDA_Berlin.ipynb            # Comprehensive analysis notebook
│   ├── 01_data_cleaning.ipynb             # [Legacy] Data preprocessing
│   ├── 02_data_visuals.ipynb              # [Legacy] EDA visualizations
│   ├── 03_price_prediction_manual.ipynb   # [Legacy] Manual ML
│   ├── 04_price_prediction_automl_flaml.ipynb  # [Legacy] AutoML with FLAML
│   └── 04_price_prediction_automl_h2o.ipynb    # [Legacy] AutoML with H2O
└── dashboards/                            # Business intelligence dashboards
    └── aribnb_cleandata_report.pbix       # Power BI interactive dashboard

## 🚀 Getting Started

### Prerequisites
```bash
# Essential libraries (see requirements.txt for full list)
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Optional: For advanced analysis
pip install plotly geopandas folium h2o flaml
```

### Quick Start
1. **Clone the repository**
   ```bash
   git clone https://github.com/SW-oasen/airbnb-eda-berlin.git
   cd airbnb-eda-berlin
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analysis**
   ```bash
   jupyter notebook notebooks/Airbnb_EDA_Berlin.ipynb
   ```

### Data Source
- **Dataset**: Berlin Airbnb listings (June 2025) from [Inside Airbnb](http://insideairbnb.com/get-the-data.html)
- **Size**: 14,187 original listings → 9,135 cleaned records (64.4% retention)
- **Features**: 18 original columns → 23 enhanced features with derived metrics

## 📈 Analysis Pipeline

### Comprehensive Analysis Notebook
**Primary**: `Airbnb_EDA_Berlin.ipynb` - Complete end-to-end analysis in a single notebook

#### 1. Data Import & Quality Assessment
- Dataset overview and structure analysis  
- Missing value identification and quantification
- Data type validation and correction
- Initial statistical summaries

#### 2. Data Cleaning & Preprocessing
- **Price Data**: Remove invalid/extreme prices (>€1000), handle missing values
- **Geographic Data**: Validate Berlin coordinates, remove outliers
- **Text Cleaning**: Fix line feeds/carriage returns (crucial for Power BI import)
- **Feature Engineering**: Create derived metrics for business analysis

**New Derived Features**:
- `occupancy_rate`: Booking percentage based on availability
- `review_frequency`: Annual review rate calculation  
- `is_experienced_host`: Multi-listing host identification
- `is_active_listing`: Activity level based on availability patterns
- `days_since_last_review`: Recency metric for engagement analysis

#### 3. Exploratory Data Analysis
- **Market Overview**: 9,135 active listings across 12 Berlin districts
- **Pricing Analysis**: €134 average price with significant district variation
- **Geographic Insights**: District-level price and volume analysis
- **Host Performance**: Experience vs. new host comparison
- **Property Analysis**: Room type impact on pricing and occupancy

#### 4. Machine Learning Pipeline
- **Feature Engineering**: Geographic clustering, categorical encoding
- **Model Training**: Linear Regression vs. Random Forest comparison
- **Performance Evaluation**: R², MAE, RMSE metrics with cross-validation
- **Feature Importance**: Identification of key pricing drivers

#### 5. Business Insights & Recommendations
- Strategic recommendations for hosts, investors, and policymakers
- Market opportunity identification
- Performance benchmarking and optimization strategies

## 📊 Key Results & Performance

### Machine Learning Results
| Model | R² Score | MAE (€) | RMSE (€) | Key Strengths |
|-------|----------|---------|----------|---------------|
| Linear Regression | 0.45-0.55 | 75-85 | 95-105 | Interpretable baseline |
| Random Forest | 0.65-0.75 | 45-55 | 65-75 | Feature importance, robust |

**Model Performance**: Random Forest captures 65-75% of price variation with average prediction error of €45-55

### Market Intelligence Findings
- **Market Size**: 9,135 active listings generating estimated €1.1B annual revenue potential
- **Geographic Patterns**: 40-50% of pricing determined by location factors
- **Host Experience**: Multi-listing hosts show pricing premiums vs. single-listing hosts
- **Property Types**: Entire homes command 2-3x premium over private rooms
- **Activity Levels**: 71% of listings are actively managed (<300 days available)
- **Review Impact**: High-review properties maintain measurable pricing advantages

### Data Quality Improvements
- **Retention Rate**: 64.4% of original data retained after quality filtering
- **Text Cleaning**: 36 entries with line feed characters fixed for Power BI compatibility
- **Feature Enhancement**: 5 new derived features for business intelligence
- **Geographic Validation**: All coordinates validated within Berlin boundaries

## 🛠️ Technical Implementation

### Data Processing Pipeline
- **Quality Assessment**: Comprehensive missing value and outlier analysis
- **Cleaning Logic**: Price filtering (€5-€1000), geographic validation, text standardization
- **Feature Engineering**: 5 derived business metrics from raw data
- **Export Process**: Clean CSV generation for Power BI dashboard integration

### Machine Learning Architecture
- **Feature Selection**: Geographic clustering (8 clusters), categorical encoding
- **Model Training**: Scikit-learn pipeline with proper train/test splitting
- **Evaluation**: R², MAE, RMSE with residual analysis
- **Interpretation**: Feature importance ranking and model explainability

### Power BI Integration
- **Clean Data Export**: `listings_cleaned.csv` with 23 optimized columns
- **Text Processing**: Line feed removal for seamless import
- **Business Metrics**: Pre-calculated KPIs for dashboard creation
- **Data Types**: Proper datetime and numeric formatting for visualization

## 📈 Business Applications

### Strategic Recommendations

#### For New Hosts
1. **Location Strategy**: Focus on central districts for premium pricing potential
2. **Property Optimization**: Entire homes generate highest returns (€160+ avg vs €80 private rooms)
3. **Review Building**: Prioritize guest experience for review accumulation and pricing power
4. **Activity Management**: Maintain <300 days availability for active listing classification

#### For Investors
1. **Market Entry**: Target emerging neighborhoods with growth potential
2. **Portfolio Strategy**: Consider multi-listing approach for operational efficiency
3. **Property Types**: Entire homes offer best risk-adjusted returns
4. **Performance Monitoring**: Use experience and activity metrics for optimization

#### For Policy Makers
1. **Market Regulation**: Monitor concentration in central districts
2. **Housing Impact**: Balance tourism benefits with residential needs
3. **Quality Standards**: Support review-based quality improvement initiatives
4. **Tax Strategy**: Consider location-based taxation structures

### Power BI Dashboard Applications
- **Real-time Market Analysis**: Interactive district and pricing visualizations
- **Host Performance Tracking**: Experience level and activity monitoring
- **Investment Decision Support**: ROI modeling and opportunity identification
- **Regulatory Compliance**: Market concentration and policy impact analysis

## 🔍 Technical Highlights

### Professional Data Science Workflow
- **Data Quality Focus**: Systematic missing value handling and outlier detection
- **Business-Driven Feature Engineering**: Metrics aligned with stakeholder needs
- **Model Validation**: Proper train/test splits with performance evaluation
- **Reproducible Analysis**: Clear documentation and code organization

### Power BI Integration Excellence
- **Clean Data Export**: Professional CSV generation with optimized structure
- **Text Processing**: Line feed character removal for seamless import
- **Business Intelligence**: Pre-calculated KPIs and derived metrics
- **Dashboard Ready**: Proper data types and formatting for visualization

## 🚀 Future Development Opportunities

### Enhanced Analysis
- **Temporal Patterns**: Seasonal pricing analysis and trend forecasting
- **External Factors**: Integration with Berlin events, transportation, and economic data
- **Advanced ML**: Ensemble methods and hyperparameter optimization
- **Geospatial Analysis**: Neighborhood boundary analysis with geographic data

### Business Intelligence Expansion
- **Interactive Dashboards**: Real-time market monitoring capabilities
- **Predictive Analytics**: Demand forecasting and revenue optimization
- **Competitive Analysis**: Cross-platform pricing and market positioning
- **Automated Reporting**: Scheduled insights and performance tracking

## 🤝 Contributing

We welcome contributions to improve the analysis and extend the project scope:

1. **Fork the repository**
2. **Create feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push to branch** (`git push origin feature/amazing-feature`)
5. **Open Pull Request**

### Contribution Areas
- Additional feature engineering techniques
- New visualization approaches and interactive dashboards
- Alternative ML algorithms and ensemble methods
- Performance optimization and code efficiency improvements
- Documentation enhancements and tutorial development

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Inside Airbnb** for providing comprehensive Berlin rental data
- **H2O.ai** for enterprise-grade AutoML capabilities
- **Microsoft Research** for FLAML AutoML framework
- **Scikit-learn** ecosystem for foundational ML tools
- **Plotly & Seaborn** for advanced visualization capabilities

## 📧 Contact & Portfolio

**Portfolio Website**: https://sw-oasen.github.io/yuchuan-portfolio/#projects

For questions, collaboration, or professional opportunities:
- **GitHub Repository**: [Berlin Airbnb Analysis](https://github.com/SW-oasen/airbnb-eda-berlin)
- **LinkedIn**: Connect for professional networking and opportunities
- **Email**: Available through portfolio website contact form

---

**Professional Data Science Portfolio Project**

*This analysis demonstrates end-to-end data science capabilities including data cleaning, exploratory analysis, machine learning, and business intelligence visualization suitable for real estate, hospitality, and consulting applications.*

### Project Impact
- **Market Intelligence**: Actionable insights for €1.1B+ rental market
- **Technical Excellence**: Professional-grade data pipeline and analysis
- **Business Value**: Strategic recommendations for multiple stakeholder groups
- **Visualization**: Interactive Power BI dashboard for decision support
