# Power BI Template Creation Walkthrough
## Berlin Airbnb Analysis Dashboard

This guide provides step-by-step instructions to create the Power BI template using all the provided assets.

---

## 🎯 Quick Start Checklist

- [ ] Power BI Desktop installed and updated
- [ ] All template files downloaded to your `/dashboards/template/` folder
- [ ] Data files available in `/data/AirBnB-Berlin/2025-06-20/`
- [ ] 2-3 hours allocated for template creation

---

## Part 1: Project Setup (15 minutes)

### Step 1: Create New Power BI File
1. **Open Power BI Desktop**
2. **File** → **New** 
3. **Save As** → Navigate to `/dashboards/` folder
4. **Name**: `Berlin_Airbnb_Analysis_Template.pbix`

### Step 2: Apply Theme Settings
1. **View** tab → **Themes** → **Customize current theme**
2. **Apply these colors**:
   - Primary: `#FF5A5F` (Airbnb red)
   - Secondary: `#00A699` (Teal)
   - Accent: `#FC642D` (Orange)
   - Background: `#FFFFFF` (White)
   - Text: `#484848` (Dark gray)

---

## Part 2: Data Import and Modeling (30 minutes)

### Step 3: Import Main Dataset
1. **Home** tab → **Get Data** → **Text/CSV**
2. **Browse** to: `data/AirBnB-Berlin/2025-06-20/listings_cleaned.csv`
3. **In Data Preview**:
   - ✅ Verify headers look correct
   - ✅ Check price column shows currency format
   - ✅ Verify coordinates are decimal numbers
4. **Transform Data** → **Power Query Editor opens**

### Step 4: Apply Data Transformations
1. **Copy the M Script** from `PowerQuery_M_Scripts.txt` - Section 1 (Main Listings)
2. **In Power Query Editor**:
   - **Advanced Editor** → **Replace entire query** with the M script
   - **Update the file path** to match your data location
   - **Close & Apply**

### Step 5: Import Supporting Tables
1. **Repeat for neighbourhoods.csv**:
   - Use Section 2 from `PowerQuery_M_Scripts.txt`
   - Update file path accordingly

2. **Repeat for reviews.csv** (optional):
   - Use Section 3 from `PowerQuery_M_Scripts.txt`
   - Update file path accordingly

### Step 6: Create Data Model Relationships
1. **Model View** (left sidebar)
2. **Create relationships**:
   - `listings_cleaned[neighbourhood_group]` ↔ `neighbourhoods[neighbourhood_group]`
   - `listings_cleaned[id]` ↔ `reviews[listing_id]` (if reviews imported)
3. **Verify cardinality**: One-to-Many from neighbourhoods to listings

---

## Part 3: Create DAX Measures (45 minutes)

### Step 7: Organize Measure Folders
1. **Data View** (left sidebar)
2. **Right-click** in the Fields pane → **New measure group**
3. **Create these folders**:
   - 📊 Pricing Measures
   - 📅 Availability Measures  
   - 👤 Host Performance
   - ⭐ Reviews & Ratings
   - 🗺️ Geographic Analysis
   - 🏠 Property Analysis

### Step 8: Import All DAX Measures
1. **Open** `DAX_Measures_Library.txt`
2. **For each measure section**:
   - Copy the DAX formula
   - **New Measure** in Power BI
   - Paste the formula
   - **Move to appropriate folder** (drag and drop)

### Step 9: Validate Key Measures
Test these critical measures work correctly:
- [ ] `[Average Price]` shows reasonable value (~€75-85)
- [ ] `[Total Listings]` matches your data count
- [ ] `[Superhosts Percentage]` shows ~30-40%
- [ ] `[Occupancy Rate]` shows reasonable percentage

---

## Part 4: Create Report Pages (90 minutes)

### Step 10: Page 1 - Executive Summary Dashboard

#### 10.1: Create Page and Layout
1. **New Page** → **Rename** to "Executive Summary"
2. **Format Page**:
   - Background: White
   - Page Size: 16:9 (1280x720)

#### 10.2: Add KPI Cards (Top Row)
Reference: `Visual_Specifications.json` → Executive Summary → First 4 visuals

1. **Insert** → **Card** visual
2. **Position**: Top-left corner
3. **Data**: Drag `Total Listings` measure
4. **Format**:
   - Font size: 36pt
   - Color: #484848
   - Background: Light gray (#F7F7F7)
5. **Duplicate** for remaining 3 KPIs:
   - Average Price (color: #FF5A5F)
   - Average Availability (color: #00A699) 
   - Superhost % (color: #FC642D)

#### 10.3: Add Price Distribution Chart
1. **Insert** → **Column Chart**
2. **Position**: Below KPI cards, left side
3. **Data**:
   - X-axis: `price_category`
   - Y-axis: Count of `id`
4. **Format**:
   - Colors: Airbnb theme colors
   - Data labels: On
   - Title: "Price Distribution by Category"

#### 10.4: Add Berlin Map
1. **Insert** → **Map** visual
2. **Position**: Right side of price chart
3. **Data**:
   - Location: `latitude`, `longitude`
   - Size: `number_of_reviews`
   - Color: `price`
4. **Format**:
   - Map style: Road
   - Zoom: Auto-fit Berlin
   - Bubble colors: Red to Green scale

#### 10.5: Add Room Type Donut Chart
1. **Insert** → **Donut Chart**
2. **Position**: Bottom left
3. **Data**:
   - Legend: `room_type`
   - Values: Count of `id`
4. **Format**:
   - Show percentages
   - Airbnb color scheme

#### 10.6: Add Occupancy Gauge
1. **Insert** → **Gauge** visual
2. **Position**: Bottom center
3. **Data**: `[Occupancy Rate]` measure
4. **Format**:
   - Min: 0, Max: 100
   - Target: 70
   - Color bands: Red(0-30), Yellow(30-70), Green(70-100)

#### 10.7: Add Top Districts Bar Chart
1. **Insert** → **Bar Chart**
2. **Position**: Bottom right
3. **Data**:
   - Y-axis: `neighbourhood_group`
   - X-axis: `[Average Price]`
4. **Format**:
   - Sort: Descending by price
   - Show top 5 only
   - Color: #FF5A5F

### Step 11: Page 2 - Geographic Analysis

#### 11.1: Create Geographic Page
1. **New Page** → **Rename** to "Geographic Analysis"
2. **Reference**: `Visual_Specifications.json` → Geographic Analysis section

#### 11.2: Add Filled Map
1. **Insert** → **Filled Map**
2. **Position**: Large area, top-left
3. **Data**:
   - Location: `neighbourhood_group`
   - Color saturation: `[Average Price]`
4. **Format**:
   - Color scale: Red to Yellow to Green
   - Show data labels
   - Border color: Dark gray

#### 11.3: Add District Summary Table
1. **Insert** → **Table** visual
2. **Position**: Top-right
3. **Columns**:
   - `neighbourhood_group`
   - Count of `id` 
   - `[Average Price]`
   - `[Median Price]`
   - `[Average Availability]`
4. **Format**:
   - Alternating row colors
   - Header: Airbnb red background
   - Currency formatting for prices

#### 11.4: Add Neighborhoods Bar Chart
1. **Insert** → **Bar Chart**
2. **Position**: Bottom-left
3. **Data**:
   - Y-axis: `neighbourhood_cleansed`
   - X-axis: `[Average Price]`
4. **Format**:
   - Sort: Descending
   - Show top 15
   - Color: Teal (#00A699)

#### 11.5: Add Price vs Availability Scatter
1. **Insert** → **Scatter Chart**
2. **Position**: Bottom-right
3. **Data**:
   - X-axis: `availability_365`
   - Y-axis: `price`
   - Legend: `neighbourhood_group`
   - Size: `number_of_reviews`
4. **Format**:
   - Show trend line
   - Multiple colors for districts

### Step 12: Page 3 - Host Analysis

#### 12.1: Create Host Analysis Page
1. **New Page** → **Rename** to "Host Analysis"
2. **Reference**: `Visual_Specifications.json` → Host Analysis section

#### 12.2: Add Superhost Comparison
1. **Insert** → **Clustered Bar Chart**
2. **Position**: Top-left, large area
3. **Data**:
   - Axis: `host_is_superhost`
   - Values: `[Average Price]`, `[Average Rating]`, `[Reviews per Month]`
4. **Format**:
   - Legend at bottom
   - Multiple colors for metrics
   - Data labels on

#### 12.3: Add Host Listing Distribution
1. **Insert** → **Histogram** (or Column Chart)
2. **Position**: Top-right
3. **Data**:
   - Axis: `calculated_host_listings_count`
   - Values: Count of `host_id` (distinct)
4. **Format**:
   - Orange color (#FC642D)
   - Bin size: 1

#### 12.4: Add Response Time Analysis
1. **Insert** → **Line Chart**
2. **Position**: Bottom-left
3. **Data**:
   - Axis: `host_response_time`
   - Values: `[Average Rating]`
4. **Format**:
   - Line: Teal
   - Markers: Red
   - Show markers

#### 12.5: Add Top Hosts Table
1. **Insert** → **Table**
2. **Position**: Bottom-right
3. **Columns**:
   - `host_name`
   - Count of `id`
   - `[Average Price]`
   - `[Estimated Annual Revenue]`
4. **Format**:
   - Sort by revenue (descending)
   - Show top 20
   - Currency formatting

### Step 13: Page 4 - Price Prediction Insights

#### 13.1: Create Prediction Page
1. **New Page** → **Rename** to "Price Prediction Insights"
2. **Reference**: `Visual_Specifications.json` → Price Prediction section

⚠️ **Note**: This page requires ML model results. Create placeholder visuals with sample data.

#### 13.2: Add Feature Importance Chart
1. **Insert** → **Bar Chart**
2. **Position**: Top-left
3. **Manual Data Entry** (based on your notebook results):
   - Room Type: 35%
   - District: 28%
   - Accommodates: 15%
   - Reviews: 12%
   - Availability: 10%
4. **Format**:
   - Horizontal bars
   - Red color (#FF5A5F)
   - Data labels on

#### 13.3: Add Model Performance Cards
1. **Insert** → **Multi-row Card**
2. **Position**: Top-right
3. **Create measures for**:
   - R-Squared: ~0.72
   - MAE: ~€31
   - RMSE: ~€45
   - MAPE: ~24%
4. **Format**:
   - Light gray background
   - Red values

#### 13.4: Add Price Distribution Box Plot
1. **Insert** → **Box Plot** (if available) or **Column Chart**
2. **Position**: Bottom-left
3. **Data**:
   - Category: `room_type`
   - Values: `price`
4. **Format**:
   - Multiple colors for room types
   - Show outliers

#### 13.5: Add Predicted vs Actual Scatter
1. **Insert** → **Scatter Chart**
2. **Position**: Bottom-right
3. **Create calculated column**: `Predicted Price = [Average Price] * (0.85 + RAND() * 0.3)`
4. **Data**:
   - X-axis: `price`
   - Y-axis: `Predicted Price`
   - Size: `number_of_reviews`
5. **Format**:
   - Teal color
   - Trend line (red)
   - Perfect prediction reference line

---

## Part 5: Add Interactivity (30 minutes)

### Step 14: Create Global Slicers

#### 14.1: Add Date Range Slicer
1. **Insert** → **Slicer**
2. **Position**: Top of Executive Summary page
3. **Data**: `last_review`
4. **Format**: Date range slider
5. **Apply to**: All pages except Price Prediction

#### 14.2: Add Price Range Slicer
1. **Insert** → **Slicer**
2. **Position**: Next to date slicer
3. **Data**: `price`
4. **Format**: Range slider (0-500)
5. **Apply to**: Executive Summary, Geographic, Price Prediction

#### 14.3: Add Room Type Filter
1. **Insert** → **Slicer**
2. **Position**: Top row
3. **Data**: `room_type`
4. **Format**: Dropdown, multi-select
5. **Apply to**: All pages

#### 14.4: Add Location Hierarchy
1. **Insert** → **Slicer**
2. **Position**: Top row
3. **Data**: `neighbourhood_group`
4. **Format**: List, single-select
5. **Apply to**: Executive Summary, Geographic, Host Analysis

### Step 15: Configure Cross-Page Filtering
1. **Format** → **Page Information** → **Enable cross-page filtering**
2. **Test**: Click on district in Executive Summary → verify filters apply to other pages
3. **Sync slicers**: Format → Sync slicers → Enable for Price Range, Room Type, Location

### Step 16: Set Up Drill-Through
1. **Geographic Analysis page**:
   - Right-click visual → **Drill through** → **Add drill-through field**: `neighbourhood_group`
2. **Host Analysis page**:
   - Add drill-through field: `host_id`

---

## Part 6: Final Formatting and Testing (20 minutes)

### Step 17: Apply Consistent Formatting

#### 17.1: Format All Page Titles
1. **Select each page title**
2. **Format**:
   - Font: Segoe UI, 24pt
   - Color: #484848
   - Bold
   - Center aligned

#### 17.2: Format All Chart Titles
1. **Select each visual title**
2. **Format**:
   - Font: Segoe UI, 14pt
   - Color: #484848
   - Bold

#### 17.3: Apply Conditional Formatting
1. **Price columns in tables**:
   - Above average: Red background
   - Below average: Green background
2. **Occupancy rate measures**:
   - >70%: Green
   - 50-70%: Yellow  
   - <50%: Red

### Step 18: Test All Functionality
- [ ] All slicers filter correctly
- [ ] Cross-page filtering works
- [ ] Drill-through functions properly
- [ ] All measures calculate correctly
- [ ] No error messages in visuals
- [ ] Mobile layout looks good

### Step 19: Optimize Performance
1. **File** → **Options** → **Data Load**:
   - Reduce number of rows for large tables
   - Disable auto date/time hierarchy if not needed
2. **Limit scatter plot data points** to 5,000-10,000 for performance
3. **Test loading time** should be under 10 seconds

---

## Part 7: Save and Document (10 minutes)

### Step 20: Final Save and Backup
1. **Save** the .pbix file
2. **File** → **Export** → **PDF** (create static backup)
3. **Create folder**: `/dashboards/template/backups/`
4. **Copy .pbix file** to backups folder with timestamp

### Step 21: Create Usage Documentation
1. **Insert** → **Text Box** on Executive Summary page
2. **Add**: 
   - Report creation date
   - Data source information  
   - Last refresh date
   - Key insights summary
3. **Position**: Small text box in bottom corner

### Step 22: Test Template with Fresh Data
1. **Get Data** → **Data source settings**
2. **Change source** to different date folder (if available)
3. **Refresh** → Verify everything updates correctly
4. **This confirms your template is reusable**

---

## 🎉 Congratulations!

You now have a complete Power BI template for Berlin Airbnb analysis!

### Template Features ✅
- 4 comprehensive analysis pages
- 20+ interactive visualizations  
- 40+ calculated DAX measures
- Cross-page filtering and drill-through
- Mobile-optimized layout
- Professional Airbnb-branded theme

### Next Steps:
1. **Share** the template with stakeholders
2. **Set up** automated data refresh (if using Power BI Service)
3. **Create** presentation slides from key visuals
4. **Document** insights and recommendations

---

## 🛠️ Troubleshooting Guide

### Common Issues:

**🔴 Data won't load**
- Check file paths in M scripts
- Verify CSV files aren't corrupted
- Ensure proper permissions to data folder

**🔴 Measures show errors**
- Check column names match exactly
- Verify data types are correct
- Look for null values in key fields

**🔴 Visuals are blank**
- Check field assignments
- Verify filters aren't excluding all data
- Look for relationship issues in model view

**🔴 Performance is slow**
- Reduce data volume with sampling
- Limit complex visuals per page
- Remove unused columns from model

**🔴 Mobile layout looks wrong**
- Check mobile view for each page
- Resize visuals for smaller screens
- Test on actual mobile device

### Getting Help:
- Check Power BI community forums
- Review official Microsoft documentation
- Test with smaller dataset first
- Create measures one at a time to isolate issues

---

**Total Estimated Time: 3.5 hours**
**Difficulty Level: Intermediate**
**Result: Professional Power BI dashboard template ready for production use**