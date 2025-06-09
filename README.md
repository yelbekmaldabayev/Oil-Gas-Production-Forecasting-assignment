# Oil Production Forecasting System

## Overview

Python application for oil production forecasting using decline curve analysis. Fits exponential, harmonic, and hyperbolic decline curves to historical data and generates 12-month production forecasts.

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scipy openpyxl python-dateutil
```

## Input Data Format

CSV file with semicolon delimiter:
- **Columns**: Date, Well_Name, Monthly_Production
- **Date Format**: DD.MM.YYYY
- **Production**: European decimal format (comma separator)

Example:
```
Date;Well_Name;Monthly_Production
30.06.2023;Well_1;1040,199291
31.07.2023;Well_1;934,2930525
```

## Decline Curve Models

### Exponential Decline
```
Qt = Qi × e^(-D×t)
```

### Harmonic Decline
```
Qt = Qi / (1 + D×t)
```

### Hyperbolic Decline
```
Qt = Qi / (1 + b×D×t)^(1/b)
```

Where:
- Qt: Production at time t
- Qi: Initial production rate
- D: Decline rate
- b: Hyperbolic exponent

## Usage

### Basic Usage
```python
from oil_production_forecasting import ProductionForecaster

forecaster = ProductionForecaster('well_prod_data.csv')
forecaster.run_analysis(months_ahead=12)
forecaster.create_plots(save_plots=True)
summary_df, detailed_df = forecaster.export_results()
forecaster.print_summary_report()
```

### Command Line
```bash
python oil_production_forecasting.py
```

## Output Files

### Generated Files
- `Well_X_analysis.png` - Individual well plots
- `all_wells_comparison.png` - Comparative analysis
- `production_forecast_summary.csv` - Summary results
- `production_forecast_detailed.csv` - Monthly forecasts
- `production_forecast_complete.xlsx` - Complete workbook

### Excel Workbook Sheets
- Summary: Model performance and totals
- Detailed_Forecast: Monthly production forecasts
- Historical_Data: Original input data

## Algorithm Workflow

1. **Data Processing**: Parse CSV, convert formats, create time indices
2. **Model Fitting**: Fit all decline curves using scipy.optimize
3. **Model Selection**: Select best model by RMSE
4. **Forecasting**: Generate 12-month forecasts
5. **Visualization**: Create plots and charts
6. **Export**: Save results in multiple formats

## Performance Metrics

- **RMSE**: Root Mean Square Error
- **R²**: Coefficient of determination

Best model selected automatically based on lowest RMSE.

## Key Features

- Automated model selection
- Robust error handling
- Multiple export formats
- Comprehensive visualization
- European data format support

## Assumptions

- Monthly production data (no gaps)
- Conventional decline behavior
- No operational interventions
- Time-based decline only

## Example Output

```
Well_1:
  Best Model: Exponential
  RMSE: 45.67
  R²: 0.8932
  12-Month Total: 7,234.5
  Monthly Average: 602.9
```

## Error Handling

- Invalid data formats
- Model fitting failures
- Missing files
- Corrupted data

Failed fits are reported but don't stop analysis.

## Limitations

- Single variable (time) models
- No external factors considered
- Accuracy decreases with forecast horizon
- Assumes consistent operating conditions

## File Structure

```
project/
├── oil_production_forecasting.py
├── well_prod_data.csv
├── README.md
└── output/
    ├── *.png (plots)
    ├── *.csv (results)
    └── *.xlsx (workbook)
```

## Troubleshooting

### Common Issues
- **Import errors**: Check package installation
- **File not found**: Verify CSV file location
- **Date parsing**: Check DD.MM.YYYY format
- **Decimal format**: Ensure comma separators

### Solutions
- Verify all packages installed
- Check file paths and formats
- Review console error messages
- Validate input data structure
