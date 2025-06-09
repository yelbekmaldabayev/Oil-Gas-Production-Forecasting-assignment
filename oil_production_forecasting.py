#!/usr/bin/env python3
"""
Oil Production Forecasting using Decline Curve Analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')


class ProductionForecaster:

    def __init__(self, csv_file_path):
        self.data = self.load_and_process_data(csv_file_path)
        self.wells = self.data['Well_Name'].unique()
        self.fitted_models = {}
        self.forecasts = {}

    def load_and_process_data(self, csv_file_path):
        try:
            # Read CSV with semicolon delimiter
            data = pd.read_csv(csv_file_path, delimiter=';')

            # Convert European decimal format
            data['Monthly_Production'] = data['Monthly_Production'].str.replace(
                ',', '.').astype(float)

            # Parse dates
            data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')

            # Sort data
            data = data.sort_values(
                ['Well_Name', 'Date']).reset_index(drop=True)

            # Add time indices
            processed_data = []
            for well in data['Well_Name'].unique():
                well_data = data[data['Well_Name'] == well].copy()
                well_data['Time_Index'] = range(1, len(well_data) + 1)
                processed_data.append(well_data)

            return pd.concat(processed_data, ignore_index=True)

        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    @staticmethod
    def exponential_decline(t, qi, d):
        # Qt = Qi * exp(-D*t)
        return qi * np.exp(-d * t)

    @staticmethod
    def harmonic_decline(t, qi, d):
        # Qt = Qi / (1 + D*t)
        return qi / (1 + d * t)

    @staticmethod
    def hyperbolic_decline(t, qi, d, b):
        # Qt = Qi / (1 + b*D*t)^(1/b)
        return qi / (1 + b * d * t) ** (1/b)

    def fit_decline_curves(self, well_name):
        well_data = self.data[self.data['Well_Name'] == well_name].copy()
        t = well_data['Time_Index'].values
        production = well_data['Monthly_Production'].values

        models = {}
        qi_estimate = production[0]

        # Exponential fit
        try:
            popt_exp, _ = curve_fit(
                self.exponential_decline,
                t,
                production,
                p0=[qi_estimate, 0.05],
                bounds=([0, 0], [np.inf, 1]),
                maxfev=5000
            )
            qi_exp, d_exp = popt_exp
            pred_exp = self.exponential_decline(t, qi_exp, d_exp)
            rmse_exp = np.sqrt(np.mean((production - pred_exp) ** 2))

            models['Exponential'] = {
                'parameters': {'qi': qi_exp, 'd': d_exp},
                'predictions': pred_exp,
                'rmse': rmse_exp,
                'r_squared': 1 - np.sum((production - pred_exp) ** 2) / np.sum((production - np.mean(production)) ** 2)
            }
        except Exception:
            models['Exponential'] = None

        # Harmonic fit
        try:
            popt_harm, _ = curve_fit(
                self.harmonic_decline,
                t,
                production,
                p0=[qi_estimate, 0.05],
                bounds=([0, 0], [np.inf, 1]),
                maxfev=5000
            )
            qi_harm, d_harm = popt_harm
            pred_harm = self.harmonic_decline(t, qi_harm, d_harm)
            rmse_harm = np.sqrt(np.mean((production - pred_harm) ** 2))

            models['Harmonic'] = {
                'parameters': {'qi': qi_harm, 'd': d_harm},
                'predictions': pred_harm,
                'rmse': rmse_harm,
                'r_squared': 1 - np.sum((production - pred_harm) ** 2) / np.sum((production - np.mean(production)) ** 2)
            }
        except Exception:
            models['Harmonic'] = None

        # Hyperbolic fit
        try:
            popt_hyp, _ = curve_fit(
                self.hyperbolic_decline,
                t,
                production,
                p0=[qi_estimate, 0.05, 0.5],
                bounds=([0, 0, 0.01], [np.inf, 1, 0.99]),
                maxfev=5000
            )
            qi_hyp, d_hyp, b_hyp = popt_hyp
            pred_hyp = self.hyperbolic_decline(t, qi_hyp, d_hyp, b_hyp)
            rmse_hyp = np.sqrt(np.mean((production - pred_hyp) ** 2))

            models['Hyperbolic'] = {
                'parameters': {'qi': qi_hyp, 'd': d_hyp, 'b': b_hyp},
                'predictions': pred_hyp,
                'rmse': rmse_hyp,
                'r_squared': 1 - np.sum((production - pred_hyp) ** 2) / np.sum((production - np.mean(production)) ** 2)
            }
        except Exception:
            models['Hyperbolic'] = None

        return models

    def select_best_model(self, models):
        # Best model by RMSE
        valid_models = {name: model for name,
                        model in models.items() if model is not None}

        if not valid_models:
            raise Exception("No valid models found")

        best_model_name = min(valid_models.keys(),
                              key=lambda x: valid_models[x]['rmse'])
        return best_model_name, valid_models[best_model_name]

    def forecast_production(self, well_name, months_ahead=12):
        if well_name not in self.fitted_models:
            self.fitted_models[well_name] = self.fit_decline_curves(well_name)

        models = self.fitted_models[well_name]
        best_model_name, best_model = self.select_best_model(models)

        # Get last data point
        well_data = self.data[self.data['Well_Name'] == well_name]
        last_date = well_data['Date'].max()
        last_time_index = well_data['Time_Index'].max()

        # Future time points
        future_time_indices = np.arange(
            last_time_index + 1, last_time_index + months_ahead + 1)

        # Future dates
        future_dates = []
        current_date = last_date
        for _ in range(months_ahead):
            current_date = current_date + relativedelta(months=1)
            future_dates.append(current_date)

        # Forecast using best model
        params = best_model['parameters']

        if best_model_name == 'Exponential':
            future_production = self.exponential_decline(
                future_time_indices, params['qi'], params['d'])
        elif best_model_name == 'Harmonic':
            future_production = self.harmonic_decline(
                future_time_indices, params['qi'], params['d'])
        elif best_model_name == 'Hyperbolic':
            future_production = self.hyperbolic_decline(
                future_time_indices, params['qi'], params['d'], params['b'])

        return {
            'well_name': well_name,
            'best_model': best_model_name,
            'model_parameters': params,
            'model_rmse': best_model['rmse'],
            'model_r_squared': best_model['r_squared'],
            'forecast_dates': future_dates,
            'forecast_production': future_production,
            'time_indices': future_time_indices
        }

    def run_analysis(self, months_ahead=12):
        print("=" * 60)
        print("OIL PRODUCTION FORECASTING ANALYSIS")
        print("=" * 60)

        for well in self.wells:
            print(f"\nAnalyzing {well}...")

            # Fit models
            models = self.fit_decline_curves(well)
            self.fitted_models[well] = models

            # Best model
            best_model_name, best_model = self.select_best_model(models)

            # Forecast
            forecast = self.forecast_production(well, months_ahead)
            self.forecasts[well] = forecast

            # Results
            print(f"Best Model: {best_model_name}")
            print(f"RMSE: {best_model['rmse']:.2f}")
            print(f"R²: {best_model['r_squared']:.4f}")
            print(f"Parameters: {best_model['parameters']}")

    def create_plots(self, save_plots=True):
        plt.style.use('seaborn-v0_8')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        # Individual well plots
        for i, well in enumerate(self.wells):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            well_data = self.data[self.data['Well_Name'] == well]
            models = self.fitted_models[well]
            forecast = self.forecasts[well]
            best_model_name = forecast['best_model']

            # Historical with fits
            ax1.scatter(well_data['Time_Index'], well_data['Monthly_Production'],
                        color=colors[i], alpha=0.7, s=50, label='Historical Data', zorder=5)

            for model_name, model_data in models.items():
                if model_data is not None:
                    linestyle = '-' if model_name == best_model_name else '--'
                    alpha = 1.0 if model_name == best_model_name else 0.6
                    ax1.plot(well_data['Time_Index'], model_data['predictions'],
                             linestyle=linestyle, alpha=alpha, linewidth=2,
                             label=f"{model_name} (RMSE: {model_data['rmse']:.1f})")

            ax1.set_xlabel('Time (Months)')
            ax1.set_ylabel('Monthly Production')
            ax1.set_title(f'{well} - Historical Data & Model Fits')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Historical + Forecast
            ax2.plot(well_data['Time_Index'], well_data['Monthly_Production'],
                     'o-', color=colors[i], markersize=4, label='Historical Data')

            ax2.plot(forecast['time_indices'], forecast['forecast_production'],
                     's-', color='red', markersize=4, label=f'Forecast ({best_model_name})')

            ax2.axvline(x=well_data['Time_Index'].max(), color='gray',
                        linestyle=':', alpha=0.7, label='Forecast Start')

            ax2.set_xlabel('Time (Months)')
            ax2.set_ylabel('Monthly Production')
            ax2.set_title(f'{well} - Historical Data & 12-Month Forecast')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_plots:
                plt.savefig(f'{well}_analysis.png',
                            dpi=300, bbox_inches='tight')

            plt.show()

        # Comparison plot
        fig, ax = plt.subplots(figsize=(15, 8))

        for i, well in enumerate(self.wells):
            well_data = self.data[self.data['Well_Name'] == well]
            forecast = self.forecasts[well]

            ax.plot(well_data['Time_Index'], well_data['Monthly_Production'],
                    'o-', color=colors[i], markersize=4, alpha=0.8, label=f'{well} (Historical)')

            ax.plot(forecast['time_indices'], forecast['forecast_production'],
                    's--', color=colors[i], markersize=4, alpha=0.6)

        ax.axvline(x=24, color='black', linestyle=':',
                   alpha=0.7, label='Forecast Start')
        ax.set_xlabel('Time (Months)')
        ax.set_ylabel('Monthly Production')
        ax.set_title('All Wells - Production History & Forecasts Comparison')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plots:
            plt.savefig('all_wells_comparison.png',
                        dpi=300, bbox_inches='tight')

        plt.show()

    def export_results(self, filename_prefix='production_forecast'):
        # Summary data
        forecast_summary = []
        detailed_forecast = []

        for well in self.wells:
            forecast = self.forecasts[well]

            forecast_summary.append({
                'Well_Name': well,
                'Best_Model': forecast['best_model'],
                'Model_RMSE': forecast['model_rmse'],
                'Model_R_Squared': forecast['model_r_squared'],
                'Parameters': str(forecast['model_parameters']),
                'Total_12_Month_Forecast': forecast['forecast_production'].sum(),
                'Average_Monthly_Forecast': forecast['forecast_production'].mean()
            })

            # Monthly details
            for i, (date, production) in enumerate(zip(forecast['forecast_dates'],
                                                       forecast['forecast_production'])):
                detailed_forecast.append({
                    'Well_Name': well,
                    'Forecast_Month': i + 1,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Forecasted_Production': production,
                    'Model_Used': forecast['best_model']
                })

        # Export
        summary_df = pd.DataFrame(forecast_summary)
        detailed_df = pd.DataFrame(detailed_forecast)

        summary_df.to_csv(f'{filename_prefix}_summary.csv', index=False)
        detailed_df.to_csv(f'{filename_prefix}_detailed.csv', index=False)

        with pd.ExcelWriter(f'{filename_prefix}_complete.xlsx', engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            detailed_df.to_excel(
                writer, sheet_name='Detailed_Forecast', index=False)
            self.data.to_excel(
                writer, sheet_name='Historical_Data', index=False)

        print(f"\nResults exported to:")
        print(f"- {filename_prefix}_summary.csv")
        print(f"- {filename_prefix}_detailed.csv")
        print(f"- {filename_prefix}_complete.xlsx")

        return summary_df, detailed_df

    def print_summary_report(self):
        print("\n" + "=" * 80)
        print("PRODUCTION FORECASTING SUMMARY REPORT")
        print("=" * 80)

        print(f"\nData Summary:")
        print(f"- Wells analyzed: {len(self.wells)}")
        print(
            f"- Historical period: {len(self.data[self.data['Well_Name'] == self.wells[0]])} months")
        print(f"- Forecast period: 12 months")

        print(f"\nModel Performance:")
        print("-" * 40)

        for well in self.wells:
            forecast = self.forecasts[well]
            models = self.fitted_models[well]

            print(f"\n{well}:")
            print(f"  Best Model: {forecast['best_model']}")
            print(f"  RMSE: {forecast['model_rmse']:.2f}")
            print(f"  R²: {forecast['model_r_squared']:.4f}")

            rmse_comparison = []
            for model_name, model_data in models.items():
                if model_data is not None:
                    rmse_comparison.append(
                        f"{model_name}: {model_data['rmse']:.2f}")
            print(f"  All RMSE: {', '.join(rmse_comparison)}")

            print(
                f"  12-Month Total: {forecast['forecast_production'].sum():.1f}")
            print(
                f"  Monthly Average: {forecast['forecast_production'].mean():.1f}")


def main():
    try:
        print("Initializing Production Forecaster...")
        forecaster = ProductionForecaster('well_prod_data.csv')

        # Run analysis
        forecaster.run_analysis(months_ahead=12)

        # Summary report
        forecaster.print_summary_report()

        # Generate plots
        print("\nGenerating plots...")
        forecaster.create_plots(save_plots=True)

        # Export results
        print("\nExporting results...")
        summary_df, detailed_df = forecaster.export_results()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("Check generated files for detailed results.")
        print("=" * 60)

        return forecaster, summary_df, detailed_df

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    forecaster, summary_df, detailed_df = main()
