# WorldScreener

![WorldScreener Logo](https://via.placeholder.com/800x200?text=WorldScreener:+Value+Stock+Screening+Tool)

**A comprehensive stock screening tool for value investors focused on European and US markets.**

## Overview

WorldScreener is a powerful, modular Python package designed to help value investors identify undervalued companies across European and US markets. It provides tools for screening, analyzing, and visualizing stocks based on proven value investing principles.

With WorldScreener, you can:

- Screen stocks across multiple regions (Spain, Europe, Eurozone, US)
- Apply customizable value criteria (P/E, dividend yield, P/B ratio, ROE, etc.)
- Analyze individual stocks in-depth
- Compare multiple stocks side-by-side
- Visualize screening results with various charts
- Generate reports in multiple formats (text, CSV, JSON, HTML, Excel)
- Build and analyze value portfolios
- Compare stocks across different regions with the new comparative analysis feature

## Installation

### Using pip

```bash
pip install worldscreener
```

### From source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/worldscreener.git
cd worldscreener
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install in development mode:
```bash
pip install -e .
```

## Quick Start

### Command Line Interface

The command-line interface is provided through the `run.py` script, which offers several subcommands:

#### Screening Stocks

```bash
# Screen European value stocks
python run.py screen --region europe --output html --output-file results.html

# Screen Spanish stocks with custom criteria
python run.py screen --region spain --max-pe 12 --min-div 4 --min-roe 12

# Screen stocks with visualization
python run.py screen --region eurozone --visualize sector_breakdown --plot-file sectors.png
```

#### Analyzing Stocks

```bash
# Analyze a specific stock
python run.py analyze SAN.MC --plot --plot-file santander.png

# Compare multiple stocks
python run.py compare SAN.MC BBVA.MC BNP.PA --plot --normalize
```

#### Comparative Analysis

```bash
# Compare stocks across different regions
python run.py comparative --regions europe us --top 10 --output-file comparative_report.html

# Compare specific sectors across regions
python run.py comparative --regions spain europe us --sector Banking --output-file banking_comparison.html
```

#### Portfolio Creation

```bash
# Create a diversified value portfolio
python run.py portfolio --regions spain europe us --stocks-per-region 5

# Create a portfolio with detailed analysis of top stocks
python run.py portfolio --regions europe us --analyze-top 5 --output-dir portfolio_output
```

#### Sector Analysis

```bash
# Analyze Banking sector in Europe
python run.py sector Banking --region europe --visualize --output-file banking_report.html

# Analyze Technology sector in US
python run.py sector Technology --region us --visualize
```

#### Running Demonstrations

```bash
# Run all demonstrations
python run.py demo

# Run specific demonstration
python run.py demo --demo-num 3  # Run portfolio creation demo
```

### Python API

```python
from src import create_screener

# Initialize the components
data_provider, screener, analyzer, visualizer, report_generator = create_screener()

# Screen stocks
results = screener.screen_stocks(region='europe', limit=20)

# Apply custom criteria
custom_criteria = {
    'max_pe_ratio': 12.0,
    'min_dividend_yield': 4.0,
    'max_pb_ratio': 1.2,
    'min_roe': 12.0,
    'max_debt_to_equity': 0.8
}
results = screener.screen_stocks(region='spain', criteria=custom_criteria)

# Analyze a stock
analysis = analyzer.analyze_stock('SAN.MC')

# Create visualizations
visualizer.create_value_comparison_chart(results, output_file='chart.png')

# Generate reports
report_generator.generate_report(results, output_format='html', output_file='report.html')

# Generate comparative report
regions_data = {
    'europe': screener.screen_stocks(region='europe', limit=10),
    'us': screener.screen_stocks(region='us', limit=10)
}
report_generator.generate_comparative_report(regions_data, output_file='comparative_report.html')
```

## API Rate Limits and Mock Data

WorldScreener uses Yahoo Finance (yfinance 0.2.58+) for retrieving stock data. Yahoo Finance has API rate limits that may cause HTTP 429 errors ("Too Many Requests") during heavy usage.

To address this limitation, WorldScreener includes a mock data generation feature that allows you to:

1. Run demonstrations without hitting API rate limits
2. Test the application's functionality offline
3. Develop and debug without depending on external API availability

### Using Mock Data

By default, the demo script uses mock data to avoid API rate limits:

```bash
# Run with mock data (default)
python notebooks/demo.py

# Run with real data (may hit API rate limits)
python notebooks/demo.py --use-real-data
```

When using the library programmatically, you can control this behavior:

```python
from src import create_screener

# Create a screener with mock data (default)
screener = create_screener(use_mock_data=True)

# Create a screener with real data
screener = create_screener(use_mock_data=False)
```

The mock data generator creates realistic stock data that passes screening criteria, allowing you to test and demonstrate the application's features without API limitations.

## Demo Script

For a comprehensive demonstration of WorldScreener's capabilities, run the demo script:

```bash
python demo.py
```

The demo script showcases:
1. Finding value stocks in Spain
2. Custom screening across regions
3. Value portfolio creation and analysis
4. Sector-specific analysis
5. Comparative analysis across different markets

All demo outputs are saved to the `output` directory.

## Features

- **Multi-region support**: Screen stocks from Spain, Europe, Eurozone, US, or globally
- **Comprehensive value metrics**: Filter by P/E ratio, dividend yield, P/B ratio, ROE, debt/equity, and more
- **Composite value score**: Proprietary algorithm that ranks stocks based on multiple value criteria
- **Detailed stock analysis**: In-depth analysis of individual stocks with key fundamental metrics
- **Advanced visualizations**: Generate various charts to visualize screening results
- **Multi-format reporting**: Generate reports in text, CSV, JSON, HTML, or Excel formats
- **Portfolio creation**: Build and analyze diversified value portfolios
- **Comparative analysis**: Compare stocks across different regions to identify global opportunities
- **API integration**: Uses yfinance API with option to integrate other data sources
- **Caching system**: Efficient data caching to minimize API calls and improve performance
- **Highly configurable**: Customize screening criteria, value score weights, and more

## Project Structure

```
worldscreener/
│
├── src/                          # Source code
│   ├── __init__.py               # Package initialization
│   ├── data_provider.py          # API connectors and data retrieval
│   ├── screener.py               # Core screening logic
│   ├── analyzer.py               # Stock analysis functions
│   ├── visualizer.py             # Plotting and visualization
│   ├── report_generator.py       # Report generation (HTML, CSV, etc.)
│   └── utils.py                  # Utility functions
│
├── notebooks/                    # Jupyter notebooks
│   └── demo.ipynb                # Demo notebook
│
├── config/                       # Configuration files
│   ├── default_config.yaml       # Default screening criteria
│   └── markets_config.yaml       # Market indices and country information
│
├── templates/                    # HTML templates
│   ├── report_template.html      # Basic report template
│   ├── portfolio_template.html   # Portfolio summary template
│   ├── analysis_template.html    # Stock analysis template
│   ├── comparative_template.html # Comparative analysis template
│   └── sector_template.html      # Sector analysis template
│
├── run.py                        # Command-line interface
├── demo.py                       # Demonstration script
├── setup.py                      # Package installation script
└── README.md                     # Project documentation
```

## Customization

WorldScreener can be customized in various ways:

1. **Configuration files**: Edit the YAML files in the `config` directory to customize screening criteria, value score weights, etc.

2. **API sources**: Extend the `DataProvider` class to integrate alternative financial data APIs

3. **Screening criteria**: Create your own custom criteria for stock screening

4. **Report templates**: Modify the HTML templates in the `templates` directory to customize report appearance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all the contributors who have helped to improve WorldScreener
- Inspired by value investing principles of Benjamin Graham and Warren Buffett