# Trustworthy Model Reuse CLI - Milestone 1

ACME Corporation Phase 1 Implementation by Team 8

## Overview

This project implements a CLI tool for evaluating the trustworthiness of machine learning models from Hugging Face and GitHub repositories. This is **Milestone 1** focusing on the abstract MetricCalculator base class and general project structure.

## Milestone 1 Deliverables (Aakash's Part)

✅ **Abstract MetricCalculator Base Class**: Implemented with proper interface for all metric calculators  
✅ **General Project Structure**: Organized codebase following UML class diagram specifications  
✅ **Results Storage System**: Central storage for metric calculations with NDJSON output support  
✅ **Exception Handling**: Consistent error handling across the system  
✅ **Unit Tests**: Comprehensive test suite for validation  

## Project Structure

\`\`\`
src/
├── __init__.py              # Package initialization
├── metric_calculator.py     # Abstract MetricCalculator base class
├── results_storage.py       # Results storage and NDJSON output
└── exceptions.py           # Custom exception classes

tests/
├── test_metric_calculator.py  # Unit tests for MetricCalculator
└── test_results_storage.py    # Unit tests for ResultsStorage

run_tests.py                 # Test runner with coverage reporting
requirements.txt             # Python dependencies
README.md                   # This file
\`\`\`

## Architecture Overview

### MetricCalculator Abstract Base Class

The `MetricCalculator` class serves as the foundation for all metric calculations:

- **Abstract Interface**: Defines `calculate_score()` method that all concrete calculators must implement
- **Score Validation**: Ensures all scores are in the range [0, 1] as required
- **Timing Tracking**: Records calculation time in milliseconds for latency reporting
- **State Management**: Provides reset functionality for reuse across multiple models

### Concrete Implementations (Future Milestones)

The abstract class is designed to support these concrete calculators:
- `SizeCalculator` - Model artifact size evaluation
- `LicenseCalculator` - License compatibility checking  
- `RampUpTimeCalculator` - Documentation and usability analysis
- `BusFactorCalculator` - Contributor diversity assessment
- `DatasetCodeScoreCalculator` - Dataset and code availability
- `DatasetQualityCalculator` - Training data quality evaluation
- `CodeQualityCalculator` - Code quality metrics
- `PerformanceClaimsCalculator` - Benchmark validation

### Results Storage System

The `ResultsStorage` class manages all calculated metrics:

- **Thread-Safe Storage**: Supports parallel metric calculation
- **NDJSON Output**: Formats results according to specification requirements
- **Completion Tracking**: Monitors when all metrics are calculated for a model
- **Aggregation Support**: Prepares data for NetScore calculation

## Testing

### Running Tests

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Run tests with coverage
python run_tests.py
\`\`\`

### Test Coverage

The test suite includes:
- **Abstract class validation**: Ensures proper inheritance patterns
- **Score validation**: Tests range checking and error handling  
- **Results storage**: Validates storage, retrieval, and NDJSON formatting
- **Exception handling**: Tests custom exception behavior

Expected output format: `X/Y test cases passed. Z% line coverage achieved.`

## Design Principles

### Extensibility (Sarah's Future Requirements)

The abstract base class design enables easy addition of new metrics:

1. **Inherit from MetricCalculator**: New metrics extend the base class
2. **Implement calculate_score()**: Define metric-specific calculation logic
3. **Use ModelContext**: Access all necessary model information
4. **Automatic Integration**: Results automatically integrate with existing storage and output systems

### Modularity

- **Separation of Concerns**: Each class has a single, well-defined responsibility
- **Dependency Injection**: Components accept dependencies through constructors
- **Interface Segregation**: Minimal, focused interfaces for each component

### Error Handling Consistency

All exceptions follow a consistent naming pattern:
- `MetricCalculationException` - Metric calculation failures
- `APIRateLimitException` - API quota exceeded
- `InvalidURLException` - URL processing errors
- `ConfigurationException` - System configuration issues

## Integration with UML Diagrams

This implementation strictly follows the provided UML diagrams:

### Class Diagram Compliance
- ✅ Abstract `MetricCalculator` base class
- ✅ `ModelContext` for passing model information
- ✅ `ResultsStorage` for centralized result management
- ✅ Exception hierarchy for error handling

### Activity Diagram Alignment
- ✅ Supports parallel metric calculation workflow
- ✅ Enables context linking between related URLs
- ✅ Provides foundation for rate limiting integration
- ✅ Supports NDJSON output format requirements

## Next Steps (Future Milestones)

1. **Milestone 2**: Implement concrete metric calculators (License, Size, etc.)
2. **Milestone 3**: Add URL processing and API integration
3. **Milestone 4**: Complete CLI interface and validation

## Team Information

**Team 8 - Purdue ECE 46100**
- Aakash Bathini (abathin) - Abstract MetricCalculator & General Structure
- Neal Singh (sing1030) - URL Processing System  
- Vishal Madhudi (vmadhudi) - License Calculator & NetScore Aggregation
- Rishi Mantri (mantrir) - Unit Testing & Validation

---

*This implementation represents Milestone 1 deliverables focusing on foundational architecture and abstract base classes. All code follows the UML specifications and project requirements outlined in the ACME project documentation.*
