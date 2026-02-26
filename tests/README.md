### Install Dependencies
```bash
pip install pytest pytest-cov pytest-mock hypothesis
```

### Running Tests

To run all tests:
```bash
PYTHONPATH=. pytest
```

To run only unit tests:
```bash
PYTHONPATH=. pytest tests/unit
```

To run only integration tests:
```bash
PYTHONPATH=. pytest tests/integration
```

Tests are designed to run on CPU and use mocks for heavy dependencies like Ray and DeepSpeed.
