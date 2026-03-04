# Use a slim Python 3.10 image as the base
FROM python:3.10-slim

# Set environment variables for Python and uv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# Install system dependencies if needed (e.g., for graphics libraries used by pdfplumber)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    # Optional: Dependencies for some complex PDF/Image libraries
    && rm -rf /var/lib/apt/lists/*

# Install uv (using pip for simplicity in this container context)
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy configuration files first to cache dependencies
COPY pyproject.toml .

# Install dependencies using uv into the system python (since we are in a container)
# We use --system flag or set UV_SYSTEM_PYTHON=1 env var
# This command installs the project dependencies listed in pyproject.toml
RUN uv pip install --system --compile-bytecode .

# Copy the rest of the application code
COPY src/ src/
COPY rubric/ rubric/
COPY tests/ tests/
# Copy README if needed for documentation tools, but optional for runtime
COPY README.md .

# Install the project itself in editable mode if desired, or just copy src
# Here we install it so 'src' packages are importable
RUN uv pip install --system --compile-bytecode --no-deps -e .

# Default command (can be overridden)
# Example: Run tests by default to verify image
CMD ["pytest", "tests/"]
