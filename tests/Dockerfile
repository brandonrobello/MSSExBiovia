# Use a slim Python base image
FROM python:3.11-slim

WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies FIRST
# This is stable and cacheable
COPY tests/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest

# Now copy the rest of the project
# This invalidates the cache only when code changes
COPY . .

# Let Python find your modules
ENV PYTHONPATH="/app"

# Run tests
CMD ["pytest", "-v", "tests/test_functionality.py"]
