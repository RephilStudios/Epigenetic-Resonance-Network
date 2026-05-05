# Use NVIDIA's official PyTorch container as the base for optimal DGX compatibility
FROM nvcr.io/nvidia/pytorch:24.07-py3

# Set working directory
WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the agent API script
COPY dgx_agent_api.py .

# Create the persistent state directory inside the container
RUN mkdir -p /app/ern_state

# Expose the API port
EXPOSE 8000

# Run the application
CMD ["python", "dgx_agent_api.py"]