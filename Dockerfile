# GB10 Grace Blackwell requires 24.12 or later
FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dgx_agent_api.py .

RUN mkdir -p /app/ern_state

EXPOSE 8000

CMD ["python", "dgx_agent_api.py"]