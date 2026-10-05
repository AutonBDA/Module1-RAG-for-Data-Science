# The public robotics demo (tools/demo_web.py) as one container for Google Cloud Run.
# Built from the folder deploy/stage.py writes - committed files at one commit, plus
# the served index - never from the working tree (deploy/README.md).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 HF_HOME=/opt/hf
WORKDIR /app

# Cloud Run has no GPU: torch's CPU build at the lock's version, then the rest of
# the Linux lock without its CUDA wheels
COPY requirements-linux.lock /tmp/requirements-linux.lock
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu \
 && grep -vE '^(torch|triton|nvidia-)' /tmp/requirements-linux.lock > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt

# the index's pinned embedding model, fetched once here; the server runs offline from it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41', device='cpu')"

COPY . /app
ENV HF_HUB_OFFLINE=1

# Cloud Run sets PORT; the Gemini key comes from the GEMINI_API_KEY secret
CMD ["sh", "-c", "exec python -B tools/demo_web.py --host 0.0.0.0 --port ${PORT:-8080}"]
