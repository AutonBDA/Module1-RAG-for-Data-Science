# Public demonstration on Google Cloud Run

The robotics demonstration page (`tools/demo_web.py`) served to the public:
https://rag4ds-robotics-181763845696.asia-northeast1.run.app

## What goes into the container

`deploy/stage.py` writes the build folder outside the repository: the demo's committed
files at one commit (the library under `src/`, `tools/demo_web.py`, `tools/answer_writer.py`,
the robotics pack's manifest and demonstration questions, the Linux lock) taken with
`git archive`, plus a built index from `data/index`, checked against its own manifest.
Nothing from the working tree, no corpus text, no key. The Dockerfile installs torch's CPU
build and the rest of the lock, fetches the index's pinned embedding model, and runs the
page with `--host 0.0.0.0` on Cloud Run's `PORT`.

## How it runs

- One container at most (`--max-instances 1`), so the page's own bounds hold: 16 requests
  at once, each arriving whole within 15 seconds, and at most 60 written answers an hour.
- The Gemini key is a Secret Manager secret given to the container as `GEMINI_API_KEY`; it
  is never in the image, the page or a log.
- Region asia-northeast1 (Tokyo). With no visitors the service scales to zero; the first
  visitor after that waits while the model loads.

## Commands

```
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com
gcloud secrets create gemini-api-key --replication-policy=automatic --data-file=-   # key on stdin
gcloud secrets add-iam-policy-binding gemini-api-key \
  --member=serviceAccount:<project number>-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
gcloud projects add-iam-policy-binding <project> \
  --member=serviceAccount:<project number>-compute@developer.gserviceaccount.com --role=roles/run.builder

python tools/build_index.py --out data/index
python deploy/stage.py --out <build folder outside the repository> --commit <commit>
docker build -t rag4ds-demo <build folder>          # local test first
gcloud run deploy rag4ds-robotics --source <build folder> --region asia-northeast1 \
  --allow-unauthenticated --max-instances 1 --concurrency 16 --memory 2Gi --cpu 1 \
  --timeout 120 --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

To take the page down: `gcloud run services delete rag4ds-robotics --region asia-northeast1`.
