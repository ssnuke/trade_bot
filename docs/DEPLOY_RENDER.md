# Render Deployment (Free Tier)

Render Free tier is fine for demos but it is not 24/7. Services sleep on inactivity and have limited free hours.

This guide deploys a single service that runs the bot and serves the dashboard.

Auto-deploy is enabled, so pushes to GitHub trigger a rebuild and deploy.

You can use the Render Blueprint in [render.yaml](../render.yaml) for one-click setup.

## 1) Create the Service

In Render:

- New > Web Service
- Connect repo
- Environment: Docker
- Dockerfile Path: `Dockerfile.bot`
- Name: `delta-bot-worker`
- Region: closest to you
- Instance Type: Free

Environment variables:

- `DELTA_API_KEY`
- `DELTA_API_SECRET`
- `DELTA_BASE_URL=https://api.india.delta.exchange`
- `PAPER_TRADING=True`

Deploy the service and note its public URL, e.g.:

```
https://delta-bot-worker.onrender.com
```

## 2) Open the Dashboard

Open the same service URL in your browser. The dashboard is served at `/`.

## 3) Blueprint Option (render.yaml)

If you prefer infra-as-code:

- In Render, use "New > Blueprint" and select this repo.
- Review and apply the service from [render.yaml](../render.yaml).

## Notes

- Free services sleep. Your bot will not run continuously.
- If you need 24/7, switch to a paid plan or a VM provider.
- Data in containers is ephemeral; use external storage if you need persistence.
