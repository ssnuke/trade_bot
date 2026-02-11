# Render Deployment (Free Tier)

Render Free tier is fine for demos but it is not 24/7. Services sleep on inactivity and have limited free hours.

This guide deploys two services:
- Bot API service (private or public)
- Dashboard web service

Auto-deploy is enabled, so pushes to GitHub trigger a rebuild and deploy.

You can use the Render Blueprint in [render.yaml](../render.yaml) for one-click setup.

## 1) Create the Bot Service

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

## 2) Create the Dashboard Service

- New > Web Service
- Connect repo
- Environment: Docker
- Dockerfile Path: `Dockerfile.dashboard`
- Name: `delta-bot-dashboard`
- Instance Type: Free

Environment variables:

- `BOT_API_URL=https://delta-bot-worker.onrender.com`

Deploy and open the dashboard URL. The Flask app uses the `PORT` env that Render provides.

## 3) Blueprint Option (render.yaml)

If you prefer infra-as-code:

- In Render, use "New > Blueprint" and select this repo.
- Review and apply the services from [render.yaml](../render.yaml).
- Update `BOT_API_URL` if you rename the bot service.

## Notes

- Free services sleep. Your bot will not run continuously.
- If you need 24/7, switch to a paid plan or a VM provider.
- Data in containers is ephemeral; use external storage if you need persistence.
