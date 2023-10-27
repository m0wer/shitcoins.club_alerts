# shitcoins.club Telegram Bot for alerts

Script that parses the prices of https://shitcoins.club/, saves them in a csv file and sends notifications to a telegram channel when the comission is below a certain threshold.

## Run it

```bash
docker run \
  -e TG_TOKEN=something \
  -e TG_CHANNEL_ID=@your_channel_or_chat_id \
  -v /path/to/your/data:/data \
  -it \
  --rm \
  --name shitcoins-club-alerts \
  ghcr.io/m0wer/shitcoins.club_alerts:master
```

Add it to a cron or something:

```
0 * * * * docker run ...
```
