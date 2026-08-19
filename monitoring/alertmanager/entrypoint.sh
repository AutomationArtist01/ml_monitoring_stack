#!/bin/sh
# Render env vars into the config (Alertmanager has no native env substitution), then start.
set -e
: "${SMTP_HOST:=mailhog}"; : "${SMTP_PORT:=1025}"; : "${SMTP_FROM:=alerts@team4.local}"
: "${ALERT_EMAIL_TO:=oncall@team4.local}"; : "${SLACK_WEBHOOK_URL:=http://webhook-catcher:8091/slack}"
: "${SMTP_USER:=}"; : "${SMTP_PASSWORD:=}"; : "${SMTP_TLS:=false}"
export SMTP_HOST SMTP_PORT SMTP_FROM ALERT_EMAIL_TO SLACK_WEBHOOK_URL SMTP_USER SMTP_PASSWORD SMTP_TLS
sed -e "s|\${SMTP_HOST}|$SMTP_HOST|g" -e "s|\${SMTP_PORT}|$SMTP_PORT|g" -e "s|\${SMTP_FROM}|$SMTP_FROM|g" \
    -e "s|\${ALERT_EMAIL_TO}|$ALERT_EMAIL_TO|g" -e "s|\${SLACK_WEBHOOK_URL}|$SLACK_WEBHOOK_URL|g" \
    -e "s|\${SMTP_USER}|$SMTP_USER|g" -e "s|\${SMTP_PASSWORD}|$SMTP_PASSWORD|g" -e "s|\${SMTP_TLS}|$SMTP_TLS|g" \
    /etc/alertmanager/alertmanager.yml > /tmp/alertmanager.yml
exec /bin/alertmanager --config.file=/tmp/alertmanager.yml --storage.path=/alertmanager "$@"
