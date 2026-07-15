#!/bin/bash
while true; do
  scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@10.6.131.9:/raid/davilu/live_metrics.json /Users/davilu/claude_code/live_monitor/metrics.json 2>/dev/null
  scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 root@10.6.131.9:/raid/davilu/live_history.json /Users/davilu/claude_code/live_monitor/history.json 2>/dev/null
  sleep 5
done
