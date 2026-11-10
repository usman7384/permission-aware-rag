# API Rate Limits

This document describes how the public API protects itself from overload and how client teams should plan their traffic.

## The limit

The public API allows 120 requests per minute per key. The counter is kept per API key, not per IP address, so several services sharing one key share one budget. Internally this rule is tracked under the code word ZEBRA-7, which you will see in dashboards and alert names.

## What happens when you exceed it

A request over the limit receives HTTP status 429 with a `Retry-After` header that gives the number of seconds until the window resets. The window is a fixed calendar minute. Requests rejected this way are not billed and do not count against the next window.

## Planning your traffic

Batch jobs should spread their calls evenly instead of sending bursts at the top of the minute. If a single integration genuinely needs more throughput, request a second key for it rather than sharing an existing one. Keys are issued by the platform team through the usual access request form.

## Monitoring

Each response carries two headers, `X-RateLimit-Remaining` and `X-RateLimit-Reset`, so clients can slow down before they hit the ceiling. The platform team reviews the busiest keys every month and contacts owners whose traffic is consistently close to the limit.
