# Authentication Design

This document records how users and services prove who they are when they call our systems.

## Tokens

We issue short-lived access tokens and longer-lived refresh tokens. Access tokens expire after 15 minutes. A client that holds a valid refresh token can obtain a new access token without asking the user to sign in again. The design review that settled these numbers was tracked as HERON-9.

## Why short lifetimes

A stolen access token is only useful until it expires, so a short lifetime limits the damage of a leak. Refresh tokens are stored only on the server side of our own applications and are rotated every time they are used. If a refresh token is presented twice, both copies are revoked and the user must sign in again.

## Signing

Tokens are signed with a key pair held in the secrets manager. The public half is published at a well-known address so that every service can verify tokens without calling the identity service. Signing keys rotate every ninety days, and the previous key stays valid for one extra day to cover tokens issued just before the rotation.

## Service accounts

Background services use the same token format. Their credentials are issued per service and never shared between teams.
