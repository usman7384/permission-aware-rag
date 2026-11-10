# Deploy Process

This page explains when and how code reaches production, and who is responsible at each stage.

## Deploy windows

Production deploys happen on Tuesdays and Thursdays. Outside those two days the production pipeline is locked, except for emergency fixes approved by the on-call lead. The release train is known internally as OTTER-3, and the scheduling bot posts its reminders under that name.

## Before the window

Every change must be merged to the main branch and pass the full test suite at least four hours before the window opens. The release captain collects the merged changes into a release note and checks that each one has a linked ticket and a named reviewer.

## During the window

The release captain starts the pipeline at ten in the morning. The pipeline deploys to a canary group of servers first and watches error rates and latency for thirty minutes. If the canary looks healthy, the rest of the fleet is updated in three waves.

## Rolling back

Any engineer may stop a deploy by pressing the halt button in the pipeline. A rollback restores the previous release within five minutes. After a rollback, the release captain writes a short incident summary and the change waits for the next scheduled window.

## Afterwards

The release note is published in the engineering channel once the last wave completes.
