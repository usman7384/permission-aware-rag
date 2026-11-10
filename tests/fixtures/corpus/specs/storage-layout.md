# Storage Layout

This page describes where the platform keeps files that users upload, and how that storage is organised.

## The bucket

Uploaded files are kept in the bucket named atlas-blobs. Nothing else is written to that bucket, which keeps its access policy simple and its costs easy to read. The migration that moved all uploads into it was project LYNX-2.

## Object keys

Each object key starts with the tenant identifier, followed by the upload date and a random identifier. Original file names are never used in keys; they are stored as metadata on the object instead. This avoids problems with unusual characters and stops one tenant from guessing the keys of another.

## Lifecycle

Objects move to a cheaper storage class after thirty days without access. Files that a user deletes are kept for fourteen days in case the deletion was a mistake, and are then removed permanently by a lifecycle rule.

## Access

Application servers read and write through a dedicated role with access to this bucket alone. Engineers do not have direct write access. Read access for debugging is granted for one day at a time through the access request form, and every read is logged.

## Backups

A nightly job copies new objects to a second region so that a regional outage does not lose data.
