---
name: Synchronization problem
about: Help us to improve syncing by reporting failed syncs
title: ''
labels: ''
assignees: ''

---

**Upload a tarball with debugging information**
1. Run the command that produces the out-of-sync subtitle output, but with the additional `--make-test-case` flag, i.e.: `subsync ref.mkv -i in.srt -o failed.srt --make-test-case`
2. This results in a file of the form `ref.mkv.$timestamp.tar.gz` or similar.
3. Please upload this file using the "upload file" button at the bottom of the text prompt.

That's all! Thank you for contributing a test case; this helps me to continue improving the sync and to add additional integration tests once improvements have been made.

**Additional context**
Add any other context about the problem here that might be helpful.
