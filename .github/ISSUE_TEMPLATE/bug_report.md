---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''

---

**Environment (please complete the following information):**
 - OS: [e.g. Windows 10, MacOS Mojave, etc.]
 - python version (`python --version`)
 - subsync version (`subsync --version`)

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
How to reproduce the behavior.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Output**
Copy+paste stdout from running the command here.

**Test case**
[Optional] You can bundle additional debugging information into a tar archive as follows:
```
subsync vid.mkv -i in.srt -o out.srt --make-test-case
```
This will create a file `vid.mkv.$timestamp.tar.gz` or similar a few KiB in size; you can attach it by clicking the "attach files" button below.

**Additional context**
Add any other context about the problem here.
