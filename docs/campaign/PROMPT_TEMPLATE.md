# Claude Session Prompt Template

Use this template if you want to create more sessions after the seven included ones.

## Structure

1. Title
2. Preferred Claude mode
3. Goal
4. Read these first
5. Constraints
6. Required workflow
7. Deliverables
8. Definition of done
9. Commit message

## Template

```text
# <Session title>

Preferred mode: <Planning or Execution>

Use `/research-engineer` if it exists locally and helps with the analysis portions. If it is unavailable, proceed normally.

Goal:
- <one sentence outcome>

Read these first:
- @README.md
- @train_gpt.py
- @<other critical files>

Constraints:
- Work inside the existing repo.
- Do not broaden scope beyond this session goal.
- Prefer local repo evidence over internet claims unless a live external fact is required.
- For execution sessions, keep experiments self-contained under `records/track_non_record_16mb/<today>_<tag>/`, where `<today>` is the current date in `YYYY-MM-DD` format.

Required workflow:
1. Inspect the targeted files and summarize the current state before changing anything.
2. State the exact plan for this session in a few bullets.
3. Execute the scoped work only.
4. Record outcomes, measurements, blockers, and next steps in repo files.
5. Commit the session changes with the message listed below.

Deliverables:
- <artifact 1>
- <artifact 2>
- <artifact 3>

Definition of done:
- <completion criterion 1>
- <completion criterion 2>
- <completion criterion 3>

Commit message:
- `<type>(campaign): <summary>`
```
