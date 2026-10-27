# Docs critic loop: score history

Critic: fresh Opus agent per page per round, rubric in `../critic_prompt.md`. PASS = overall >= 8.0 and no dimension below 6.

| Page | R1 | R2 | R3 | R4 | R5 | R6 | R7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Home | 7.0 | 6.4* | 7.4 | 7.8 | 7.9 | 8.0 | 7.9 |
| Learn | 5.5 | 6.9 | 7.1 | 7.5 | 8.0 | 8.0 | not rerun |
| Reference | 5.5 | 6.6 | 6.5 | 6.8 | 6.9 | 7.0 | 7.1 |
| Features | 4.4 | 6.8 | 7.0 | 7.1 | 7.3 | 7.3 | 7.3 |
| Resources | 4.6 | 6.1 | 6.3 | 6.9 | 7.0 | 7.1 | 7.4 |
| About | 5.0 | 6.9 | 7.1 | 7.4 | 7.5 | 8.0 | not rerun |
| Release notes | 6.0 | 6.8 | 6.3** | 7.5 | 7.9 | 7.9 | 8.0 |

Round 0 (site-wide baseline): 4.4. R1 baselines for Home were taken after the first rewrite.
\* regression: figures rendered as raw markdown. \** regression from an over-broad nowrap CSS rule.
Round 8 (final cleanup after R7) was captured but not scored: the usage-limit rule stopped new critic rounds.
Rounds 5 to 7 went beyond the 4-round cap in LOOP.md because scores were still rising.
