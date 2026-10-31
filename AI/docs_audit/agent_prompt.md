You are the docs critic for one review round. Read /mnt/raid5/gustavo/fastmob/AI/docs_audit/critic_prompt.md and follow its SYSTEM section exactly (rubric, evidence rules, output format).

PAGE: {{PAGE}} (role: {{ROLE}}). ROUND: {{ROUND}}.
Screenshots (view with the Read tool): /mnt/raid5/gustavo/fastmob/AI/docs_audit/runs/pages/r{{ROUND}}/{{DIR}}/ -- desktop-light-fold.png, desktop-dark-fold.png, mobile-fold.png, and *-full.png (full page; view at least desktop-light-full and mobile-full; if very tall, crop slices with Python PIL into the scratchpad).
Page Markdown: /mnt/raid5/gustavo/fastmob/docs/src/{{MD}}
Navigation tree: the `nav = [...]` block in /mnt/raid5/gustavo/fastmob/zensical.toml.
Previous round output: {{PREV}}
Judge mainly from screenshots. Do NOT edit repo files. Write your output in the required format to /mnt/raid5/gustavo/fastmob/AI/docs_audit/runs/pages/r{{ROUND}}/{{DIR}}/critic.md and also return it.
