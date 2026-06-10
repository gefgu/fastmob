"""Convert Sphinx-style docstring math to Arithmatex-compatible Markdown."""

from __future__ import annotations

import re

from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor


INLINE_MATH_RE = re.compile(r":math:`([^`]+)`")
FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})")


class RstMathPreprocessor(Preprocessor):
    """Translate common RST math constructs before Markdown parsing."""

    def run(self, lines: list[str]) -> list[str]:
        output: list[str] = []
        in_fence = False
        fence_marker = ""
        i = 0

        while i < len(lines):
            line = lines[i]
            fence_match = FENCE_RE.match(line)
            if fence_match:
                marker = fence_match.group(2)
                if not in_fence:
                    in_fence = True
                    fence_marker = marker[0]
                elif marker.startswith(fence_marker * 3):
                    in_fence = False
                    fence_marker = ""
                output.append(line)
                i += 1
                continue

            if in_fence:
                output.append(line)
                i += 1
                continue

            stripped = line.lstrip()
            directive_indent = len(line) - len(stripped)
            if stripped == ".. math::":
                converted, next_index = self._convert_display_math(lines, i, directive_indent)
                if converted:
                    output.extend(converted)
                    i = next_index
                    continue

            output.append(INLINE_MATH_RE.sub(r"\\(\1\\)", line))
            i += 1

        return output

    def _convert_display_math(
        self,
        lines: list[str],
        directive_index: int,
        directive_indent: int,
    ) -> tuple[list[str] | None, int]:
        i = directive_index + 1
        while i < len(lines) and not lines[i].strip():
            i += 1

        body_start = i
        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                body.append("")
                i += 1
                continue

            indent = len(line) - len(line.lstrip())
            if indent <= directive_indent:
                break

            body.append(line)
            i += 1

        while body and not body[-1].strip():
            body.pop()

        if not body:
            return None, directive_index + 1

        content_indents = [len(line) - len(line.lstrip()) for line in body if line.strip()]
        strip_indent = min(content_indents) if content_indents else directive_indent + 4
        formula = [line[strip_indent:] if len(line) >= strip_indent else "" for line in body]
        converted = ["\\[", *formula, "\\]"]
        if directive_index > 0 and lines[directive_index - 1].strip():
            converted.insert(0, "")
        if i < len(lines) and lines[i].strip():
            converted.append("")

        return converted, max(i, body_start)


class RstMathExtension(Extension):
    """Markdown extension entry point."""

    def extendMarkdown(self, md):
        md.preprocessors.register(RstMathPreprocessor(md), "rst_math", 35)


def makeExtension(**kwargs):
    return RstMathExtension(**kwargs)
