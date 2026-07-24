"""Repository policy tests for MCP-backed project skills."""

from __future__ import annotations

from pathlib import Path
import re
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKILLS_ROOT = REPOSITORY_ROOT / ".agents" / "skills"


class SkillMCPPolicyTests(unittest.TestCase):
    def test_every_project_skill_declares_an_mcp_dependency(self) -> None:
        skill_directories = sorted(
            path.parent for path in SKILLS_ROOT.glob("*/SKILL.md")
        )
        self.assertTrue(skill_directories)

        for skill_directory in skill_directories:
            with self.subTest(skill=skill_directory.name):
                metadata_path = skill_directory / "agents" / "openai.yaml"
                self.assertTrue(
                    metadata_path.is_file(),
                    f"{skill_directory.name} has no agents/openai.yaml",
                )
                metadata = metadata_path.read_text(encoding="utf-8")
                self.assertIn("dependencies:", metadata)
                self.assertIn("tools:", metadata)
                self.assertRegex(
                    metadata,
                    re.compile(r'^\s*-\s+type:\s+"mcp"\s*$', re.MULTILINE),
                )

    def test_every_project_skill_fails_closed_when_mcp_is_unavailable(
        self,
    ) -> None:
        for skill_path in sorted(SKILLS_ROOT.glob("*/SKILL.md")):
            with self.subTest(skill=skill_path.parent.name):
                instructions = skill_path.read_text(encoding="utf-8").lower()
                self.assertIn("mcp", instructions)
                self.assertIn("unavailable", instructions)
                self.assertIn("stop", instructions)


if __name__ == "__main__":
    unittest.main()
